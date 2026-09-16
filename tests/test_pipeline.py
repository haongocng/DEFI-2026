import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from drap.agents import Analyst, AnalystOutputError, Reasoner, ReasonerOutputError
from drap.config import LifecycleConfig, RetrievalConfig, UtilityConfig
from drap.embeddings import HashingEmbedder
from drap.memory import RuleMemory
from drap.models import Candidate, DatasetSample, Rule, SessionFeedback, SessionInput
from drap.pipeline import DRAPPipeline
from drap.regime import RegimeConfig, RegimeDetector
from drap.retrieval import RuleRetriever


class FakeClient:
    def __init__(self) -> None:
        self.reasoner_calls = 0
        self.analyst_calls = 0

    def complete(self, prompt: str) -> str:
        if prompt.startswith("You are the Reasoner"):
            self.reasoner_calls += 1
            # The second calibration sample must observe the rule induced by the first.
            selected = 1 if "exact recent category" in prompt else 2
            other = 2 if selected == 1 else 1
            return json.dumps(
                {
                    "selected_intent": "recent category",
                    "ranked_candidate_ids": [selected, other],
                }
            )
        if prompt.startswith("You are the Analyst"):
            self.analyst_calls += 1
            return json.dumps(
                {
                    "condition": "The user repeats a recent category",
                    "instruction": "Prefer the exact recent category",
                    "diagnosis": "Recent intent was underweighted",
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt[:80]}")


def sample(sample_id: str) -> DatasetSample:
    return DatasetSample(
        session=SessionInput(
            sample_id=sample_id,
            history=("Horror Game",),
            candidates=(
                Candidate(1, "Horror Sequel"),
                Candidate(2, "Cooking Game"),
            ),
        ),
        feedback=SessionFeedback(target_id=1, target_text="Horror Sequel"),
    )


class PipelineTests(unittest.TestCase):
    def test_calibration_updates_memory_before_next_sample_and_evaluation_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            retriever = RuleRetriever(HashingEmbedder(dimensions=1024))
            memory = RuleMemory(Path(directory) / "rules.json", retriever)
            detector = RegimeDetector(
                retriever.embedder, RegimeConfig(n_clusters=1)
            ).fit([sample("fit").session])
            analyst_event_path = Path(directory) / "analyst_events.jsonl"
            pipeline = DRAPPipeline(
                Reasoner(client),
                Analyst(client),
                memory,
                retriever,
                detector,
                success_k=1,
                analyst_event_path=analyst_event_path,
            )

            records = pipeline.calibrate([sample("first"), sample("second")])
            self.assertFalse(records[0].success)
            self.assertTrue(records[1].success)
            self.assertEqual(records[0].target_rank, 2)
            self.assertEqual(records[1].target_rank, 1)
            self.assertEqual(client.analyst_calls, 1)
            self.assertEqual(len(memory.active_rules()), 1)
            event = json.loads(
                analyst_event_path.read_text(encoding="utf-8").strip()
            )
            self.assertEqual(event["diagnosis"], "Recent intent was underweighted")
            self.assertEqual(event["maintenance_action"], "ADD")
            self.assertEqual(
                event["proposed_rule_id"], memory.active_rules()[0].id
            )
            self.assertNotIn("diagnosis", memory.active_rules()[0].to_dict())

            before = [rule.to_dict() for rule in memory.rules]
            events_before = analyst_event_path.read_text(encoding="utf-8")
            metrics, evaluation = pipeline.evaluate([sample("test")])
            after = [rule.to_dict() for rule in memory.rules]
            self.assertEqual(before, after)
            self.assertEqual(
                analyst_event_path.read_text(encoding="utf-8"), events_before
            )
            self.assertTrue(evaluation[0].success)
            self.assertEqual(metrics["HR@1"], 1.0)

    def test_reasoner_failure_writes_bounded_error_artifact(self) -> None:
        class InvalidClient:
            def complete(self, prompt: str) -> str:
                return "invalid json"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retriever = RuleRetriever(HashingEmbedder())
            detector = RegimeDetector(
                retriever.embedder, RegimeConfig(n_clusters=1)
            ).fit([sample("fit").session])
            error_path = root / "errors.jsonl"
            pipeline = DRAPPipeline(
                Reasoner(InvalidClient(), max_attempts=2),
                Analyst(InvalidClient()),
                RuleMemory(root / "rules.json", retriever),
                retriever,
                detector,
                error_path=error_path,
            )
            with self.assertRaises(ReasonerOutputError):
                pipeline.infer(sample("broken"))
            artifact = json.loads(error_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["stage"], "reasoner")
            self.assertEqual(artifact["session_id"], "broken")
            self.assertEqual(len(artifact["responses"]), 2)

    def test_analyst_failure_writes_bounded_error_artifact(self) -> None:
        class FailedAnalystClient:
            def complete(self, prompt: str) -> str:
                if prompt.startswith("You are the Reasoner"):
                    return json.dumps(
                        {
                            "selected_intent": "wrong intent",
                            "ranked_candidate_ids": [2, 1],
                        }
                    )
                return "invalid analyst json"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = FailedAnalystClient()
            retriever = RuleRetriever(HashingEmbedder())
            memory = RuleMemory(root / "rules.json", retriever)
            exposed_rule = Rule(
                "recent horror interaction",
                "prefer horror continuations",
                regime_id=0,
            )
            memory.add_rule(exposed_rule, current_step=0)
            memory.save()
            memory_before = memory.path.read_text(encoding="utf-8")
            detector = RegimeDetector(
                retriever.embedder, RegimeConfig(n_clusters=1)
            ).fit([sample("fit").session])
            error_path = root / "errors.jsonl"
            pipeline = DRAPPipeline(
                Reasoner(client),
                Analyst(client, max_attempts=2),
                memory,
                retriever,
                detector,
                success_k=1,
                error_path=error_path,
                analyst_event_path=root / "analyst_events.jsonl",
            )
            with self.assertRaises(AnalystOutputError):
                pipeline.calibrate([sample("broken-analyst")])
            artifact = json.loads(error_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["stage"], "analyst")
            self.assertEqual(len(artifact["responses"]), 2)
            self.assertEqual(exposed_rule.num_used, 0)
            self.assertEqual(exposed_rule.num_success, 0)
            self.assertEqual(memory.path.read_text(encoding="utf-8"), memory_before)
            self.assertFalse((root / "analyst_events.jsonl").exists())

    def test_shared_outcome_credit_updates_every_retrieved_rule(self) -> None:
        class SuccessClient:
            def complete(self, prompt: str) -> str:
                if prompt.startswith("You are the Reasoner"):
                    return json.dumps(
                        {
                            "selected_intent": "horror continuation",
                            "ranked_candidate_ids": [1, 2],
                        }
                    )
                raise AssertionError("Analyst must not run on success")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retriever = RuleRetriever(
                HashingEmbedder(),
                RetrievalConfig(stm_k=2, ltm_k=0, total_k=2),
            )
            memory = RuleMemory(root / "rules.json", retriever)
            rules = [
                Rule("recent horror", "prefer horror", regime_id=0),
                Rule("game continuation", "prefer sequels", regime_id=0),
            ]
            for rule in rules:
                memory.add_rule(rule, current_step=0)
            detector = RegimeDetector(
                retriever.embedder, RegimeConfig(n_clusters=1)
            ).fit([sample("fit").session])
            pipeline = DRAPPipeline(
                Reasoner(SuccessClient()),
                Analyst(SuccessClient()),
                memory,
                retriever,
                detector,
                success_k=10,
            )
            records = pipeline.calibrate([sample("shared-credit")])
            self.assertTrue(records[0].success)
            self.assertEqual(len(records[0].retrieved_rule_ids), 2)
            self.assertTrue(all(rule.num_used == 1 for rule in rules))
            self.assertTrue(all(rule.num_success == 1 for rule in rules))

    def test_success_k_ten_boundary_controls_analyst_trigger(self) -> None:
        class BoundaryClient:
            def __init__(self) -> None:
                self.reasoner_calls = 0
                self.analyst_calls = 0

            def complete(self, prompt: str) -> str:
                if prompt.startswith("You are the Reasoner"):
                    self.reasoner_calls += 1
                    ranking = (
                        list(range(1, 10)) + [11, 10]
                        if self.reasoner_calls == 1
                        else list(range(1, 12))
                    )
                    return json.dumps(
                        {
                            "selected_intent": "fixture intent",
                            "ranked_candidate_ids": ranking,
                        }
                    )
                self.analyst_calls += 1
                return json.dumps(
                    {
                        "diagnosis": "the target pattern was underweighted",
                        "condition": "a reusable underweighted pattern",
                        "instruction": "prioritize the matching continuation",
                    }
                )

        candidates = tuple(
            Candidate(candidate_id, f"Candidate {candidate_id}")
            for candidate_id in range(1, 12)
        )

        def boundary_sample(sample_id: str) -> DatasetSample:
            return DatasetSample(
                SessionInput(sample_id, ("History",), candidates),
                SessionFeedback(11, "Candidate 11"),
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = BoundaryClient()
            retriever = RuleRetriever(HashingEmbedder())
            detector = RegimeDetector(
                retriever.embedder, RegimeConfig(n_clusters=1)
            ).fit([boundary_sample("fit").session])
            pipeline = DRAPPipeline(
                Reasoner(client),
                Analyst(client),
                RuleMemory(root / "rules.json", retriever),
                retriever,
                detector,
                success_k=10,
                analyst_event_path=root / "analyst_events.jsonl",
            )
            records = pipeline.calibrate(
                [boundary_sample("rank-10"), boundary_sample("rank-11")]
            )
            self.assertEqual([record.target_rank for record in records], [10, 11])
            self.assertEqual([record.success for record in records], [True, False])
            self.assertEqual(client.analyst_calls, 1)
            self.assertEqual(
                len(
                    (root / "analyst_events.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                1,
            )

    def test_analyst_event_records_replacement_decision(self) -> None:
        class ReplacementClient:
            def complete(self, prompt: str) -> str:
                if prompt.startswith("You are the Reasoner"):
                    return json.dumps(
                        {
                            "selected_intent": "wrong intent",
                            "ranked_candidate_ids": [2, 1],
                        }
                    )
                return json.dumps(
                    {
                        "diagnosis": "the old guidance underweighted recency",
                        "condition": "repeated recent category",
                        "instruction": "prefer the most recent category",
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = ReplacementClient()
            retriever = RuleRetriever(HashingEmbedder())
            memory = RuleMemory(root / "rules.json", retriever)
            old = Rule(
                "repeated recent category",
                "prefer the oldest category",
                regime_id=0,
            )
            memory.add_rule(old, current_step=0)
            detector = RegimeDetector(
                retriever.embedder, RegimeConfig(n_clusters=1)
            ).fit([sample("fit").session])
            event_path = root / "analyst_events.jsonl"
            pipeline = DRAPPipeline(
                Reasoner(client),
                Analyst(client),
                memory,
                retriever,
                detector,
                success_k=1,
                analyst_event_path=event_path,
            )
            pipeline.calibrate([sample("replace")])
            event = json.loads(event_path.read_text(encoding="utf-8"))
            self.assertEqual(event["maintenance_action"], "REPLACE")
            self.assertEqual(event["affected_rule_ids"], [old.id])
            self.assertFalse(old.active)
            self.assertEqual(old.replaced_by, event["proposed_rule_id"])

    def test_analyst_event_is_kept_when_duplicate_rule_is_ignored(self) -> None:
        class DuplicateClient:
            def complete(self, prompt: str) -> str:
                if prompt.startswith("You are the Reasoner"):
                    return json.dumps(
                        {
                            "selected_intent": "wrong intent",
                            "ranked_candidate_ids": [2, 1],
                        }
                    )
                return json.dumps(
                    {
                        "diagnosis": "the ranking still missed the target pattern",
                        "condition": "repeated recent category",
                        "instruction": "prefer the most recent category",
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = DuplicateClient()
            retriever = RuleRetriever(HashingEmbedder())
            memory = RuleMemory(root / "rules.json", retriever)
            existing = Rule(
                "repeated recent category",
                "prefer the most recent category",
                regime_id=0,
            )
            memory.add_rule(existing, current_step=0)
            detector = RegimeDetector(
                retriever.embedder, RegimeConfig(n_clusters=1)
            ).fit([sample("fit").session])
            event_path = root / "analyst_events.jsonl"
            DRAPPipeline(
                Reasoner(client),
                Analyst(client),
                memory,
                retriever,
                detector,
                success_k=1,
                analyst_event_path=event_path,
            ).calibrate([sample("duplicate")])
            event = json.loads(event_path.read_text(encoding="utf-8"))
            self.assertEqual(event["maintenance_action"], "IGNORE")
            self.assertEqual(event["affected_rule_ids"], [existing.id])
            self.assertEqual(len(memory.active_rules()), 1)
            self.assertTrue(existing.active)

    def test_failed_memory_commit_restores_in_memory_state(self) -> None:
        class SuccessClient:
            def complete(self, prompt: str) -> str:
                return json.dumps(
                    {
                        "selected_intent": "horror continuation",
                        "ranked_candidate_ids": [1, 2],
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retriever = RuleRetriever(HashingEmbedder())
            memory = RuleMemory(root / "rules.json", retriever)
            rule = Rule("recent horror", "prefer horror", regime_id=0)
            memory.add_rule(rule, current_step=0)
            detector = RegimeDetector(
                retriever.embedder, RegimeConfig(n_clusters=1)
            ).fit([sample("fit").session])
            pipeline = DRAPPipeline(
                Reasoner(SuccessClient()),
                Analyst(SuccessClient()),
                memory,
                retriever,
                detector,
            )
            with patch.object(memory, "save", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    pipeline.calibrate([sample("rollback")])
            restored = memory.by_id(rule.id)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.num_used, 0)
            self.assertEqual(restored.num_success, 0)
            self.assertEqual(memory.current_step, 0)

    def test_failure_session_applies_lifecycle_once(self) -> None:
        class FailureClient:
            def complete(self, prompt: str) -> str:
                if prompt.startswith("You are the Reasoner"):
                    return json.dumps(
                        {
                            "selected_intent": "wrong intent",
                            "ranked_candidate_ids": [2, 1],
                        }
                    )
                return json.dumps(
                    {
                        "diagnosis": "the recent preference was underweighted",
                        "condition": "a novel recent preference pattern",
                        "instruction": "prioritize matching recent continuations",
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            utility = UtilityConfig(alpha=1.0, beta=0.0, gamma=0.0)
            retriever = RuleRetriever(
                HashingEmbedder(),
                RetrievalConfig(
                    stm_k=0,
                    ltm_k=1,
                    total_k=1,
                    utility=utility,
                ),
            )
            lifecycle = LifecycleConfig(
                minimum_uses=1,
                promotion_success=0.9,
                promotion_utility=0.9,
                demotion_utility=0.4,
                deletion_utility=0.2,
                stale_window=100,
            )
            memory = RuleMemory(
                root / "rules.json",
                retriever,
                lifecycle=lifecycle,
                utility=utility,
            )
            weak_ltm = Rule(
                "legacy recent preference",
                "follow the legacy preference",
                regime_id=0,
                memory_type="LTM",
                num_used=10,
                num_success=0,
                last_used_step=1,
            )
            memory.rules.append(weak_ltm)
            detector = RegimeDetector(
                retriever.embedder, RegimeConfig(n_clusters=1)
            ).fit([sample("fit").session])
            pipeline = DRAPPipeline(
                Reasoner(FailureClient()),
                Analyst(FailureClient()),
                memory,
                retriever,
                detector,
                success_k=1,
            )

            pipeline.calibrate([sample("single-lifecycle")])

            self.assertTrue(weak_ltm.active)
            self.assertEqual(weak_ltm.memory_type, "STM")

    def test_analyst_event_failure_rolls_back_rules_and_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retriever = RuleRetriever(HashingEmbedder())
            memory = RuleMemory(root / "rules.json", retriever)
            memory.save()
            rules_before = memory.path.read_bytes()
            detector = RegimeDetector(
                retriever.embedder, RegimeConfig(n_clusters=1)
            ).fit([sample("fit").session])
            pipeline = DRAPPipeline(
                Reasoner(FakeClient()),
                Analyst(FakeClient()),
                memory,
                retriever,
                detector,
                success_k=1,
                analyst_event_path=root / "analyst_events.jsonl",
                calibration_prediction_path=root / "calibration_predictions.jsonl",
            )

            with patch.object(
                pipeline,
                "_record_analyst_event",
                side_effect=OSError("audit disk failure"),
            ):
                with self.assertRaises(OSError):
                    pipeline.calibrate([sample("audit-rollback")])

            self.assertEqual(memory.path.read_bytes(), rules_before)
            self.assertEqual(memory.current_step, 0)
            self.assertEqual(memory.rules, [])
            self.assertFalse((root / "analyst_events.jsonl").exists())
            self.assertFalse((root / "calibration_predictions.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
