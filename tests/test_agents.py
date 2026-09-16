import json
from pathlib import Path
import unittest
from unittest.mock import patch

from drap.agents import Analyst, AnalystOutputError, Reasoner, ReasonerOutputError
from drap.models import (
    BehavioralRegime,
    Candidate,
    RankedOutput,
    Rule,
    SessionFeedback,
    SessionInput,
)
from drap.prompts import DEFAULT_BASE_PROMPT_PATH, analyst_prompt, load_base_prompt


class StaticClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.prompt = ""

    def complete(self, prompt: str) -> str:
        self.prompt = prompt
        return json.dumps(self.payload)


def fixture_sample() -> SessionInput:
    return SessionInput(
        sample_id="sample",
        history=("HDMI cable", "home theater receiver"),
        candidates=(Candidate(1, "HDMI adapter"), Candidate(2, "camera bag")),
    )


class ReasonerTests(unittest.TestCase):
    def test_full_permutation_and_prompt_hides_rule_metadata(self) -> None:
        client = StaticClient(
            {
                "selected_intent": "connect home theater video equipment",
                "ranked_candidate_ids": [1, 2],
            }
        )
        rule = Rule(
            "recent interactions concern video connectivity",
            "prioritize compatible display adapters",
            regime_id=0,
            memory_type="LTM",
        )
        output = Reasoner(client).rank(
            fixture_sample(), [rule], BehavioralRegime(0, "focused/consistent")
        )
        self.assertEqual(output.ranked_candidate_ids, (1, 2))
        self.assertEqual(output.selected_intent, "connect home theater video equipment")
        self.assertNotIn(rule.id, client.prompt)
        self.assertNotIn("memory_type", client.prompt)
        self.assertNotIn("utility", client.prompt.casefold())
        self.assertNotIn("focused/consistent", client.prompt)

    def test_regime_name_is_explicitly_opt_in(self) -> None:
        client = StaticClient(
            {"selected_intent": "intent", "ranked_candidate_ids": [1, 2]}
        )
        Reasoner(client, include_regime_name=True).rank(
            fixture_sample(), [], BehavioralRegime(0, "focused/consistent")
        )
        self.assertIn("focused/consistent", client.prompt)

    def test_missing_candidate_id_is_rejected(self) -> None:
        client = StaticClient(
            {"selected_intent": "intent", "ranked_candidate_ids": [1]}
        )
        with self.assertRaises(ReasonerOutputError):
            Reasoner(client).rank(
                fixture_sample(), [], BehavioralRegime(0, "focused/consistent")
            )

    def test_non_integer_candidate_ids_are_rejected(self) -> None:
        for bad_id in (1.0, True, "1"):
            with self.subTest(bad_id=bad_id):
                client = StaticClient(
                    {
                        "selected_intent": "intent",
                        "ranked_candidate_ids": [bad_id, 2],
                    }
                )
                with self.assertRaises(ReasonerOutputError):
                    Reasoner(client, max_attempts=1).rank(
                        fixture_sample(), [], BehavioralRegime(0, "hidden")
                    )

    def test_selected_intent_must_be_a_non_empty_json_string(self) -> None:
        client = StaticClient(
            {"selected_intent": 123, "ranked_candidate_ids": [1, 2]}
        )
        with self.assertRaises(ReasonerOutputError):
            Reasoner(client, max_attempts=1).rank(
                fixture_sample(), [], BehavioralRegime(0, "hidden")
            )

    def test_invalid_json_is_repaired_once(self) -> None:
        class RepairClient:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def complete(self, prompt: str) -> str:
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    return "not json"
                return json.dumps(
                    {"selected_intent": "intent", "ranked_candidate_ids": [1, 2]}
                )

        client = RepairClient()
        output = Reasoner(client).rank(
            fixture_sample(), [], BehavioralRegime(0, "hidden")
        )
        self.assertEqual(output.ranked_candidate_ids, (1, 2))
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("[OUTPUT REPAIR]", client.prompts[1])

    def test_duplicate_only_ranking_is_normalized_without_llm_repair(self) -> None:
        class DuplicateClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, prompt: str) -> str:
                self.calls += 1
                return json.dumps(
                    {
                        "selected_intent": "fixture intent",
                        "ranked_candidate_ids": [3, 1, 3, 2],
                    }
                )

        session = SessionInput(
            sample_id="duplicate-only",
            history=("recent item",),
            candidates=tuple(Candidate(index, str(index)) for index in (1, 2, 3, 4)),
        )
        client = DuplicateClient()
        output = Reasoner(client).rank(
            session, [], BehavioralRegime(0, "hidden")
        )

        self.assertEqual(client.calls, 1)
        self.assertEqual(output.ranked_candidate_ids, (3, 1, 2, 4))
        self.assertIsNotNone(output.ranking_normalization)
        normalization = output.ranking_normalization
        assert normalization is not None
        self.assertEqual(normalization.duplicate_candidate_ids, (3,))
        self.assertEqual(normalization.missing_candidate_ids, (4,))
        self.assertEqual(
            normalization.policy, "stable_first_seen_append_missing"
        )

    def test_legacy_duplicate_policy_still_uses_bounded_llm_repair(self) -> None:
        class DuplicateThenValidClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, prompt: str) -> str:
                self.calls += 1
                ranking = [1, 1] if self.calls == 1 else [1, 2]
                return json.dumps(
                    {
                        "selected_intent": "fixture intent",
                        "ranked_candidate_ids": ranking,
                    }
                )

        client = DuplicateThenValidClient()
        output = Reasoner(client, duplicate_policy="llm_repair").rank(
            fixture_sample(), [], BehavioralRegime(0, "hidden")
        )
        self.assertEqual(client.calls, 2)
        self.assertEqual(output.ranked_candidate_ids, (1, 2))
        self.assertIsNone(output.ranking_normalization)

    def test_unknown_duplicate_id_is_not_locally_normalized(self) -> None:
        client = StaticClient(
            {
                "selected_intent": "fixture intent",
                "ranked_candidate_ids": [99, 99],
            }
        )
        with self.assertRaises(ReasonerOutputError):
            Reasoner(client, max_attempts=1).rank(
                fixture_sample(), [], BehavioralRegime(0, "hidden")
            )

    def test_reasoner_client_exception_is_bounded_and_wrapped(self) -> None:
        class FailingClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, prompt: str) -> str:
                self.calls += 1
                raise TimeoutError("fixture timeout")

        client = FailingClient()
        with self.assertRaises(ReasonerOutputError) as raised:
            Reasoner(client, max_attempts=2).rank(
                fixture_sample(), [], BehavioralRegime(0, "hidden")
            )
        self.assertEqual(client.calls, 2)
        self.assertEqual(raised.exception.responses, ())
        self.assertIsInstance(raised.exception.__cause__, TimeoutError)

    def test_packaged_prompt_matches_source_prompt(self) -> None:
        packaged = (
            DEFAULT_BASE_PROMPT_PATH.parents[1]
            / "src"
            / "drap"
            / "resources"
            / "base_prompt.txt"
        )
        self.assertEqual(
            load_base_prompt(), packaged.read_text(encoding="utf-8").strip()
        )
        with patch(
            "drap.prompts.DEFAULT_BASE_PROMPT_PATH",
            Path("/path/that/does/not/exist/base_prompt.txt"),
        ):
            self.assertEqual(
                load_base_prompt(), packaged.read_text(encoding="utf-8").strip()
            )

    def test_analyst_also_hides_unverified_regime_name_by_default(self) -> None:
        session = fixture_sample()
        feedback = SessionFeedback(1, "HDMI adapter")
        output = RankedOutput((2, 1), "intent", "{}", "prompt")
        regime = BehavioralRegime(0, "unverified label")
        hidden = analyst_prompt(session, feedback, output, [], regime)
        visible = analyst_prompt(
            session,
            feedback,
            output,
            [],
            regime,
            include_regime_name=True,
        )
        self.assertNotIn("unverified label", hidden)
        self.assertIn("unverified label", visible)

    def test_analyst_prompt_includes_authoritative_computed_target_rank(self) -> None:
        prompt = analyst_prompt(
            fixture_sample(),
            SessionFeedback(1, "HDMI adapter"),
            RankedOutput((2, 1), "intent", "{}", "prompt"),
            [],
            BehavioralRegime(0, "hidden"),
            computed_target_rank=2,
        )
        self.assertIn("Computed target rank: 2", prompt)
        self.assertIn("Do not confuse the candidate ID", prompt)

    def test_analyst_rejects_target_brand_model_and_distinctive_phrase(self) -> None:
        payloads = (
            {
                "diagnosis": "the camera continuation was under-ranked",
                "condition": "the session needs camera equipment",
                "instruction": "prioritize Nikon-compatible equipment",
            },
            {
                "diagnosis": "the camera continuation was under-ranked",
                "condition": "the session needs model D3500 accessories",
                "instruction": "prioritize camera equipment",
            },
            {
                "diagnosis": "the camera continuation was under-ranked",
                "condition": "the user seeks a mirrorless camera",
                "instruction": "prioritize camera equipment",
            },
        )
        session = SessionInput(
            sample_id="brand-model",
            history=("photography accessories",),
            candidates=(
                Candidate(1, "Nikon D3500 Mirrorless Camera"),
                Candidate(2, "generic bag"),
            ),
        )
        feedback = SessionFeedback(1, "Nikon D3500 Mirrorless Camera")
        output = RankedOutput((2, 1), "camera equipment", "{}", "prompt")
        for payload in payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(AnalystOutputError):
                    Analyst(StaticClient(payload), max_attempts=1).induce(
                        session,
                        feedback,
                        output,
                        [],
                        BehavioralRegime(0, "hidden"),
                        computed_target_rank=2,
                    )

    def test_analyst_diagnosis_may_name_target_but_rule_must_be_generic(self) -> None:
        payload = {
            "diagnosis": "Nikon D3500 Mirrorless Camera was under-ranked",
            "condition": "the session repeatedly concerns photography equipment",
            "instruction": "prioritize compatible equipment in the same category",
        }
        proposal = Analyst(StaticClient(payload), max_attempts=1).induce(
            SessionInput(
                "generic-rule",
                ("photography accessories",),
                (Candidate(1, "Nikon D3500 Mirrorless Camera"), Candidate(2, "bag")),
            ),
            SessionFeedback(1, "Nikon D3500 Mirrorless Camera"),
            RankedOutput((2, 1), "camera equipment", "{}", "prompt"),
            [],
            BehavioralRegime(0, "hidden"),
            computed_target_rank=2,
        )
        self.assertIn("Nikon", proposal.diagnosis)
        self.assertNotIn("Nikon", proposal.condition)

    def test_analyst_allows_category_phrase_supported_by_other_candidates(self) -> None:
        session = SessionInput(
            sample_id="shared-category-phrase",
            history=(
                "portable battery charger for tablets and smartphones",
                "tablet case",
            ),
            candidates=(
                Candidate(
                    1,
                    "Brand X battery charger replacement kit for digital cameras",
                ),
                Candidate(2, "digital video camera"),
                Candidate(3, "protective case for DSLR cameras"),
            ),
        )
        payload = {
            "diagnosis": "portable power interest did not transfer across devices",
            "condition": "the user seeks portable power for electronic devices",
            "instruction": (
                "prioritize compatible power accessories even for digital cameras"
            ),
        }
        proposal = Analyst(StaticClient(payload), max_attempts=1).induce(
            session,
            SessionFeedback(
                1,
                "Brand X battery charger replacement kit for digital cameras",
            ),
            RankedOutput((2, 3, 1), "portable power", "{}", "prompt"),
            [],
            BehavioralRegime(0, "hidden"),
            computed_target_rank=3,
        )
        self.assertIn("digital cameras", proposal.instruction)

    def test_distinctive_phrase_error_names_the_rejected_phrase(self) -> None:
        payload = {
            "diagnosis": "the product subtype was under-ranked",
            "condition": "the user seeks a mirrorless camera",
            "instruction": "prioritize photography equipment",
        }
        with self.assertRaises(AnalystOutputError) as raised:
            Analyst(StaticClient(payload), max_attempts=1).induce(
                SessionInput(
                    "distinctive-phrase",
                    ("photography accessories",),
                    (
                        Candidate(1, "Nikon D3500 Mirrorless Camera"),
                        Candidate(2, "generic bag"),
                    ),
                ),
                SessionFeedback(1, "Nikon D3500 Mirrorless Camera"),
                RankedOutput((2, 1), "camera equipment", "{}", "prompt"),
                [],
                BehavioralRegime(0, "hidden"),
                computed_target_rank=2,
            )
        self.assertIn("'mirrorless camera'", str(raised.exception))

    def test_analyst_invalid_json_is_repaired_once(self) -> None:
        class RepairClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, prompt: str) -> str:
                self.calls += 1
                if self.calls == 1:
                    return "not json"
                return json.dumps(
                    {
                        "diagnosis": "connector continuity was underweighted",
                        "condition": "repeated connector need",
                        "instruction": "prefer adapters",
                    }
                )

        client = RepairClient()
        proposal = Analyst(client).induce(
            fixture_sample(),
            SessionFeedback(1, "HDMI adapter"),
            RankedOutput((2, 1), "connect devices", "{}", "prompt"),
            [],
            BehavioralRegime(0, "hidden"),
        )
        self.assertEqual(proposal.instruction, "prefer adapters")
        self.assertEqual(
            proposal.diagnosis, "connector continuity was underweighted"
        )
        self.assertFalse(hasattr(proposal, "memory_type"))
        self.assertEqual(client.calls, 2)

    def test_analyst_requires_three_non_empty_json_strings(self) -> None:
        invalid_payloads = (
            {"condition": "condition", "instruction": "instruction"},
            {
                "diagnosis": 7,
                "condition": "condition",
                "instruction": "instruction",
            },
            {
                "diagnosis": "diagnosis",
                "condition": " ",
                "instruction": "instruction",
            },
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(AnalystOutputError):
                    Analyst(StaticClient(payload), max_attempts=1).induce(
                        fixture_sample(),
                        SessionFeedback(1, "HDMI adapter"),
                        RankedOutput((2, 1), "intent", "{}", "prompt"),
                        [],
                        BehavioralRegime(0, "hidden"),
                    )

    def test_analyst_rejects_ground_truth_memorization_and_repairs(self) -> None:
        class RepairClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, prompt: str) -> str:
                self.calls += 1
                if self.calls == 1:
                    return json.dumps(
                        {
                            "diagnosis": "the target was under-ranked",
                            "condition": "HDMI adapter is the expected item",
                            "instruction": "prioritize candidate ID 1",
                        }
                    )
                return json.dumps(
                    {
                        "diagnosis": "connector continuity was underweighted",
                        "condition": (
                            "the session repeatedly needs compatible connectors"
                        ),
                        "instruction": "prioritize compatible connector continuations",
                    }
                )

        client = RepairClient()
        proposal = Analyst(client).induce(
            fixture_sample(),
            SessionFeedback(1, "HDMI adapter"),
            RankedOutput((2, 1), "connect devices", "{}", "prompt"),
            [],
            BehavioralRegime(0, "hidden"),
        )
        self.assertEqual(client.calls, 2)
        self.assertEqual(
            proposal.instruction, "prioritize compatible connector continuations"
        )

    def test_analyst_rejects_id_and_title_references_independently(self) -> None:
        payloads = (
            {
                "diagnosis": "the target was under-ranked",
                "condition": "the session needs a compatible connector",
                "instruction": "prioritize candidate ID 1",
            },
            {
                "diagnosis": "the target was under-ranked",
                "condition": "the user is looking for HDMI adapter",
                "instruction": "prioritize a compatible continuation",
            },
            {
                "diagnosis": "the target was under-ranked",
                "condition": "the session contains mixed product categories",
                "instruction": "prioritize the ground-truth item's category",
            },
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(AnalystOutputError):
                    Analyst(StaticClient(payload), max_attempts=1).induce(
                        fixture_sample(),
                        SessionFeedback(1, "HDMI adapter"),
                        RankedOutput((2, 1), "connect devices", "{}", "prompt"),
                        [],
                        BehavioralRegime(0, "hidden"),
                    )

    def test_analyst_client_exception_is_bounded_and_wrapped(self) -> None:
        class FailingClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, prompt: str) -> str:
                self.calls += 1
                raise ConnectionError("fixture connection failure")

        client = FailingClient()
        with self.assertRaises(AnalystOutputError) as raised:
            Analyst(client, max_attempts=2).induce(
                fixture_sample(),
                SessionFeedback(1, "HDMI adapter"),
                RankedOutput((2, 1), "connect devices", "{}", "prompt"),
                [],
                BehavioralRegime(0, "hidden"),
            )
        self.assertEqual(client.calls, 2)
        self.assertEqual(raised.exception.responses, ())
        self.assertIsInstance(raised.exception.__cause__, ConnectionError)


if __name__ == "__main__":
    unittest.main()
