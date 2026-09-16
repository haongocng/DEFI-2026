import json
from pathlib import Path
import tempfile
import unittest

from drap.embeddings import HashingEmbedder
from drap.config import ExperimentConfig, LifecycleConfig, RetrievalConfig
from drap.experiment import run_experiment


class AlwaysFirstClient:
    def __init__(self) -> None:
        self.reasoner_calls = 0

    def complete(self, prompt: str) -> str:
        if prompt.startswith("You are the Reasoner"):
            self.reasoner_calls += 1
            return json.dumps(
                {
                    "selected_intent": "fixture intent",
                    "ranked_candidate_ids": [1, 2],
                }
            )
        raise AssertionError("The analyst should not run when every calibration rank is correct")


class ExperimentTests(unittest.TestCase):
    def test_single_configuration_skips_selection_rebuild_and_keeps_metrics(self) -> None:
        rows = [
            {
                "input": (
                    'Current session interactions: [1."Recent Item"]\n'
                    'Candidate Set: [1."Correct Item", 2."Other Item"]'
                ),
                "target": "Correct Item",
                "target_index": 1,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "validation", "test"):
                (root / f"{split}.json").write_text(
                    json.dumps(rows), encoding="utf-8"
                )
            client = AlwaysFirstClient()
            run_experiment(
                train_path=root / "train.json",
                validation_path=root / "validation.json",
                test_path=root / "test.json",
                output_dir=root / "run",
                client=client,
                embedder=HashingEmbedder(),
                seeds=(0,),
                regime_k_grid=(1,),
                retrieval_k_grid=(1,),
            )

            seed_dir = root / "run/seed_0"
            self.assertEqual(client.reasoner_calls, 3)
            self.assertFalse((seed_dir / "selection").exists())
            run_config = json.loads(
                (seed_dir / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                run_config["execution"]["single_configuration_fast_path"]
            )
            self.assertFalse(
                run_config["execution"]["selection_pass_performed"]
            )
            self.assertEqual(run_config["success_k"], 5)
            self.assertEqual(run_config["llm"]["duplicate_policy"], "stable_local")
            self.assertEqual(
                (
                    run_config["selected_configuration"]["retrieval"][
                        "lambda_similarity"
                    ],
                    run_config["selected_configuration"]["retrieval"][
                        "lambda_regime"
                    ],
                    run_config["selected_configuration"]["retrieval"][
                        "lambda_utility"
                    ],
                ),
                (0.65, 0.25, 0.10),
            )
            validation_metrics = json.loads(
                (seed_dir / "validation_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                validation_metrics["selection_mode"],
                "single_configuration_fast_path",
            )
            test_metrics = json.loads(
                (seed_dir / "test_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(test_metrics["samples"], 1.0)
            self.assertEqual(
                len(
                    (seed_dir / "test_predictions.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                1,
            )

    def test_end_to_end_artifacts_and_seed_aggregation(self) -> None:
        rows = [
            {
                "input": (
                    'Current session interactions: [1."Recent Item"]\n'
                    'Candidate Set: [1."Correct Item", 2."Other Item"]'
                ),
                "target": "Correct Item",
                "target_index": 1,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "validation", "test"):
                (root / f"{split}.json").write_text(json.dumps(rows), encoding="utf-8")
            client = AlwaysFirstClient()
            summary = run_experiment(
                train_path=root / "train.json",
                validation_path=root / "validation.json",
                test_path=root / "test.json",
                output_dir=root / "run",
                client=client,
                embedder=HashingEmbedder(),
                seeds=(0, 42),
                regime_k_grid=(1,),
                retrieval_k_grid=(1, 3),
            )
            self.assertEqual(summary["aggregate"]["HR@1"]["mean"], 1.0)
            # Per seed: two candidate selection passes (2 calls each), followed
            # by one final calibration/validation/test pass (3 calls).
            self.assertEqual(client.reasoner_calls, 14)
            for seed in (0, 42):
                seed_dir = root / "run" / f"seed_{seed}"
                self.assertTrue((seed_dir / "rules.json").exists())
                self.assertTrue((seed_dir / "test_predictions.jsonl").exists())
                self.assertTrue((seed_dir / "run_config.json").exists())
                self.assertTrue((seed_dir / "regime_detector.json").exists())
                self.assertTrue((seed_dir / "regime_diagnostics.json").exists())
                self.assertTrue((seed_dir / "analyst_events.jsonl").exists())
                self.assertTrue((seed_dir / "llm_traces.jsonl").exists())
                self.assertTrue((seed_dir / "llm_trace_summary.json").exists())
                self.assertEqual(
                    (seed_dir / "analyst_events.jsonl").read_text(encoding="utf-8"),
                    "",
                )
                run_config = json.loads(
                    (seed_dir / "run_config.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    run_config["credit_assignment"]["mode"], "shared_outcome"
                )
                self.assertFalse(
                    run_config["credit_assignment"]["per_rule_counterfactual"]
                )
                self.assertTrue(run_config["analyst_diagnosis"]["required"])
                trace_summary = json.loads(
                    (seed_dir / "llm_trace_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(trace_summary["accepted_calls"], 3)
                self.assertEqual(trace_summary["counts_by_phase"]["calibration"], 1)
                self.assertEqual(trace_summary["counts_by_phase"]["validation"], 1)
                self.assertEqual(trace_summary["counts_by_phase"]["test"], 1)
            self.assertTrue((root / "run" / "summary.json").exists())

    def test_complete_candidate_configuration_is_selected_and_recorded(self) -> None:
        rows = [
            {
                "input": (
                    'Current session interactions: [1."Recent Item"]\n'
                    'Candidate Set: [1."Correct Item", 2."Other Item"]'
                ),
                "target": "Correct Item",
                "target_index": 1,
            }
        ]
        candidates = (
            ExperimentConfig(
                name="b_success_heavy",
                regime_clusters=1,
                retrieval=RetrievalConfig(
                    stm_k=1,
                    ltm_k=0,
                    total_k=1,
                    lambda_similarity=0.75,
                    lambda_regime=0.0,
                    lambda_utility=0.25,
                ),
                lifecycle=LifecycleConfig(minimum_uses=5),
            ),
            ExperimentConfig(
                name="a_default",
                regime_clusters=1,
                retrieval=RetrievalConfig(stm_k=1, ltm_k=0, total_k=1),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "validation", "test"):
                (root / f"{split}.json").write_text(
                    json.dumps(rows), encoding="utf-8"
                )
            run_experiment(
                train_path=root / "train.json",
                validation_path=root / "validation.json",
                test_path=root / "test.json",
                output_dir=root / "run",
                client=AlwaysFirstClient(),
                embedder=HashingEmbedder(),
                seeds=(0,),
                candidate_configs=candidates,
            )
            recorded = json.loads(
                (root / "run/seed_0/run_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                recorded["selected_configuration"]["name"], "b_success_heavy"
            )
            self.assertEqual(
                recorded["selected_configuration"]["lifecycle"]["minimum_uses"],
                5,
            )


if __name__ == "__main__":
    unittest.main()
