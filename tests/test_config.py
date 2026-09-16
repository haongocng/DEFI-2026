import json
from pathlib import Path
import unittest

from drap.config import ExperimentConfig, LifecycleConfig, RetrievalConfig


class ConfigurationTests(unittest.TestCase):
    def test_top_one_experiment_retrieval_defaults(self) -> None:
        config = RetrievalConfig()
        self.assertEqual((config.stm_k, config.ltm_k, config.total_k), (2, 3, 5))
        self.assertEqual(
            (
                config.lambda_similarity,
                config.lambda_regime,
                config.lambda_utility,
            ),
            (0.65, 0.25, 0.10),
        )

    def test_repository_presets_are_valid_and_cover_ablation_dimensions(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs/recommendation_presets.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates = [ExperimentConfig.from_dict(row) for row in payload["candidates"]]
        self.assertEqual(len(candidates), len({candidate.name for candidate in candidates}))
        self.assertTrue(all(not candidate.include_regime_name for candidate in candidates))
        self.assertIn(0.0, {candidate.retrieval.lambda_regime for candidate in candidates})
        self.assertIn(0.25, {candidate.retrieval.lambda_regime for candidate in candidates})
        self.assertGreater(len({candidate.retrieval.utility.alpha for candidate in candidates}), 1)
        self.assertGreater(len({candidate.lifecycle.minimum_uses for candidate in candidates}), 1)
        self.assertEqual({candidate.regime_clusters for candidate in candidates}, {2, 3, 4})

    def test_incoherent_lifecycle_thresholds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LifecycleConfig(
                deletion_utility=0.5,
                demotion_utility=0.3,
                promotion_utility=0.6,
            )


if __name__ == "__main__":
    unittest.main()
