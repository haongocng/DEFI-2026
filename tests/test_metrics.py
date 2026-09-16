import unittest

from drap.metrics import aggregate_seed_metrics, ranking_metrics
from drap.models import PredictionRecord


class MetricsTests(unittest.TestCase):
    def test_single_relevant_item_metrics(self) -> None:
        records = [
            PredictionRecord("a", 1, (1, 2), (), True, ""),
            PredictionRecord("b", 2, (1, 2), (), True, ""),
            PredictionRecord("c", 3, (1, 2), (), False, ""),
        ]
        metrics = ranking_metrics(records)
        self.assertAlmostEqual(metrics["HR@1"], 1 / 3)
        self.assertAlmostEqual(metrics["HR@5"], 2 / 3)
        self.assertAlmostEqual(metrics["MAP@5"], 0.5)

    def test_seed_aggregation(self) -> None:
        summary = aggregate_seed_metrics([{"HR@1": 0.2}, {"HR@1": 0.4}])
        self.assertAlmostEqual(summary["HR@1"]["mean"], 0.3)
        self.assertAlmostEqual(summary["HR@1"]["std"], 0.1)

    def test_inconsistent_stored_target_rank_is_rejected(self) -> None:
        record = PredictionRecord(
            "a", 2, (1, 2), (), True, "", target_rank=1
        )
        with self.assertRaises(ValueError):
            ranking_metrics([record])


if __name__ == "__main__":
    unittest.main()
