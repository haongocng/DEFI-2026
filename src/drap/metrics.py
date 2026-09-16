from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence

from .models import PredictionRecord


def ranking_metrics(
    records: Iterable[PredictionRecord], ks: Sequence[int] = (1, 5, 10)
) -> dict[str, float]:
    rows = list(records)
    metrics: dict[str, float] = {"samples": float(len(rows))}
    for k in ks:
        hits: list[float] = []
        ndcgs: list[float] = []
        aps: list[float] = []
        for row in rows:
            try:
                derived_rank = row.predicted_ids.index(row.target_id) + 1
            except ValueError:
                derived_rank = math.inf
            if row.target_rank is not None and row.target_rank != derived_rank:
                raise ValueError(
                    f"Inconsistent target_rank for sample {row.sample_id}: "
                    f"stored={row.target_rank}, derived={derived_rank}"
                )
            rank = derived_rank
            hits.append(float(rank <= k))
            ndcgs.append(1.0 / math.log2(rank + 1) if rank <= k else 0.0)
            aps.append(1.0 / rank if rank <= k else 0.0)
        metrics[f"HR@{k}"] = statistics.fmean(hits) if hits else 0.0
        metrics[f"NDCG@{k}"] = statistics.fmean(ndcgs) if ndcgs else 0.0
        metrics[f"MAP@{k}"] = statistics.fmean(aps) if aps else 0.0
    return metrics


def aggregate_seed_metrics(results: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not results:
        return {}
    keys = sorted(set.intersection(*(set(result) for result in results)))
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(result[key]) for result in results]
        summary[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
        }
    return summary
