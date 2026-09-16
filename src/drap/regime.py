from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .embeddings import Embedder, cosine_similarity
from .models import BehavioralRegime, SessionInput


FEATURE_NAMES = (
    "log_session_length",
    "history_coherence",
    "recent_alignment",
    "preference_shift",
)


@dataclass(frozen=True)
class RegimeConfig:
    n_clusters: int = 3
    seed: int = 0
    n_init: int = 10
    max_iterations: int = 100
    tolerance: float = 1e-6
    recent_window: int = 3
    representative_count: int = 3
    include_name_in_prompt: bool = False

    def __post_init__(self) -> None:
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if self.n_init <= 0:
            raise ValueError("n_init must be positive")
        if (
            self.max_iterations <= 0
            or self.recent_window <= 0
            or self.representative_count <= 0
        ):
            raise ValueError(
                "max_iterations, recent_window and representative_count must be positive"
            )
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    dimensions = len(vectors[0])
    if dimensions == 0:
        return []
    result = [0.0] * dimensions
    for vector in vectors:
        if len(vector) != dimensions:
            raise ValueError("embedding dimensions must be consistent")
        for index, value in enumerate(vector):
            result[index] += value
    return [value / len(vectors) for value in result]


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


class RegimeDetector:
    """Train-only standardized K-Means over session-interaction features."""

    def __init__(self, embedder: Embedder, config: RegimeConfig | None = None):
        self.embedder = embedder
        self.config = config or RegimeConfig()
        self.means: tuple[float, ...] = ()
        self.scales: tuple[float, ...] = ()
        self.centroids: tuple[tuple[float, ...], ...] = ()
        self.labels: tuple[str, ...] = ()
        self.diagnostics: dict = {}
        self._embedding_cache: dict[str, tuple[float, ...]] = {}

    @property
    def fitted(self) -> bool:
        return bool(self.centroids)

    def _encode(self, text: str) -> tuple[float, ...]:
        cached = self._embedding_cache.get(text)
        if cached is None:
            cached = tuple(float(value) for value in self.embedder.encode(text))
            self._embedding_cache[text] = cached
        return cached

    def extract_features(self, session: SessionInput) -> tuple[float, ...]:
        vectors = [self._encode(text) for text in session.history]
        length = len(vectors)
        if length == 0:
            return (0.0, 0.0, 0.0, 0.0)

        pairwise = [
            cosine_similarity(vectors[left], vectors[right])
            for left in range(length)
            for right in range(left + 1, length)
        ]
        coherence = sum(pairwise) / len(pairwise) if pairwise else 1.0

        recent_size = min(self.config.recent_window, max(1, length // 2))
        recent_vectors = vectors[-recent_size:]
        full_mean = _mean_vector(vectors)
        recent_mean = _mean_vector(recent_vectors)
        recent_alignment = cosine_similarity(full_mean, recent_mean)

        if length < 2:
            preference_shift = 0.0
        else:
            early_vectors = vectors[:-recent_size]
            preference_shift = 1.0 - cosine_similarity(
                _mean_vector(early_vectors), _mean_vector(recent_vectors)
            )

        return (
            math.log1p(length),
            max(-1.0, min(1.0, coherence)),
            max(-1.0, min(1.0, recent_alignment)),
            max(0.0, min(2.0, preference_shift)),
        )

    def fit(self, sessions: Sequence[SessionInput]) -> RegimeDetector:
        if len(sessions) < self.config.n_clusters:
            raise ValueError(
                "number of training sessions must be at least n_clusters"
            )
        raw = [self.extract_features(session) for session in sessions]
        dimensions = len(FEATURE_NAMES)
        means = [sum(row[i] for row in raw) / len(raw) for i in range(dimensions)]
        scales: list[float] = []
        for index in range(dimensions):
            variance = sum((row[index] - means[index]) ** 2 for row in raw) / len(raw)
            scale = math.sqrt(variance)
            scales.append(scale if scale > 1e-12 else 1.0)
        standardized = [
            tuple((row[i] - means[i]) / scales[i] for i in range(dimensions))
            for row in raw
        ]
        self.means = tuple(means)
        self.scales = tuple(scales)
        best: tuple[
            float,
            tuple[tuple[float, ...], ...],
            tuple[int, ...],
            int,
            int,
        ] | None = None
        initialization_runs: list[dict[str, float | int]] = []
        for initialization in range(self.config.n_init):
            run_seed = self.config.seed + initialization * 104_729
            centers, assignments, iterations = self._fit_kmeans(
                standardized, run_seed
            )
            inertia = sum(
                _squared_distance(row, centers[assignment])
                for row, assignment in zip(standardized, assignments)
            )
            initialization_runs.append(
                {
                    "initialization": initialization,
                    "seed": run_seed,
                    "inertia": inertia,
                    "iterations": iterations,
                }
            )
            candidate = (
                inertia,
                centers,
                assignments,
                iterations,
                initialization,
            )
            if best is None or (candidate[0], candidate[4]) < (best[0], best[4]):
                best = candidate
        assert best is not None
        inertia, self.centroids, assignments, iterations, initialization = best
        self.labels = self._derive_labels()
        self.diagnostics = self._build_diagnostics(
            sessions=sessions,
            raw=raw,
            standardized=standardized,
            assignments=assignments,
            inertia=inertia,
            iterations=iterations,
            selected_initialization=initialization,
            initialization_runs=initialization_runs,
        )
        return self

    def _fit_kmeans(
        self, rows: Sequence[tuple[float, ...]], seed: int
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[int, ...], int]:
        rng = random.Random(seed)
        selected = [rng.randrange(len(rows))]
        while len(selected) < self.config.n_clusters:
            remaining = [index for index in range(len(rows)) if index not in selected]
            next_index = max(
                remaining,
                key=lambda index: (
                    min(
                        _squared_distance(rows[index], rows[center])
                        for center in selected
                    ),
                    -index,
                ),
            )
            selected.append(next_index)
        centers = [tuple(rows[index]) for index in selected]

        iterations = 0
        for iteration in range(1, self.config.max_iterations + 1):
            iterations = iteration
            assignments = [self._nearest(row, centers) for row in rows]
            updated: list[tuple[float, ...]] = []
            for cluster_id in range(self.config.n_clusters):
                members = [
                    row for row, assignment in zip(rows, assignments) if assignment == cluster_id
                ]
                updated.append(
                    tuple(_mean_vector(members)) if members else centers[cluster_id]
                )
            movement = max(
                _squared_distance(old, new) for old, new in zip(centers, updated)
            )
            centers = updated
            if movement <= self.config.tolerance**2:
                break
        assignments = tuple(self._nearest(row, centers) for row in rows)
        return tuple(centers), assignments, iterations

    @staticmethod
    def _nearest(
        row: Sequence[float], centers: Sequence[Sequence[float]]
    ) -> int:
        return min(
            range(len(centers)),
            key=lambda index: (_squared_distance(row, centers[index]), index),
        )

    def _raw_centroids(self) -> list[tuple[float, ...]]:
        return [
            tuple(center[i] * self.scales[i] + self.means[i] for i in range(len(center)))
            for center in self.centroids
        ]

    def _derive_labels(self) -> tuple[str, ...]:
        raw = self._raw_centroids()
        cluster_ids = list(range(len(raw)))
        shift_id = max(cluster_ids, key=lambda index: (raw[index][3], -index))
        remaining = [index for index in cluster_ids if index != shift_id]
        focused_id = (
            max(remaining, key=lambda index: (raw[index][1], raw[index][2], -index))
            if remaining
            else shift_id
        )
        labels: list[str] = []
        exploratory_assigned = False
        for cluster_id in cluster_ids:
            if cluster_id == shift_id and len(cluster_ids) > 1:
                labels.append("recent preference shift")
            elif cluster_id == focused_id:
                labels.append("focused/consistent")
            elif not exploratory_assigned:
                labels.append("exploratory/diverse")
                exploratory_assigned = True
            else:
                labels.append("mixed session behavior")
        return tuple(labels)

    def _build_diagnostics(
        self,
        *,
        sessions: Sequence[SessionInput],
        raw: Sequence[tuple[float, ...]],
        standardized: Sequence[tuple[float, ...]],
        assignments: Sequence[int],
        inertia: float,
        iterations: int,
        selected_initialization: int,
        initialization_runs: list[dict[str, float | int]],
    ) -> dict:
        raw_centroids = self._raw_centroids()
        cluster_sizes = [
            sum(assignment == cluster_id for assignment in assignments)
            for cluster_id in range(self.config.n_clusters)
        ]
        representatives: list[list[dict]] = []
        for cluster_id in range(self.config.n_clusters):
            member_indices = [
                index
                for index, assignment in enumerate(assignments)
                if assignment == cluster_id
            ]
            closest = sorted(
                member_indices,
                key=lambda index: (
                    _squared_distance(
                        standardized[index], self.centroids[cluster_id]
                    ),
                    sessions[index].sample_id,
                    index,
                ),
            )[: self.config.representative_count]
            representatives.append(
                [
                    {
                        "sample_id": sessions[index].sample_id,
                        "history": list(sessions[index].history),
                        "distance_to_centroid": math.sqrt(
                            _squared_distance(
                                standardized[index], self.centroids[cluster_id]
                            )
                        ),
                        "features": {
                            name: raw[index][feature_index]
                            for feature_index, name in enumerate(FEATURE_NAMES)
                        },
                    }
                    for index in closest
                ]
            )

        warnings: list[str] = []
        for cluster_id, size in enumerate(cluster_sizes):
            if size < 2:
                warnings.append(
                    f"cluster {cluster_id} has only {size} training session(s)"
                )
        for left in range(len(self.centroids)):
            for right in range(left + 1, len(self.centroids)):
                if _squared_distance(
                    self.centroids[left], self.centroids[right]
                ) <= 1e-8:
                    warnings.append(
                        f"clusters {left} and {right} have near-identical centroids"
                    )

        return {
            "inertia": inertia,
            "iterations": iterations,
            "selected_initialization": selected_initialization,
            "initialization_runs": initialization_runs,
            "cluster_sizes": cluster_sizes,
            "standardized_centroids": [list(row) for row in self.centroids],
            "raw_centroids": [
                {
                    name: row[index]
                    for index, name in enumerate(FEATURE_NAMES)
                }
                for row in raw_centroids
            ],
            "representative_sessions": representatives,
            "labels_verified": False,
            "warnings": warnings,
        }

    def predict(self, session: SessionInput) -> BehavioralRegime:
        if not self.fitted:
            raise RuntimeError("RegimeDetector must be fit on training data first")
        raw = self.extract_features(session)
        standardized = tuple(
            (raw[i] - self.means[i]) / self.scales[i] for i in range(len(raw))
        )
        regime_id = self._nearest(standardized, self.centroids)
        return BehavioralRegime(regime_id, self.labels[regime_id], raw)

    def to_dict(self) -> dict:
        if not self.fitted:
            raise RuntimeError("cannot serialize an unfitted RegimeDetector")
        return {
            "schema_version": 2,
            "feature_names": list(FEATURE_NAMES),
            "config": asdict(self.config),
            "means": list(self.means),
            "scales": list(self.scales),
            "centroids": [list(row) for row in self.centroids],
            "labels": list(self.labels),
            "diagnostics": self.diagnostics,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> RegimeDetector:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        detector = cls(embedder, RegimeConfig(**value["config"]))
        detector.means = tuple(float(item) for item in value["means"])
        detector.scales = tuple(float(item) for item in value["scales"])
        detector.centroids = tuple(
            tuple(float(item) for item in row) for row in value["centroids"]
        )
        detector.labels = tuple(str(item) for item in value["labels"])
        detector.diagnostics = dict(value.get("diagnostics", {}))
        return detector
