from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


def _require_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _require_unit_sum(name: str, values: tuple[float, ...]) -> None:
    if any(value < 0.0 for value in values):
        raise ValueError(f"{name} weights must be non-negative")
    if abs(sum(values) - 1.0) > 1e-9:
        raise ValueError(f"{name} weights must sum to 1.0")


@dataclass(frozen=True)
class UtilityConfig:
    """Weights for success, frequency and recency rule evidence."""

    alpha: float = 0.60
    beta: float = 0.20
    gamma: float = 0.20
    kappa: float = 5.0
    tau: float = 10.0

    def __post_init__(self) -> None:
        _require_unit_sum("utility", (self.alpha, self.beta, self.gamma))
        if self.kappa <= 0.0 or self.tau <= 0.0:
            raise ValueError("kappa and tau must be positive")


@dataclass(frozen=True)
class RetrievalConfig:
    """Regime-aware dual-memory retrieval configuration."""

    stm_k: int = 2
    ltm_k: int = 3
    total_k: int = 5
    lambda_similarity: float = 0.65
    lambda_regime: float = 0.25
    lambda_utility: float = 0.10
    minimum_score: float = 0.0
    utility: UtilityConfig = field(default_factory=UtilityConfig)

    def __post_init__(self) -> None:
        if self.stm_k < 0 or self.ltm_k < 0 or self.total_k < 0:
            raise ValueError("retrieval budgets must be non-negative")
        if self.total_k > self.stm_k + self.ltm_k:
            raise ValueError("total_k cannot exceed stm_k + ltm_k")
        _require_unit_sum(
            "retrieval",
            (
                self.lambda_similarity,
                self.lambda_regime,
                self.lambda_utility,
            ),
        )
        _require_probability("minimum_score", self.minimum_score)


@dataclass(frozen=True)
class LifecycleConfig:
    """Deterministic create/promote/demote/archive policy."""

    minimum_uses: int = 3
    promotion_success: float = 0.60
    promotion_utility: float = 0.60
    demotion_utility: float = 0.35
    deletion_utility: float = 0.20
    stale_window: int = 25
    max_stm_rules: int = 100
    max_ltm_rules: int = 100
    novelty_threshold: float = 0.90

    def __post_init__(self) -> None:
        if self.minimum_uses <= 0:
            raise ValueError("minimum_uses must be positive")
        if self.stale_window <= 0:
            raise ValueError("stale_window must be positive")
        if self.max_stm_rules <= 0 or self.max_ltm_rules <= 0:
            raise ValueError("memory budgets must be positive")
        for name, value in (
            ("promotion_success", self.promotion_success),
            ("promotion_utility", self.promotion_utility),
            ("demotion_utility", self.demotion_utility),
            ("deletion_utility", self.deletion_utility),
            ("novelty_threshold", self.novelty_threshold),
        ):
            _require_probability(name, value)
        if not (
            self.deletion_utility
            <= self.demotion_utility
            <= self.promotion_utility
        ):
            raise ValueError(
                "lifecycle utility thresholds must satisfy "
                "deletion <= demotion <= promotion"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    """One complete configuration whose memory must be calibrated independently."""

    name: str
    regime_clusters: int = 3
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    include_regime_name: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name):
            raise ValueError(
                "configuration name may contain only letters, digits, _, -, and ."
            )
        if self.regime_clusters <= 0:
            raise ValueError("regime_clusters must be positive")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExperimentConfig:
        retrieval_value = dict(value.get("retrieval", {}))
        utility = UtilityConfig(**dict(retrieval_value.pop("utility", {})))
        retrieval = RetrievalConfig(utility=utility, **retrieval_value)
        lifecycle = LifecycleConfig(**dict(value.get("lifecycle", {})))
        return cls(
            name=str(value["name"]),
            regime_clusters=int(value.get("regime_clusters", 3)),
            retrieval=retrieval,
            lifecycle=lifecycle,
            include_regime_name=bool(value.get("include_regime_name", False)),
        )
