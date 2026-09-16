from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Literal
from uuid import uuid4

from .config import UtilityConfig


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Candidate:
    id: int
    text: str


@dataclass(frozen=True)
class SessionInput:
    sample_id: str
    history: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    raw_input: str = ""

    @property
    def context(self) -> str:
        if not self.history:
            return "No previous interactions are available."
        numbered = "\n".join(f"{i}. {item}" for i, item in enumerate(self.history, 1))
        return f"Chronological interaction history (oldest to newest):\n{numbered}"


@dataclass(frozen=True)
class SessionFeedback:
    target_id: int
    target_text: str


@dataclass(frozen=True)
class DatasetSample:
    session: SessionInput
    feedback: SessionFeedback


@dataclass(frozen=True)
class BehavioralRegime:
    id: int
    name: str
    features: tuple[float, ...] = ()


@dataclass
class Rule:
    condition: str
    instruction: str
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    active: bool = True
    regime_id: int = -1
    memory_type: Literal["STM", "LTM"] = "STM"
    condition_embedding: tuple[float, ...] = ()
    embedding_signature: str = ""
    num_used: int = 0
    num_success: int = 0
    last_used_step: int | None = None
    created_step: int = 0
    creation_index: int = -1
    last_utility: float = 0.0
    status: Literal["active", "archived"] = "active"
    source_sample_id: str | None = None
    replaced_by: str | None = None

    @property
    def success_rate(self) -> float:
        return (self.num_success + 1) / (self.num_used + 2)

    def utility_at(self, current_step: int, config: UtilityConfig) -> float:
        frequency = 1.0 - math.exp(-self.num_used / config.kappa)
        reference_step = (
            self.last_used_step if self.last_used_step is not None else self.created_step
        )
        age = max(0, current_step - reference_step)
        recency = math.exp(-age / config.tau)
        return (
            config.alpha * self.success_rate
            + config.beta * frequency
            + config.gamma * recency
        )

    @property
    def retrieval_count(self) -> int:
        """Compatibility alias for rule files produced by the base implementation."""
        return self.num_used

    @property
    def success_count(self) -> int:
        return self.num_success

    @property
    def failure_count(self) -> int:
        return self.num_used - self.num_success

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Rule":
        # Accept the field names used by the legacy implementation.
        mapped = dict(value)
        mapped["condition"] = mapped.pop(
            "condition", mapped.pop("condition_description", "")
        )
        mapped["num_used"] = mapped.pop(
            "num_used", mapped.pop("retrieval_count", 0)
        )
        mapped["num_success"] = mapped.pop(
            "num_success", mapped.pop("success_count", 0)
        )
        mapped.pop("failure_count", None)
        if "condition_embedding" in mapped:
            mapped["condition_embedding"] = tuple(mapped["condition_embedding"] or ())
        if not mapped.get("active", True):
            mapped.setdefault("status", "archived")
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: val for key, val in mapped.items() if key in allowed})


@dataclass(frozen=True)
class RankingNormalization:
    """Auditable local repair for an otherwise valid duplicate-only ranking."""

    policy: Literal["stable_first_seen_append_missing"]
    original_ranked_candidate_ids: tuple[int, ...]
    duplicate_candidate_ids: tuple[int, ...]
    missing_candidate_ids: tuple[int, ...]
    normalized_ranked_candidate_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "original_ranked_candidate_ids",
            "duplicate_candidate_ids",
            "missing_candidate_ids",
            "normalized_ranked_candidate_ids",
        ):
            value[key] = list(value[key])
        return value


@dataclass(frozen=True)
class RankedOutput:
    ranked_candidate_ids: tuple[int, ...]
    selected_intent: str
    raw_response: str
    prompt: str
    retrieved_rule_ids: tuple[str, ...] = ()
    ranking_normalization: RankingNormalization | None = None


@dataclass(frozen=True)
class RuleProposal:
    """Analyst output before system-owned rule metadata is attached."""

    diagnosis: str
    condition: str
    instruction: str


MaintenanceAction = Literal["ADD", "REPLACE", "MERGE", "IGNORE"]


@dataclass(frozen=True)
class MaintenanceDecision:
    action: MaintenanceAction
    target_rule_ids: tuple[str, ...] = ()
    condition: str | None = None
    instruction: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class AnalystEvent:
    """Auditable record of one post-prediction Analyst decision."""

    sample_id: str
    current_step: int
    regime_id: int
    target_id: int
    target_rank: int
    selected_intent: str
    ranked_candidate_ids: tuple[int, ...]
    retrieved_rule_ids: tuple[str, ...]
    diagnosis: str
    proposed_condition: str
    proposed_instruction: str
    proposed_rule_id: str
    maintenance_action: MaintenanceAction
    affected_rule_ids: tuple[str, ...] = ()
    maintenance_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["ranked_candidate_ids"] = list(self.ranked_candidate_ids)
        value["retrieved_rule_ids"] = list(self.retrieved_rule_ids)
        value["affected_rule_ids"] = list(self.affected_rule_ids)
        return value


@dataclass(frozen=True)
class RetrievalTrace:
    rule_id: str
    memory_type: Literal["STM", "LTM"]
    score: float
    similarity: float
    regime_match: float
    utility: float


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    target_id: int
    predicted_ids: tuple[int, ...]
    retrieved_rule_ids: tuple[str, ...]
    success: bool
    selected_intent: str
    target_rank: int | None = None
    retrieval_trace: tuple[RetrievalTrace, ...] = ()
    regime_id: int = -1
    regime_name: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["predicted_ids"] = list(self.predicted_ids)
        value["retrieved_rule_ids"] = list(self.retrieved_rule_ids)
        return value
