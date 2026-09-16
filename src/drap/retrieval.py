from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import RetrievalConfig
from .embeddings import Embedder, cosine_similarity
from .models import BehavioralRegime, Rule, SessionInput


def _normalized_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return max(0.0, min(1.0, (cosine_similarity(left, right) + 1.0) / 2.0))


def _weighted_mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    dimensions = len(vectors[0])
    result = [0.0] * dimensions
    denominator = 0.0
    for position, vector in enumerate(vectors, start=1):
        if len(vector) != dimensions:
            raise ValueError("embedding dimensions must be consistent")
        weight = float(position)
        denominator += weight
        for index, value in enumerate(vector):
            result[index] += weight * value
    return [value / denominator for value in result]


@dataclass(frozen=True)
class RetrievedRule:
    rule: Rule
    score: float
    similarity: float
    regime_match: float
    utility: float


class RuleRetriever:
    def __init__(
        self, embedder: Embedder, config: RetrievalConfig | None = None
    ):
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self._text_cache: dict[str, tuple[float, ...]] = {}
        detail = getattr(
            embedder,
            "model_name",
            getattr(embedder, "dimensions", "unspecified"),
        )
        self.embedding_signature = f"{type(embedder).__name__}:{detail}"

    def _encode(self, text: str) -> tuple[float, ...]:
        cached = self._text_cache.get(text)
        if cached is None:
            cached = tuple(float(value) for value in self.embedder.encode(text))
            self._text_cache[text] = cached
        return cached

    def session_embedding(self, session: SessionInput) -> tuple[float, ...]:
        vectors = [self._encode(text) for text in session.history]
        return tuple(_weighted_mean(vectors))

    def ensure_rule_embedding(self, rule: Rule) -> tuple[float, ...]:
        if (
            not rule.condition_embedding
            or rule.embedding_signature != self.embedding_signature
        ):
            rule.condition_embedding = self._encode(rule.condition)
            rule.embedding_signature = self.embedding_signature
        return rule.condition_embedding

    def _score(
        self,
        rule: Rule,
        session_vector: Sequence[float],
        regime: BehavioralRegime,
        current_step: int,
    ) -> RetrievedRule:
        similarity = _normalized_cosine(
            session_vector, self.ensure_rule_embedding(rule)
        )
        regime_match = 1.0 if rule.regime_id == regime.id else 0.0
        utility = rule.utility_at(current_step, self.config.utility)
        score = (
            self.config.lambda_similarity * similarity
            + self.config.lambda_regime * regime_match
            + self.config.lambda_utility * utility
        )
        return RetrievedRule(rule, score, similarity, regime_match, utility)

    def retrieve(
        self,
        rules: Sequence[Rule],
        session: SessionInput,
        regime: BehavioralRegime,
        current_step: int,
    ) -> list[RetrievedRule]:
        if self.config.total_k <= 0:
            return []
        session_vector = self.session_embedding(session)
        scored = [
            self._score(rule, session_vector, regime, current_step)
            for rule in rules
            if rule.active and rule.status == "active"
        ]
        eligible = [
            item
            for item in scored
            if item.score >= self.config.minimum_score
            and not (
                item.rule.memory_type == "STM"
                and item.rule.num_used == 0
                and item.regime_match == 0.0
            )
        ]

        def ranking_key(
            item: RetrievedRule,
        ) -> tuple[float, float, int, int, str, str]:
            return (
                -item.score,
                -item.similarity,
                item.rule.created_step,
                item.rule.creation_index,
                item.rule.condition.casefold(),
                item.rule.instruction.casefold(),
            )

        stm = sorted(
            (item for item in eligible if item.rule.memory_type == "STM"),
            key=ranking_key,
        )[: self.config.stm_k]
        ltm = sorted(
            (item for item in eligible if item.rule.memory_type == "LTM"),
            key=ranking_key,
        )[: self.config.ltm_k]
        return sorted(stm + ltm, key=ranking_key)[: self.config.total_k]

    def nearest(
        self, rules: Sequence[Rule], proposed: Rule, top_k: int | None = 5
    ) -> list[RetrievedRule]:
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative or None")
        query = self.ensure_rule_embedding(proposed)
        scored: list[RetrievedRule] = []
        for rule in rules:
            if not rule.active or rule.status != "active":
                continue
            similarity = _normalized_cosine(
                query, self.ensure_rule_embedding(rule)
            )
            scored.append(
                RetrievedRule(
                    rule=rule,
                    score=similarity,
                    similarity=similarity,
                    regime_match=(
                        1.0 if rule.regime_id == proposed.regime_id else 0.0
                    ),
                    utility=0.0,
                )
            )
        ordered = sorted(
            scored,
            key=lambda item: (
                -item.score,
                item.rule.created_step,
                item.rule.creation_index,
                item.rule.condition.casefold(),
                item.rule.instruction.casefold(),
            ),
        )
        return ordered if top_k is None else ordered[:top_k]
