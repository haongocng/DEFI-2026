from __future__ import annotations

from collections import Counter
import re
import time
from typing import Literal

from .llm import LLMClient, parse_json_object
from .models import (
    BehavioralRegime,
    RankingNormalization,
    RankedOutput,
    Rule,
    RuleProposal,
    SessionFeedback,
    SessionInput,
)
from .prompts import analyst_prompt, load_base_prompt, reasoner_prompt
from .tracing import AgentTraceContext, LLMTraceRecorder, TraceStatus


class AgentOutputError(ValueError):
    def __init__(self, message: str, responses: list[str]):
        super().__init__(message)
        self.responses = tuple(responses)


class ReasonerOutputError(AgentOutputError):
    pass


class AnalystOutputError(AgentOutputError):
    pass


def _client_failure(exc: Exception) -> str:
    return f"LLM client raised {type(exc).__name__}: {exc}"


def _normalized_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def _contains_token_sequence(
    tokens: tuple[str, ...], sequence: tuple[str, ...]
) -> bool:
    if not sequence or len(sequence) > len(tokens):
        return False
    width = len(sequence)
    return any(
        tokens[index : index + width] == sequence
        for index in range(len(tokens) - width + 1)
    )


def _validate_reusable_text(
    field: str,
    value: str,
    session: SessionInput,
    feedback: SessionFeedback,
) -> None:
    feedback_meta = re.compile(
        r"(?i)\b(?:ground[- ]?truth(?:\s+item)?|"
        r"target\s+(?:item|product|candidate)|expected\s+item)\b"
    )
    if feedback_meta.search(value):
        raise ValueError(
            f"Analyst {field} contains feedback-specific meta language"
        )

    target_id = re.escape(str(feedback.target_id))
    explicit_id = re.compile(
        rf"(?i)(?:\b(?:candidate|item|product|target|id|index)\s*"
        rf"(?:id\s*)?(?:number\s*)?[#:=.-]?\s*{target_id}\b|#{target_id}\b)"
    )
    if explicit_id.search(value):
        raise ValueError(f"Analyst {field} contains the ground-truth candidate ID")

    # A complete multi-token title is a reliable memorization signal. Single-token
    # titles are not rejected because they are often reusable categories.
    title_tokens = _normalized_tokens(feedback.target_text)
    if len(title_tokens) >= 2 and _contains_token_sequence(
        _normalized_tokens(value), title_tokens
    ):
        raise ValueError(f"Analyst {field} memorizes the ground-truth product title")

    history_tokens = set(_normalized_tokens(" ".join(session.history)))
    value_tokens = _normalized_tokens(value)
    target_only_tokens = tuple(
        token for token in title_tokens if token not in history_tokens
    )

    # Product/model codes are catalog-specific even when copied as one token.
    model_like = {
        token
        for token in target_only_tokens
        if any(character.isalpha() for character in token)
        and any(character.isdigit() for character in token)
    }
    copied_model_like = sorted(model_like & set(value_tokens))
    if copied_model_like:
        raise ValueError(
            f"Analyst {field} copies a ground-truth model/code token"
        )

    # Preserve case for a conservative brand/proper-name check. Generic category
    # words are normally lower-cased in reusable prose, while copied brand names
    # retain their catalog spelling.
    raw_target_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", feedback.target_text)
    raw_history = " ".join(session.history)
    copied_proper_names = {
        token
        for token in raw_target_tokens
        if len(token) >= 4
        and token[0].isupper()
        and token not in raw_history
        and re.search(rf"(?<!\w){re.escape(token)}(?!\w)", value)
    }
    if copied_proper_names:
        raise ValueError(
            f"Analyst {field} copies a ground-truth brand/proper-name token"
        )

    # A phrase is distinctive only when its words are absent not just from the
    # interaction history but also from every non-target candidate. This keeps
    # catalog-specific phrases blocked while allowing reusable category phrases
    # such as "digital cameras" when that vocabulary is already present in the
    # candidate context.
    non_target_candidate_text = " ".join(
        candidate.text
        for candidate in session.candidates
        if candidate.id != feedback.target_id
    )
    reusable_context_tokens = history_tokens | set(
        _normalized_tokens(non_target_candidate_text)
    )
    distinctive_token_set = {
        token for token in title_tokens if token not in reusable_context_tokens
    }
    for left, right in zip(title_tokens, title_tokens[1:]):
        if (
            left in distinctive_token_set
            and right in distinctive_token_set
            and _contains_token_sequence(value_tokens, (left, right))
        ):
            phrase = f"{left} {right}"
            raise ValueError(
                f"Analyst {field} copies the distinctive ground-truth phrase "
                f"{phrase!r}"
            )


class Reasoner:
    def __init__(
        self,
        client: LLMClient,
        *,
        base_prompt: str | None = None,
        include_regime_name: bool = False,
        max_attempts: int = 2,
        trace_recorder: LLMTraceRecorder | None = None,
        duplicate_policy: Literal["stable_local", "llm_repair"] = "stable_local",
        # Accepted only so callers of the base implementation fail gracefully.
        top_n: int | None = None,
    ):
        self.client = client
        self.base_prompt = base_prompt or load_base_prompt()
        self.include_regime_name = include_regime_name
        self.top_n = top_n
        self.trace_recorder = trace_recorder
        if duplicate_policy not in ("stable_local", "llm_repair"):
            raise ValueError(
                "duplicate_policy must be 'stable_local' or 'llm_repair'"
            )
        self.duplicate_policy = duplicate_policy
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = max_attempts

    def rank(
        self,
        session: SessionInput,
        rules: list[Rule],
        regime: BehavioralRegime,
        *,
        trace_context: AgentTraceContext | None = None,
    ) -> RankedOutput:
        prompt = reasoner_prompt(
            session,
            rules,
            regime,
            base_prompt=self.base_prompt,
            include_regime_name=self.include_regime_name,
        )
        responses: list[str] = []
        errors: list[str] = []
        current_prompt = prompt
        context = trace_context or AgentTraceContext()
        retrieved_rule_ids = tuple(rule.id for rule in rules)
        last_client_error: Exception | None = None
        for attempt in range(self.max_attempts):
            attempt_number = attempt + 1
            started = time.perf_counter()
            try:
                response = self.client.complete(current_prompt)
                if not isinstance(response, str):
                    raise TypeError("LLM client must return a string response")
            except Exception as exc:
                last_client_error = exc
                error = _client_failure(exc)
                errors.append(error)
                self._trace(
                    context=context,
                    session=session,
                    regime=regime,
                    retrieved_rule_ids=retrieved_rule_ids,
                    attempt=attempt_number,
                    prompt=current_prompt,
                    response=None,
                    status="client_error",
                    started=started,
                    error=error,
                )
                continue
            responses.append(response)
            try:
                output = self._parse_response(
                    session,
                    rules,
                    prompt,
                    response,
                    normalize_duplicates=self.duplicate_policy == "stable_local",
                )
            except ValueError as exc:
                error = str(exc)
                errors.append(error)
                self._trace(
                    context=context,
                    session=session,
                    regime=regime,
                    retrieved_rule_ids=retrieved_rule_ids,
                    attempt=attempt_number,
                    prompt=current_prompt,
                    response=response,
                    status="rejected",
                    started=started,
                    error=error,
                )
                if attempt + 1 < self.max_attempts:
                    current_prompt = self._repair_prompt(prompt, response, str(exc))
            else:
                self._trace(
                    context=context,
                    session=session,
                    regime=regime,
                    retrieved_rule_ids=retrieved_rule_ids,
                    attempt=attempt_number,
                    prompt=current_prompt,
                    response=response,
                    status="accepted",
                    started=started,
                    ranking_normalization=output.ranking_normalization,
                )
                return output
        error = ReasonerOutputError(
            "Reasoner output remained invalid after "
            f"{self.max_attempts} attempts: {'; '.join(errors)}",
            responses,
        )
        if last_client_error is not None:
            raise error from last_client_error
        raise error

    def _trace(
        self,
        *,
        context: AgentTraceContext,
        session: SessionInput,
        regime: BehavioralRegime,
        retrieved_rule_ids: tuple[str, ...],
        attempt: int,
        prompt: str,
        response: str | None,
        status: TraceStatus,
        started: float,
        error: str | None = None,
        ranking_normalization: RankingNormalization | None = None,
    ) -> None:
        if self.trace_recorder is None:
            return
        self.trace_recorder.record(
            stage="reasoner",
            context=context,
            sample_id=session.sample_id,
            regime_id=regime.id,
            retrieved_rule_ids=retrieved_rule_ids,
            attempt=attempt,
            max_attempts=self.max_attempts,
            prompt=prompt,
            response=response,
            status=status,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=error,
            output_normalization=(
                ranking_normalization.to_dict()
                if ranking_normalization is not None
                else None
            ),
        )

    @staticmethod
    def _repair_prompt(original_prompt: str, response: str, error: str) -> str:
        return f"""{original_prompt}

[OUTPUT REPAIR]
The previous response violated the output contract.
Validation error: {error}
Previous response: {response}

Return a corrected JSON object only. Re-evaluate the complete candidate set and
do not copy an invalid or incomplete ranking.
"""

    @staticmethod
    def _parse_response(
        session: SessionInput,
        rules: list[Rule],
        prompt: str,
        response: str,
        *,
        normalize_duplicates: bool = True,
    ) -> RankedOutput:
        value = parse_json_object(response)
        raw_intent = value.get("selected_intent")
        selected_intent = raw_intent.strip() if isinstance(raw_intent, str) else ""
        raw_ranking = value.get("ranked_candidate_ids")
        if not selected_intent:
            raise ReasonerOutputError(
                "Reasoner returned an empty selected_intent", [response]
            )
        if not isinstance(raw_ranking, list):
            raise ReasonerOutputError(
                "ranked_candidate_ids must be a JSON list", [response]
            )

        if any(type(raw_id) is not int for raw_id in raw_ranking):
            raise ReasonerOutputError(
                "ranked_candidate_ids must contain JSON integers only", [response]
            )
        ranked = tuple(raw_ranking)

        expected = tuple(candidate.id for candidate in session.candidates)
        normalization = (
            Reasoner._normalize_duplicate_only_ranking(ranked, expected)
            if normalize_duplicates
            else None
        )
        if normalization is not None:
            ranked = normalization.normalized_ranked_candidate_ids

        Reasoner._validate_candidate_permutation(ranked, expected, response)

        return RankedOutput(
            ranked_candidate_ids=ranked,
            selected_intent=selected_intent,
            raw_response=response,
            prompt=prompt,
            retrieved_rule_ids=tuple(rule.id for rule in rules),
            ranking_normalization=normalization,
        )

    @staticmethod
    def _normalize_duplicate_only_ranking(
        ranked: tuple[int, ...],
        expected: tuple[int, ...],
    ) -> RankingNormalization | None:
        """Repair only a full-length, in-vocabulary ranking with duplicates.

        The first occurrence of every returned ID keeps its relative position.
        Candidate IDs made missing by duplication are appended in the original
        candidate-set order. Any unknown ID or independent length error remains
        invalid and is left for the bounded LLM repair path.
        """
        if len(ranked) != len(expected) or len(ranked) == len(set(ranked)):
            return None
        expected_set = set(expected)
        if any(candidate_id not in expected_set for candidate_id in ranked):
            return None

        counts = Counter(ranked)
        seen: set[int] = set()
        stable_unique: list[int] = []
        duplicate_ids: list[int] = []
        for candidate_id in ranked:
            if candidate_id not in seen:
                seen.add(candidate_id)
                stable_unique.append(candidate_id)
                if counts[candidate_id] > 1:
                    duplicate_ids.append(candidate_id)

        missing_ids = tuple(
            candidate_id for candidate_id in expected if candidate_id not in seen
        )
        normalized = tuple(stable_unique) + missing_ids
        return RankingNormalization(
            policy="stable_first_seen_append_missing",
            original_ranked_candidate_ids=ranked,
            duplicate_candidate_ids=tuple(duplicate_ids),
            missing_candidate_ids=missing_ids,
            normalized_ranked_candidate_ids=normalized,
        )

    @staticmethod
    def _validate_candidate_permutation(
        ranked: tuple[int, ...],
        expected: tuple[int, ...],
        response: str,
    ) -> None:
        if len(ranked) != len(expected):
            raise ReasonerOutputError(
                f"Reasoner returned {len(ranked)} IDs; expected {len(expected)}",
                [response],
            )
        if len(ranked) != len(set(ranked)):
            raise ReasonerOutputError(
                "Reasoner returned duplicate candidate IDs", [response]
            )
        if set(ranked) != set(expected):
            missing = sorted(set(expected) - set(ranked))
            unknown = sorted(set(ranked) - set(expected))
            raise ReasonerOutputError(
                f"Reasoner ranking is not a candidate permutation; "
                f"missing={missing}, unknown={unknown}",
                [response],
            )


class Analyst:
    def __init__(
        self,
        client: LLMClient,
        *,
        include_regime_name: bool = False,
        max_attempts: int = 2,
        trace_recorder: LLMTraceRecorder | None = None,
    ):
        self.client = client
        self.include_regime_name = include_regime_name
        self.trace_recorder = trace_recorder
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = max_attempts

    def induce(
        self,
        session: SessionInput,
        feedback: SessionFeedback,
        output: RankedOutput,
        retrieved_rules: list[Rule],
        regime: BehavioralRegime,
        *,
        computed_target_rank: int | None = None,
        trace_context: AgentTraceContext | None = None,
    ) -> RuleProposal:
        prompt = analyst_prompt(
            session,
            feedback,
            output,
            retrieved_rules,
            regime,
            computed_target_rank=computed_target_rank,
            include_regime_name=self.include_regime_name,
        )
        current_prompt = prompt
        responses: list[str] = []
        errors: list[str] = []
        context = trace_context or AgentTraceContext()
        retrieved_rule_ids = tuple(rule.id for rule in retrieved_rules)
        last_client_error: Exception | None = None
        for attempt in range(self.max_attempts):
            attempt_number = attempt + 1
            started = time.perf_counter()
            try:
                response = self.client.complete(current_prompt)
                if not isinstance(response, str):
                    raise TypeError("LLM client must return a string response")
            except Exception as exc:
                last_client_error = exc
                error = _client_failure(exc)
                errors.append(error)
                self._trace(
                    context=context,
                    session=session,
                    regime=regime,
                    retrieved_rule_ids=retrieved_rule_ids,
                    attempt=attempt_number,
                    prompt=current_prompt,
                    response=None,
                    status="client_error",
                    started=started,
                    error=error,
                )
                continue
            responses.append(response)
            try:
                value = parse_json_object(response)
                raw_diagnosis = value.get("diagnosis")
                raw_condition = value.get("condition")
                raw_instruction = value.get("instruction")
                diagnosis = (
                    raw_diagnosis.strip()
                    if isinstance(raw_diagnosis, str)
                    else ""
                )
                condition = (
                    raw_condition.strip() if isinstance(raw_condition, str) else ""
                )
                instruction = (
                    raw_instruction.strip()
                    if isinstance(raw_instruction, str)
                    else ""
                )
                if not diagnosis or not condition or not instruction:
                    raise ValueError(
                        "Analyst must return non-empty JSON strings for diagnosis, "
                        "condition and instruction"
                    )
                _validate_reusable_text(
                    "condition", condition, session, feedback
                )
                _validate_reusable_text(
                    "instruction", instruction, session, feedback
                )
                proposal = RuleProposal(
                    diagnosis=diagnosis,
                    condition=condition,
                    instruction=instruction,
                )
            except ValueError as exc:
                error = str(exc)
                errors.append(error)
                self._trace(
                    context=context,
                    session=session,
                    regime=regime,
                    retrieved_rule_ids=retrieved_rule_ids,
                    attempt=attempt_number,
                    prompt=current_prompt,
                    response=response,
                    status="rejected",
                    started=started,
                    error=error,
                )
                if attempt + 1 < self.max_attempts:
                    current_prompt = self._repair_prompt(prompt, response, error)
            else:
                self._trace(
                    context=context,
                    session=session,
                    regime=regime,
                    retrieved_rule_ids=retrieved_rule_ids,
                    attempt=attempt_number,
                    prompt=current_prompt,
                    response=response,
                    status="accepted",
                    started=started,
                )
                return proposal
        output_error = AnalystOutputError(
            "Analyst output remained invalid after "
            f"{self.max_attempts} attempts: {'; '.join(errors)}",
            responses,
        )
        if last_client_error is not None:
            raise output_error from last_client_error
        raise output_error

    def _trace(
        self,
        *,
        context: AgentTraceContext,
        session: SessionInput,
        regime: BehavioralRegime,
        retrieved_rule_ids: tuple[str, ...],
        attempt: int,
        prompt: str,
        response: str | None,
        status: TraceStatus,
        started: float,
        error: str | None = None,
    ) -> None:
        if self.trace_recorder is None:
            return
        self.trace_recorder.record(
            stage="analyst",
            context=context,
            sample_id=session.sample_id,
            regime_id=regime.id,
            retrieved_rule_ids=retrieved_rule_ids,
            attempt=attempt,
            max_attempts=self.max_attempts,
            prompt=prompt,
            response=response,
            status=status,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=error,
        )

    @staticmethod
    def _repair_prompt(original_prompt: str, response: str, error: str) -> str:
        return f"""{original_prompt}

[OUTPUT REPAIR]
The previous response violated the output contract.
Validation error: {error}
Previous response: {response}

Rewrite both reusable fields instead of repeating them verbatim. Preserve the
diagnosis if it is valid, but replace any prohibited target-specific wording
named in the validation error with broader category or behavioral language.
Return one corrected JSON object only.
"""
