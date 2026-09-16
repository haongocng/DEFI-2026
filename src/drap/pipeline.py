from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

from .agents import AgentOutputError, Analyst, Reasoner
from .memory import RuleMemory
from .metrics import ranking_metrics
from .models import (
    AnalystEvent,
    DatasetSample,
    PredictionRecord,
    RetrievalTrace,
    Rule,
    RuleProposal,
)
from .regime import RegimeDetector
from .retrieval import RetrievedRule, RuleRetriever
from .progress import track
from .tracing import AgentTraceContext


class DRAPPipeline:
    """Causal calibration and frozen recommendation evaluation."""

    def __init__(
        self,
        reasoner: Reasoner,
        analyst: Analyst,
        memory: RuleMemory,
        retriever: RuleRetriever,
        regime_detector: RegimeDetector,
        success_k: int = 5,
        error_path: str | Path | None = None,
        analyst_event_path: str | Path | None = None,
        calibration_prediction_path: str | Path | None = None,
    ):
        if not regime_detector.fitted:
            raise ValueError("regime_detector must be fit on train data")
        if success_k <= 0:
            raise ValueError("success_k must be positive")
        self.reasoner = reasoner
        self.analyst = analyst
        self.memory = memory
        self.retriever = retriever
        self.regime_detector = regime_detector
        self.success_k = success_k
        self.error_path = (
            Path(error_path)
            if error_path is not None
            else memory.path.with_name("errors.jsonl")
        )
        self.analyst_event_path = (
            Path(analyst_event_path)
            if analyst_event_path is not None
            else memory.path.with_name("analyst_events.jsonl")
        )
        self.calibration_prediction_path = (
            Path(calibration_prediction_path)
            if calibration_prediction_path is not None
            else memory.path.with_name("calibration_predictions.jsonl")
        )

    @staticmethod
    def _append_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _append_checkpoint(path: Path) -> tuple[bool, int]:
        existed = path.exists()
        return (existed, path.stat().st_size if existed else 0)

    @staticmethod
    def _restore_append_checkpoint(
        path: Path, checkpoint: tuple[bool, int]
    ) -> None:
        existed, size = checkpoint
        if not existed:
            path.unlink(missing_ok=True)
            return
        with path.open("r+b") as stream:
            stream.truncate(size)
            stream.flush()
            os.fsync(stream.fileno())

    def _record_error(
        self,
        *,
        stage: str,
        session_id: str,
        current_step: int,
        error: Exception,
    ) -> None:
        payload = {
            "stage": stage,
            "session_id": session_id,
            "current_step": current_step,
            "error_type": type(error).__name__,
            "message": str(error),
            "responses": list(getattr(error, "responses", ())),
        }
        self._append_json(self.error_path, payload)

    def _record_analyst_event(self, event: AnalystEvent) -> None:
        self._append_json(self.analyst_event_path, event.to_dict())

    def _record_calibration_prediction(self, record: PredictionRecord) -> None:
        self._append_json(self.calibration_prediction_path, record.to_dict())

    @staticmethod
    def _materialize_rule(
        proposal: RuleProposal,
        *,
        regime_id: int,
        current_step: int,
        sample_id: str,
    ) -> Rule:
        return Rule(
            condition=proposal.condition,
            instruction=proposal.instruction,
            regime_id=regime_id,
            memory_type="STM",
            created_step=current_step,
            source_sample_id=sample_id,
        )

    @staticmethod
    def _retrieval_trace(
        retrieved: Sequence[RetrievedRule],
    ) -> tuple[RetrievalTrace, ...]:
        return tuple(
            RetrievalTrace(
                rule_id=item.rule.id,
                memory_type=item.rule.memory_type,
                score=item.score,
                similarity=item.similarity,
                regime_match=item.regime_match,
                utility=item.utility,
            )
            for item in retrieved
        )

    def infer(
        self,
        sample: DatasetSample,
        *,
        current_step: int | None = None,
        phase: str = "inference",
    ) -> PredictionRecord:
        """Read-only inference; feedback is used only after the ranking exists."""
        step = self.memory.current_step if current_step is None else current_step
        session = sample.session
        feedback = sample.feedback
        regime = self.regime_detector.predict(session)
        retrieved = self.retriever.retrieve(
            self.memory.active_rules(), session, regime, step
        )
        try:
            output = self.reasoner.rank(
                session,
                [item.rule for item in retrieved],
                regime,
                trace_context=AgentTraceContext(
                    phase=phase,
                    current_step=step,
                ),
            )
        except AgentOutputError as exc:
            self._record_error(
                stage="reasoner",
                session_id=session.sample_id,
                current_step=step,
                error=exc,
            )
            raise
        success = (
            feedback.target_id in output.ranked_candidate_ids[: self.success_k]
        )
        target_rank = output.ranked_candidate_ids.index(feedback.target_id) + 1
        return PredictionRecord(
            sample_id=session.sample_id,
            target_id=feedback.target_id,
            predicted_ids=output.ranked_candidate_ids,
            retrieved_rule_ids=output.retrieved_rule_ids,
            success=success,
            selected_intent=output.selected_intent,
            target_rank=target_rank,
            retrieval_trace=self._retrieval_trace(retrieved),
            regime_id=regime.id,
            regime_name=regime.name,
        )

    def calibrate(
        self,
        samples: list[DatasetSample],
        *,
        show_progress: bool = False,
        progress_description: str = "calibration",
    ) -> list[PredictionRecord]:
        """Sequential updates at step s become available only from step s+1."""
        records: list[PredictionRecord] = []
        start_step = self.memory.current_step + 1
        visible_samples = track(
            samples,
            total=len(samples),
            description=progress_description,
            enabled=show_progress,
        )
        for current_step, sample in enumerate(visible_samples, start=start_step):
            session = sample.session
            feedback = sample.feedback
            regime = self.regime_detector.predict(session)
            retrieved = self.retriever.retrieve(
                self.memory.active_rules(), session, regime, current_step
            )
            retrieved_rules = [item.rule for item in retrieved]
            try:
                output = self.reasoner.rank(
                    session,
                    retrieved_rules,
                    regime,
                    trace_context=AgentTraceContext(
                        phase="calibration",
                        current_step=current_step,
                    ),
                )
            except AgentOutputError as exc:
                self._record_error(
                    stage="reasoner",
                    session_id=session.sample_id,
                    current_step=current_step,
                    error=exc,
                )
                raise

            # Ground truth is first consulted after Reasoner output is complete.
            success = (
                feedback.target_id
                in output.ranked_candidate_ids[: self.success_k]
            )
            target_rank = output.ranked_candidate_ids.index(feedback.target_id) + 1
            record = PredictionRecord(
                sample_id=session.sample_id,
                target_id=feedback.target_id,
                predicted_ids=output.ranked_candidate_ids,
                retrieved_rule_ids=output.retrieved_rule_ids,
                success=success,
                selected_intent=output.selected_intent,
                target_rank=target_rank,
                retrieval_trace=self._retrieval_trace(retrieved),
                regime_id=regime.id,
                regime_name=regime.name,
            )
            proposal: RuleProposal | None = None
            if not success:
                try:
                    proposal = self.analyst.induce(
                        session,
                        feedback,
                        output,
                        retrieved_rules,
                        regime,
                        computed_target_rank=target_rank,
                        trace_context=AgentTraceContext(
                            phase="calibration",
                            current_step=current_step,
                        ),
                    )
                except AgentOutputError as exc:
                    self._record_error(
                        stage="analyst",
                        session_id=session.sample_id,
                        current_step=current_step,
                        error=exc,
                    )
                    raise

            # No memory mutation occurs until every required model output for
            # this session has passed validation. A failed commit restores both
            # in-memory state and the persisted session artifacts.
            rules_before = deepcopy(self.memory.rules)
            step_before = self.memory.current_step
            creation_index_before = self.memory.next_creation_index
            persisted_before = self.memory.persisted_snapshot()
            prediction_checkpoint = self._append_checkpoint(
                self.calibration_prediction_path
            )
            event_checkpoint = self._append_checkpoint(self.analyst_event_path)
            proposed_rule: Rule | None = None
            decision = None
            try:
                self.memory.record_outcome(
                    output.retrieved_rule_ids, success, current_step
                )
                if proposal is not None:
                    proposed_rule = self._materialize_rule(
                        proposal,
                        regime_id=regime.id,
                        current_step=current_step,
                        sample_id=session.sample_id,
                    )
                    decision = self.memory.add_rule(
                        proposed_rule,
                        current_step,
                        failed_rule_ids=output.retrieved_rule_ids,
                    )
                self.memory.apply_lifecycle(current_step)
                self.memory.save()
                self._record_calibration_prediction(record)
                if (
                    proposal is not None
                    and proposed_rule is not None
                    and decision is not None
                ):
                    self._record_analyst_event(
                        AnalystEvent(
                            sample_id=session.sample_id,
                            current_step=current_step,
                            regime_id=regime.id,
                            target_id=feedback.target_id,
                            target_rank=target_rank,
                            selected_intent=output.selected_intent,
                            ranked_candidate_ids=output.ranked_candidate_ids,
                            retrieved_rule_ids=output.retrieved_rule_ids,
                            diagnosis=proposal.diagnosis,
                            proposed_condition=proposal.condition,
                            proposed_instruction=proposal.instruction,
                            proposed_rule_id=proposed_rule.id,
                            maintenance_action=decision.action,
                            affected_rule_ids=decision.target_rule_ids,
                            maintenance_reason=decision.reason,
                        )
                    )
            except Exception:
                self.memory.rules = rules_before
                self.memory.current_step = step_before
                self.memory.next_creation_index = creation_index_before
                rollback_errors: list[Exception] = []
                rollback_actions = (
                    lambda: self.memory.restore_persisted_snapshot(
                        persisted_before
                    ),
                    lambda: self._restore_append_checkpoint(
                        self.calibration_prediction_path, prediction_checkpoint
                    ),
                    lambda: self._restore_append_checkpoint(
                        self.analyst_event_path, event_checkpoint
                    ),
                )
                for rollback in rollback_actions:
                    try:
                        rollback()
                    except Exception as rollback_error:
                        rollback_errors.append(rollback_error)
                if rollback_errors:
                    raise RuntimeError(
                        "Calibration session commit failed and rollback was "
                        f"incomplete ({len(rollback_errors)} rollback error(s))"
                    ) from rollback_errors[0]
                raise

            records.append(record)

        return records

    def evaluate(
        self,
        samples: list[DatasetSample],
        *,
        phase: str = "evaluation",
        show_progress: bool = False,
        progress_description: str | None = None,
    ) -> tuple[dict[str, float], list[PredictionRecord]]:
        """Frozen validation/test: no counters, utilities or rules are changed."""
        visible_samples = track(
            samples,
            total=len(samples),
            description=progress_description or phase,
            enabled=show_progress,
        )
        records = [self.infer(sample, phase=phase) for sample in visible_samples]
        return ranking_metrics(records), records
