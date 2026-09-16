from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Literal

from .config import LifecycleConfig, UtilityConfig
from .models import MaintenanceDecision, Rule, utc_now
from .retrieval import RuleRetriever


MaintenancePolicy = Callable[[Rule, list[Rule]], MaintenanceDecision]


class RuleMemory:
    """Persistent dual-timescale memory with deterministic lifecycle updates."""

    def __init__(
        self,
        path: str | Path,
        retriever: RuleRetriever,
        lifecycle: LifecycleConfig | None = None,
        utility: UtilityConfig | None = None,
        reset: bool = False,
        # Compatibility inputs accepted from the original clean implementation.
        max_active_rules: int | None = None,
        novelty_threshold: float | None = None,
    ):
        self.path = Path(path)
        self.retriever = retriever
        self.lifecycle = lifecycle or LifecycleConfig()
        if max_active_rules is not None:
            self.lifecycle = replace(
                self.lifecycle,
                max_stm_rules=max_active_rules,
                max_ltm_rules=max_active_rules,
            )
        if novelty_threshold is not None:
            self.lifecycle = replace(
                self.lifecycle, novelty_threshold=novelty_threshold
            )
        self.utility = utility or retriever.config.utility
        self.current_step = 0
        self.rules = [] if reset else self._load()
        for rule in self.rules:
            self.retriever.ensure_rule_embedding(rule)
        self.next_creation_index = 1 + max(
            (rule.creation_index for rule in self.rules), default=-1
        )

    def _load(self) -> list[Rule]:
        if not self.path.exists():
            return []
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            rows = value.get("rules", [])
            self.current_step = int(value.get("current_step", 0))
        else:
            rows = value
        return [Rule.from_dict(row) for row in rows]

    def save(self) -> None:
        payload = {
            "schema_version": 2,
            "current_step": self.current_step,
            "lifecycle": asdict(self.lifecycle),
            "utility": asdict(self.utility),
            "rules": [rule.to_dict() for rule in self.rules],
        }
        self._atomic_write(
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        )

    def persisted_snapshot(self) -> bytes | None:
        return self.path.read_bytes() if self.path.exists() else None

    def restore_persisted_snapshot(self, snapshot: bytes | None) -> None:
        if snapshot is None:
            self.path.unlink(missing_ok=True)
            return
        self._atomic_write(snapshot)

    def _atomic_write(self, payload: bytes) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def active_rules(
        self, memory_type: Literal["STM", "LTM"] | None = None
    ) -> list[Rule]:
        return [
            rule
            for rule in self.rules
            if rule.active
            and rule.status == "active"
            and (memory_type is None or rule.memory_type == memory_type)
        ]

    def by_id(self, rule_id: str) -> Rule | None:
        return next((rule for rule in self.rules if rule.id == rule_id), None)

    def record_outcome(
        self, rule_ids: tuple[str, ...], success: bool, current_step: int
    ) -> None:
        """Mutate evidence only during calibration, after feedback is revealed."""
        self.current_step = max(self.current_step, current_step)
        for rule_id in rule_ids:
            rule = self.by_id(rule_id)
            if rule is None or not rule.active or rule.status != "active":
                continue
            rule.num_used += 1
            if success:
                rule.num_success += 1
            rule.last_used_step = current_step
            rule.last_utility = rule.utility_at(current_step, self.utility)
            rule.updated_at = utc_now()

    def add_rule(
        self,
        proposed: Rule,
        current_step: int,
        failed_rule_ids: tuple[str, ...] = (),
    ) -> MaintenanceDecision:
        """Novel rules enter STM; same-regime semantic duplicates are rejected."""
        self.current_step = max(self.current_step, current_step)
        proposed.memory_type = "STM"
        proposed.created_step = current_step
        proposed.last_used_step = None
        proposed.status = "active"
        proposed.active = True
        self.retriever.ensure_rule_embedding(proposed)
        proposed.last_utility = proposed.utility_at(current_step, self.utility)

        active_same_regime = [
            rule
            for rule in self.active_rules()
            if rule.regime_id == proposed.regime_id
        ]
        # Maintenance must inspect the complete bounded memory. Retrieval's top-k
        # is a prompt budget, not a duplicate-detection budget.
        nearest = self.retriever.nearest(
            self.active_rules(), proposed, top_k=None
        )
        exact = next(
            (
                rule
                for rule in active_same_regime
                if rule.condition.casefold().strip()
                == proposed.condition.casefold().strip()
                and rule.instruction.casefold().strip()
                == proposed.instruction.casefold().strip()
            ),
            None,
        )
        if exact is not None:
            return MaintenanceDecision(
                action="IGNORE",
                target_rule_ids=(exact.id,),
                reason="Exact same-regime duplicate rejected.",
            )

        same_condition = [
            rule
            for rule in active_same_regime
            if rule.condition.casefold().strip()
            == proposed.condition.casefold().strip()
        ]
        same_condition_by_id = {rule.id: rule for rule in same_condition}
        failed_same_condition = next(
            (
                same_condition_by_id[rule_id]
                for rule_id in failed_rule_ids
                if rule_id in same_condition_by_id
            ),
            None,
        )
        if failed_same_condition is not None:
            self._archive(failed_same_condition)
            failed_same_condition.replaced_by = proposed.id
            self._append_rule(proposed)
            return MaintenanceDecision(
                action="REPLACE",
                target_rule_ids=(failed_same_condition.id,),
                reason=(
                    "A same-condition rule participated in the failed ranking; "
                    "the corrective instruction replaced it."
                ),
            )
        if same_condition:
            self._append_rule(proposed)
            return MaintenanceDecision(
                action="ADD",
                target_rule_ids=tuple(rule.id for rule in same_condition),
                reason=(
                    "No conflicting same-condition rule was used in this failure; "
                    "the correction is retained as a separate STM hypothesis rather "
                    "than replacing or blocking existing rules."
                ),
            )

        redundant_instruction = next(
            (
                item.rule
                for item in nearest
                if item.rule.regime_id == proposed.regime_id
                and item.score >= self.lifecycle.novelty_threshold
                and item.rule.instruction.casefold().strip()
                == proposed.instruction.casefold().strip()
            ),
            None,
        )
        if redundant_instruction is not None:
            return MaintenanceDecision(
                action="IGNORE",
                target_rule_ids=(redundant_instruction.id,),
                reason=(
                    "Semantically overlapping rule with the same instruction "
                    "rejected."
                ),
            )

        # A semantically overlapping but behaviorally different instruction is kept
        # as a distinct STM hypothesis instead of being discarded without evidence.
        self._append_rule(proposed)
        return MaintenanceDecision(action="ADD", reason="Novel rule added to STM.")

    def _append_rule(self, rule: Rule) -> None:
        if rule.creation_index < 0:
            rule.creation_index = self.next_creation_index
            self.next_creation_index += 1
        self.rules.append(rule)

    def apply_lifecycle(self, current_step: int) -> None:
        self.current_step = max(self.current_step, current_step)
        demoted_this_step: set[str] = set()
        for rule in list(self.active_rules()):
            utility = rule.utility_at(current_step, self.utility)
            rule.last_utility = utility
            reference = (
                rule.last_used_step
                if rule.last_used_step is not None
                else rule.created_step
            )
            stale = current_step - reference > self.lifecycle.stale_window

            if rule.memory_type == "STM":
                if (
                    rule.num_used >= self.lifecycle.minimum_uses
                    and rule.success_rate >= self.lifecycle.promotion_success
                    and utility >= self.lifecycle.promotion_utility
                ):
                    rule.memory_type = "LTM"
                    rule.updated_at = utc_now()
                elif (
                    rule.num_used >= self.lifecycle.minimum_uses
                    and utility < self.lifecycle.deletion_utility
                ) or stale:
                    self._archive(rule)
            elif (
                rule.num_used >= self.lifecycle.minimum_uses
                and utility < self.lifecycle.demotion_utility
            ) or stale:
                rule.memory_type = "STM"
                demoted_this_step.add(rule.id)
                rule.updated_at = utc_now()

        self._enforce_budgets(current_step, demoted_this_step)

    def _archive(self, rule: Rule) -> None:
        rule.active = False
        rule.status = "archived"
        rule.updated_at = utc_now()

    def _enforce_budgets(
        self, current_step: int, demoted_this_step: set[str]
    ) -> None:
        ltm = self.active_rules("LTM")
        overflow = len(ltm) - self.lifecycle.max_ltm_rules
        if overflow > 0:
            weakest = sorted(
                ltm,
                key=lambda rule: (
                    rule.utility_at(current_step, self.utility),
                    rule.num_used,
                    rule.created_step,
                    rule.creation_index,
                    rule.condition.casefold(),
                ),
            )[:overflow]
            for rule in weakest:
                rule.memory_type = "STM"
                demoted_this_step.add(rule.id)
                rule.updated_at = utc_now()

        stm = self.active_rules("STM")
        overflow = len(stm) - self.lifecycle.max_stm_rules
        if overflow > 0:
            def weakness_key(rule: Rule) -> tuple[float, int, int, int, str]:
                return (
                    rule.utility_at(current_step, self.utility),
                    rule.num_used,
                    rule.created_step,
                    rule.creation_index,
                    rule.condition.casefold(),
                )

            # Prefer archiving rules that were already in STM. If the memory is
            # still over budget, capacity is a hard invariant and the weakest
            # just-demoted rules are archived too.
            existing_stm = sorted(
                (rule for rule in stm if rule.id not in demoted_this_step),
                key=weakness_key,
            )
            just_demoted = sorted(
                (rule for rule in stm if rule.id in demoted_this_step),
                key=weakness_key,
            )
            weakest = (existing_stm + just_demoted)[:overflow]
            for rule in weakest:
                self._archive(rule)

    def maintain(
        self,
        proposed: Rule,
        policy: MaintenancePolicy | None = None,
        current_step: int | None = None,
    ) -> MaintenanceDecision:
        """Compatibility wrapper; new pipeline uses deterministic ``add_rule``."""
        step = self.current_step if current_step is None else current_step
        if policy is None:
            return self.add_rule(proposed, step)

        nearest = self.retriever.nearest(self.active_rules(), proposed)
        similar = [
            item.rule
            for item in nearest
            if item.score >= self.lifecycle.novelty_threshold
        ]
        decision = policy(proposed, similar) if similar else MaintenanceDecision("ADD")
        if decision.action == "IGNORE":
            return decision
        if decision.action == "ADD":
            return self.add_rule(proposed, step)

        targets = [
            rule
            for rule_id in decision.target_rule_ids
            if (rule := self.by_id(rule_id)) is not None and rule.active
        ]
        if not targets:
            return self.add_rule(proposed, step)
        replacement = proposed
        if decision.action == "MERGE":
            if not decision.condition or not decision.instruction:
                return self.add_rule(proposed, step)
            replacement = Rule(
                condition=decision.condition,
                instruction=decision.instruction,
                regime_id=proposed.regime_id,
                source_sample_id=proposed.source_sample_id,
            )
        for target in targets:
            self._archive(target)
            target.replaced_by = replacement.id
        result = self.add_rule(replacement, step)
        return decision if result.action == "ADD" else result
