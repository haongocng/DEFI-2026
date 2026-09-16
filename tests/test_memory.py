from pathlib import Path
import tempfile
import unittest

from drap.embeddings import HashingEmbedder
from drap.config import LifecycleConfig, RetrievalConfig, UtilityConfig
from drap.memory import RuleMemory
from drap.models import MaintenanceDecision, Rule
from drap.retrieval import RuleRetriever


class MemoryTests(unittest.TestCase):
    def test_merge_archives_sources_and_keeps_audit_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = RuleRetriever(HashingEmbedder(dimensions=1024))
            memory = RuleMemory(
                Path(directory) / "rules.json",
                retriever,
                novelty_threshold=0.0,
            )
            old = Rule("recent horror item", "prefer horror")
            memory.rules.append(old)

            def merge(proposed: Rule, similar: list[Rule]) -> MaintenanceDecision:
                return MaintenanceDecision(
                    action="MERGE",
                    target_rule_ids=(similar[0].id,),
                    condition="recent genre repetition",
                    instruction="prefer candidates matching the most recent genre",
                )

            memory.maintain(Rule("recent action item", "prefer action"), merge)
            self.assertFalse(old.active)
            self.assertIsNotNone(old.replaced_by)
            self.assertEqual(len(memory.active_rules()), 1)
            self.assertEqual(memory.active_rules()[0].condition, "recent genre repetition")

    def test_rule_promotes_then_weak_ltm_demotes_before_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            utility = UtilityConfig(alpha=1.0, beta=0.0, gamma=0.0)
            retriever = RuleRetriever(
                HashingEmbedder(), RetrievalConfig(utility=utility)
            )
            lifecycle = LifecycleConfig(
                minimum_uses=2,
                promotion_success=0.5,
                promotion_utility=0.5,
                demotion_utility=0.4,
                deletion_utility=0.2,
                stale_window=100,
            )
            memory = RuleMemory(
                Path(directory) / "rules.json",
                retriever,
                lifecycle=lifecycle,
                utility=utility,
            )
            rule = Rule("repeated category", "prefer the category", regime_id=0)
            memory.add_rule(rule, current_step=1)
            memory.record_outcome((rule.id,), True, current_step=2)
            memory.record_outcome((rule.id,), True, current_step=3)
            memory.apply_lifecycle(3)
            self.assertEqual(rule.memory_type, "LTM")

            rule.num_used = 10
            rule.num_success = 0
            memory.apply_lifecycle(4)
            self.assertTrue(rule.active)
            self.assertEqual(rule.memory_type, "STM")
            memory.apply_lifecycle(5)
            self.assertFalse(rule.active)
            self.assertEqual(rule.status, "archived")

    def test_failed_same_condition_rule_is_replaced_by_correction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = RuleRetriever(HashingEmbedder())
            memory = RuleMemory(Path(directory) / "rules.json", retriever)
            old = Rule("repeated category", "prefer older category", regime_id=0)
            memory.add_rule(old, current_step=1)
            correction = Rule(
                "repeated category", "prefer most recent category", regime_id=0
            )
            decision = memory.add_rule(
                correction, current_step=2, failed_rule_ids=(old.id,)
            )
            self.assertEqual(decision.action, "REPLACE")
            self.assertFalse(old.active)
            self.assertEqual(old.replaced_by, correction.id)
            self.assertIn(correction, memory.active_rules())

    def test_failed_hypothesis_is_replaced_with_multiple_same_conditions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = RuleRetriever(HashingEmbedder())
            memory = RuleMemory(Path(directory) / "rules.json", retriever)
            unobserved = Rule(
                "repeated category", "prefer the oldest category", regime_id=0
            )
            failed = Rule(
                "repeated category", "prefer the broad category", regime_id=0
            )
            memory.add_rule(unobserved, current_step=1)
            memory.add_rule(failed, current_step=2)

            correction = Rule(
                "repeated category", "prefer the most recent category", regime_id=0
            )
            decision = memory.add_rule(
                correction, current_step=3, failed_rule_ids=(failed.id,)
            )

            self.assertEqual(decision.action, "REPLACE")
            self.assertEqual(decision.target_rule_ids, (failed.id,))
            self.assertTrue(unobserved.active)
            self.assertFalse(failed.active)
            self.assertEqual(failed.replaced_by, correction.id)
            self.assertTrue(correction.active)

    def test_unobserved_conflict_is_retained_as_stm_hypothesis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = RuleRetriever(HashingEmbedder())
            memory = RuleMemory(Path(directory) / "rules.json", retriever)
            old = Rule("repeated category", "prefer older category", regime_id=0)
            memory.add_rule(old, current_step=1)
            alternative = Rule(
                "repeated category", "prefer most recent category", regime_id=0
            )
            decision = memory.add_rule(alternative, current_step=2)
            self.assertEqual(decision.action, "ADD")
            self.assertEqual(len(memory.active_rules("STM")), 2)

    def test_semantic_duplicate_check_is_not_limited_to_retrieval_top_five(
        self,
    ) -> None:
        class MappingEmbedder:
            model_name = "fixture-mapping"

            def encode(self, text: str) -> list[float]:
                if text == "semantic duplicate":
                    return [0.8, 0.2]
                return [1.0, 0.0]

        with tempfile.TemporaryDirectory() as directory:
            retriever = RuleRetriever(MappingEmbedder())
            memory = RuleMemory(
                Path(directory) / "rules.json",
                retriever,
                lifecycle=LifecycleConfig(novelty_threshold=0.9),
            )
            for index in range(5):
                memory.rules.append(
                    Rule(
                        f"closer condition {index}",
                        f"different instruction {index}",
                        regime_id=0,
                    )
                )
            redundant = Rule(
                "semantic duplicate", "shared instruction", regime_id=0
            )
            memory.rules.append(redundant)

            decision = memory.add_rule(
                Rule("proposed condition", "shared instruction", regime_id=0),
                current_step=1,
            )

            self.assertEqual(decision.action, "IGNORE")
            self.assertEqual(decision.target_rule_ids, (redundant.id,))
            self.assertEqual(len(memory.rules), 6)

    def test_ltm_demotions_cannot_overrun_hard_stm_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            utility = UtilityConfig(alpha=1.0, beta=0.0, gamma=0.0)
            retriever = RuleRetriever(
                HashingEmbedder(), RetrievalConfig(utility=utility)
            )
            lifecycle = LifecycleConfig(
                minimum_uses=1,
                promotion_success=0.5,
                promotion_utility=0.5,
                demotion_utility=0.4,
                deletion_utility=0.2,
                stale_window=100,
                max_stm_rules=1,
                max_ltm_rules=10,
            )
            memory = RuleMemory(
                Path(directory) / "rules.json",
                retriever,
                lifecycle=lifecycle,
                utility=utility,
            )
            for index in range(2):
                rule = Rule(
                    f"condition {index}",
                    f"instruction {index}",
                    memory_type="LTM",
                    num_used=10,
                    num_success=0,
                    creation_index=index,
                )
                memory.rules.append(rule)
            memory.apply_lifecycle(current_step=1)
            self.assertLessEqual(len(memory.active_rules("STM")), 1)

    def test_add_rule_does_not_apply_global_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            utility = UtilityConfig(alpha=1.0, beta=0.0, gamma=0.0)
            retriever = RuleRetriever(
                HashingEmbedder(), RetrievalConfig(utility=utility)
            )
            lifecycle = LifecycleConfig(
                minimum_uses=1,
                promotion_success=0.9,
                promotion_utility=0.9,
                demotion_utility=0.4,
                deletion_utility=0.2,
                stale_window=100,
            )
            memory = RuleMemory(
                Path(directory) / "rules.json",
                retriever,
                lifecycle=lifecycle,
                utility=utility,
            )
            weak_ltm = Rule(
                "legacy condition",
                "legacy instruction",
                regime_id=0,
                memory_type="LTM",
                num_used=10,
                num_success=0,
                last_used_step=1,
            )
            memory.rules.append(weak_ltm)

            memory.add_rule(
                Rule("new condition", "new instruction", regime_id=0),
                current_step=1,
            )
            self.assertEqual(weak_ltm.memory_type, "LTM")
            self.assertTrue(weak_ltm.active)

            memory.apply_lifecycle(current_step=1)
            self.assertEqual(weak_ltm.memory_type, "STM")
            self.assertTrue(weak_ltm.active)


if __name__ == "__main__":
    unittest.main()
