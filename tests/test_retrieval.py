import unittest

from drap.embeddings import HashingEmbedder, cosine_similarity
from drap.config import RetrievalConfig
from drap.models import BehavioralRegime, Candidate, Rule, SessionInput
from drap.retrieval import RuleRetriever


class RetrievalTests(unittest.TestCase):
    def test_zero_use_stm_requires_matching_regime_until_observed(self) -> None:
        retriever = RuleRetriever(
            HashingEmbedder(),
            RetrievalConfig(stm_k=2, ltm_k=0, total_k=2),
        )
        session = SessionInput("s", ("generic item",), (Candidate(1, "item"),))
        cold_match = Rule("match", "match", regime_id=0, memory_type="STM")
        cold_mismatch = Rule(
            "mismatch", "mismatch", regime_id=1, memory_type="STM"
        )

        cold = retriever.retrieve(
            [cold_match, cold_mismatch],
            session,
            BehavioralRegime(0, "hidden"),
            current_step=1,
        )
        self.assertEqual([item.rule for item in cold], [cold_match])

        cold_mismatch.num_used = 1
        observed = retriever.retrieve(
            [cold_match, cold_mismatch],
            session,
            BehavioralRegime(0, "hidden"),
            current_step=2,
        )
        self.assertEqual({item.rule.id for item in observed}, {
            cold_match.id,
            cold_mismatch.id,
        })

    def test_cosine_and_top_k_retrieval(self) -> None:
        embedder = HashingEmbedder(dimensions=2048)
        retriever = RuleRetriever(
            embedder,
            RetrievalConfig(
                stm_k=1,
                ltm_k=0,
                total_k=1,
                lambda_similarity=1.0,
                lambda_regime=0.0,
                lambda_utility=0.0,
            ),
        )
        rules = [
            Rule(condition="recent horror games", instruction="prefer horror", regime_id=0),
            Rule(condition="restaurant gift cards", instruction="prefer dining cards", regime_id=1),
        ]
        sample = SessionInput(
            sample_id="s",
            history=("user recently selected a horror game",),
            candidates=(Candidate(1, "horror sequel"),),
        )
        result = retriever.retrieve(
            rules, sample, BehavioralRegime(0, "focused/consistent"), current_step=1
        )
        self.assertEqual(result[0].rule.instruction, "prefer horror")
        self.assertGreater(result[0].similarity, 0.5)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_separate_stm_ltm_budgets_and_regime_component(self) -> None:
        retriever = RuleRetriever(
            HashingEmbedder(),
            RetrievalConfig(
                stm_k=1,
                ltm_k=1,
                total_k=2,
                lambda_similarity=0.0,
                lambda_regime=1.0,
                lambda_utility=0.0,
            ),
        )
        sample = SessionInput(
            sample_id="s",
            history=("anything",),
            candidates=(Candidate(1, "candidate"),),
        )
        rules = [
            Rule("a", "a", regime_id=0, memory_type="STM"),
            Rule("b", "b", regime_id=1, memory_type="STM"),
            Rule("c", "c", regime_id=0, memory_type="LTM"),
            Rule("d", "d", regime_id=1, memory_type="LTM"),
        ]
        result = retriever.retrieve(
            rules, sample, BehavioralRegime(0, "focused"), current_step=1
        )
        self.assertEqual(len(result), 2)
        self.assertEqual({item.rule.memory_type for item in result}, {"STM", "LTM"})
        self.assertTrue(all(item.regime_match == 1.0 for item in result))

    def test_uuid_does_not_break_score_ties(self) -> None:
        retriever = RuleRetriever(
            HashingEmbedder(),
            RetrievalConfig(
                stm_k=1,
                ltm_k=0,
                total_k=1,
                lambda_similarity=1.0,
                lambda_regime=0.0,
                lambda_utility=0.0,
            ),
        )
        session = SessionInput("s", ("same",), (Candidate(1, "same"),))
        older = Rule(
            "same", "same", id="z-uuid", creation_index=0, regime_id=0
        )
        newer = Rule(
            "same", "same", id="a-uuid", creation_index=1, regime_id=0
        )
        result = retriever.retrieve(
            [newer, older], session, BehavioralRegime(0, "hidden"), current_step=1
        )
        self.assertEqual(result[0].rule.id, "z-uuid")


if __name__ == "__main__":
    unittest.main()
