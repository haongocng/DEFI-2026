from pathlib import Path
import json
import unittest

from drap.data import DatasetFormatError, load_dataset, parse_sample


class DataParserTests(unittest.TestCase):
    def test_escaped_quotes_and_commas_inside_titles(self) -> None:
        row = {
            "input": (
                'Current session interactions: [1.\\"First, Deluxe Edition\\", '
                '2.\\"Second Item\\"]\n'
                'Candidate Set: [1.\\"Wrong Item\\", '
                '2.\\"Target, Collector Edition\\"]'
            ),
            "target": "Target, Collector Edition",
            "target_index": 2,
        }
        sample = parse_sample(row, 7)
        self.assertEqual(sample.session.sample_id, "7")
        self.assertEqual(
            sample.session.history, ("First, Deluxe Edition", "Second Item")
        )
        self.assertEqual(
            [c.text for c in sample.session.candidates],
            ["Wrong Item", "Target, Collector Edition"],
        )
        self.assertEqual(sample.feedback.target_id, 2)

    def test_all_paper_dataset_rows_parse_without_silent_filtering(self) -> None:
        project = Path(__file__).resolve().parents[1]
        datasets = {
            "games": ("datasets/games/game_test_seed_42.json", 1000),
            "bundle": ("datasets/bundle/test_seed_42.json", 238),
            "gift_cards": ("datasets/gift_cards/test_dataset.json", 1000),
            "gift_cards_clean": (
                "datasets/gift_cards_clean/test_seed_42.json",
                1000,
            ),
            "ml-1m": ("datasets/ml-1m/test_seed_42.json", 1000),
        }
        for name, (relative, expected) in datasets.items():
            with self.subTest(name=name):
                samples = load_dataset(project / relative)
                self.assertEqual(len(samples), expected)
                self.assertTrue(all(sample.session.candidates for sample in samples))
                self.assertTrue(
                    all(
                        sample.feedback.target_id
                        in {candidate.id for candidate in sample.session.candidates}
                        for sample in samples
                    )
                )

    def test_clean_gift_cards_splits_are_user_and_query_disjoint(self) -> None:
        project = Path(__file__).resolve().parents[1]
        files = (
            project / "datasets/gift_cards_clean/train_50.json",
            project / "datasets/gift_cards_clean/valid.json",
            project / "datasets/gift_cards_clean/test_seed_42.json",
        )
        seen_users: set[str] = set()
        seen_queries: set[tuple[tuple[str, ...], str]] = set()
        for path in files:
            rows = json.loads(path.read_text(encoding="utf-8"))
            samples = load_dataset(path)
            split_users = {str(row["user_hash"]) for row in rows}
            self.assertEqual(len(split_users), len(rows))
            self.assertFalse(seen_users.intersection(split_users))
            seen_users.update(split_users)
            for sample in samples:
                history = tuple(item.casefold() for item in sample.session.history)
                target = sample.feedback.target_text.casefold()
                self.assertNotIn(target, history)
                self.assertEqual(len(sample.session.candidates), 20)
                self.assertEqual(
                    len({candidate.text.casefold() for candidate in sample.session.candidates}),
                    20,
                )
                query = (history, target)
                self.assertNotIn(query, seen_queries)
                seen_queries.add(query)

    def test_multiline_candidate_title_is_not_dropped(self) -> None:
        row = {
            "input": (
                'Current session interactions: [1."History"]\n'
                'Candidate Set: [1."First line\nsecond line", 2."Target"]'
            ),
            "target": "Target",
            "target_index": 2,
        }
        sample = parse_sample(row)
        self.assertEqual(len(sample.session.candidates), 2)
        self.assertEqual(
            sample.session.candidates[0].text, "First line second line"
        )

    def test_conflicting_target_text_and_index_is_rejected(self) -> None:
        row = {
            "input": (
                'Current session interactions: [1."History"]\n'
                'Candidate Set: [1."Actual Target", 2."Other"]'
            ),
            "target": "Actual Target",
            "target_index": 2,
        }
        with self.assertRaises(DatasetFormatError):
            parse_sample(row)


if __name__ == "__main__":
    unittest.main()
