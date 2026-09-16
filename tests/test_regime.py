from pathlib import Path
import tempfile
import unittest

from drap.embeddings import HashingEmbedder
from drap.models import Candidate, SessionInput
from drap.regime import RegimeConfig, RegimeDetector


def make_sample(sample_id: str, history: tuple[str, ...]) -> SessionInput:
    return SessionInput(
        sample_id=sample_id,
        history=history,
        candidates=(Candidate(1, "Candidate A"), Candidate(2, "Candidate B")),
    )


class RegimeDetectorTests(unittest.TestCase):
    def test_fit_predict_persist_and_ignore_target(self) -> None:
        samples = [
            make_sample("a", ("horror game", "horror sequel", "horror expansion")),
            make_sample("b", ("tea", "green tea", "black tea")),
            make_sample("c", ("camera", "dress", "coffee")),
            make_sample("d", ("laptop", "laptop case", "usb adapter")),
        ]
        embedder = HashingEmbedder(dimensions=256)
        detector = RegimeDetector(
            embedder, RegimeConfig(n_clusters=2, seed=7)
        ).fit(samples)
        first = detector.predict(samples[0])

        same_history_other_target = make_sample("changed", samples[0].history)
        self.assertEqual(
            detector.extract_features(samples[0]),
            detector.extract_features(same_history_other_target),
        )
        self.assertIn(first.name, detector.labels)
        self.assertFalse(detector.diagnostics["labels_verified"])
        self.assertEqual(sum(detector.diagnostics["cluster_sizes"]), len(samples))
        self.assertEqual(len(detector.diagnostics["initialization_runs"]), 10)
        self.assertEqual(
            len(detector.diagnostics["representative_sessions"]), 2
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regime.json"
            detector.save(path)
            loaded = RegimeDetector.load(path, embedder)
            self.assertEqual(loaded.predict(samples[0]).id, first.id)
            self.assertEqual(loaded.predict(samples[0]).name, first.name)
            self.assertEqual(loaded.diagnostics, detector.diagnostics)


if __name__ == "__main__":
    unittest.main()
