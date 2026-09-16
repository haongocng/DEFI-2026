import json
from pathlib import Path
import tempfile
import unittest

from drap.agents import Reasoner
from drap.models import BehavioralRegime, Candidate, SessionInput
from drap.tracing import AgentTraceContext, LLMTraceRecorder


class RepairClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return "invalid json"
        return json.dumps(
            {"selected_intent": "fixture intent", "ranked_candidate_ids": [1, 2]}
        )


class TracingTests(unittest.TestCase):
    def test_every_attempt_is_traced_and_summarized(self) -> None:
        session = SessionInput(
            sample_id="trace-sample",
            history=("recent item",),
            candidates=(Candidate(1, "first"), Candidate(2, "second")),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = LLMTraceRecorder(root / "llm_traces.jsonl", reset=True)
            Reasoner(RepairClient(), trace_recorder=recorder).rank(
                session,
                [],
                BehavioralRegime(3, "hidden"),
                trace_context=AgentTraceContext(
                    phase="validation",
                    current_step=9,
                ),
            )

            rows = [
                json.loads(line)
                for line in recorder.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["status"] for row in rows], ["rejected", "accepted"])
            self.assertEqual([row["attempt"] for row in rows], [1, 2])
            self.assertFalse(rows[0]["is_repair"])
            self.assertTrue(rows[1]["is_repair"])
            self.assertEqual(rows[1]["phase"], "validation")
            self.assertEqual(rows[1]["sample_id"], "trace-sample")
            self.assertEqual(rows[1]["current_step"], 9)
            self.assertIn("[OUTPUT REPAIR]", rows[1]["prompt"])

            summary = recorder.write_summary(root / "llm_trace_summary.json")
            self.assertEqual(summary["attempts"], 2)
            self.assertEqual(summary["accepted_calls"], 1)
            self.assertEqual(summary["accepted_after_repair"], 1)
            self.assertEqual(summary["counts_by_status"]["rejected"], 1)
            self.assertEqual(summary["counts_by_phase"]["validation"], 2)
            self.assertEqual(summary["accepted_after_local_normalization"], 0)

    def test_local_duplicate_normalization_is_auditable(self) -> None:
        class DuplicateClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, prompt: str) -> str:
                self.calls += 1
                return json.dumps(
                    {
                        "selected_intent": "fixture intent",
                        "ranked_candidate_ids": [2, 2],
                    }
                )

        session = SessionInput(
            sample_id="normalized-trace",
            history=("recent item",),
            candidates=(Candidate(1, "first"), Candidate(2, "second")),
        )
        with tempfile.TemporaryDirectory() as directory:
            recorder = LLMTraceRecorder(
                Path(directory) / "llm_traces.jsonl", reset=True
            )
            client = DuplicateClient()
            output = Reasoner(client, trace_recorder=recorder).rank(
                session,
                [],
                BehavioralRegime(0, "hidden"),
                trace_context=AgentTraceContext(phase="test", current_step=4),
            )

            row = json.loads(recorder.path.read_text(encoding="utf-8"))
            normalization = row["output_normalization"]
            self.assertEqual(client.calls, 1)
            self.assertEqual(output.ranked_candidate_ids, (2, 1))
            self.assertEqual(row["status"], "accepted")
            self.assertFalse(row["is_repair"])
            self.assertEqual(normalization["duplicate_candidate_ids"], [2])
            self.assertEqual(normalization["missing_candidate_ids"], [1])
            self.assertEqual(
                recorder.summary()["accepted_after_local_normalization"], 1
            )

    def test_redacted_trace_keeps_measurements_but_not_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = LLMTraceRecorder(
                Path(directory) / "trace.jsonl",
                include_content=False,
                reset=True,
            )
            recorder.record(
                stage="reasoner",
                context=AgentTraceContext(phase="test", current_step=4),
                sample_id="sample",
                regime_id=0,
                retrieved_rule_ids=(),
                attempt=1,
                max_attempts=2,
                prompt="private prompt",
                response="private response",
                status="accepted",
                latency_ms=2.5,
            )
            row = json.loads(recorder.path.read_text(encoding="utf-8"))
            self.assertNotIn("prompt", row)
            self.assertNotIn("response", row)
            self.assertEqual(row["prompt_chars"], len("private prompt"))
            self.assertEqual(row["response_chars"], len("private response"))
            self.assertEqual(len(row["prompt_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
