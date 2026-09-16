from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Literal


TraceStatus = Literal["accepted", "rejected", "client_error"]


@dataclass(frozen=True)
class AgentTraceContext:
    """Experiment coordinates attached to one Reasoner or Analyst call."""

    phase: str = "unspecified"
    current_step: int | None = None


class LLMTraceRecorder:
    """Append-only audit log for every bounded LLM attempt.

    The recorder sees prompts and model outputs only. API keys and request headers
    are owned by the client and therefore cannot be written to this artifact.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        include_content: bool = True,
        reset: bool = False,
    ):
        self.path = Path(path)
        self.include_content = include_content
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.write_text("", encoding="utf-8")

    def record(
        self,
        *,
        stage: str,
        context: AgentTraceContext,
        sample_id: str,
        regime_id: int,
        retrieved_rule_ids: tuple[str, ...],
        attempt: int,
        max_attempts: int,
        prompt: str,
        response: str | None,
        status: TraceStatus,
        latency_ms: float,
        error: str | None = None,
        output_normalization: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "phase": context.phase,
            "stage": stage,
            "sample_id": sample_id,
            "current_step": context.current_step,
            "regime_id": regime_id,
            "retrieved_rule_ids": list(retrieved_rule_ids),
            "attempt": attempt,
            "max_attempts": max_attempts,
            "is_repair": attempt > 1,
            "status": status,
            "latency_ms": round(max(0.0, latency_ms), 3),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
            "response_chars": len(response) if response is not None else 0,
            "error": error,
            "output_normalization": output_normalization,
        }
        if self.include_content:
            payload["prompt"] = prompt
            payload["response"] = response

        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def summary(self) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        if self.path.exists():
            for line_number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid trace JSON at {self.path}:{line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Trace row must be an object at {self.path}:{line_number}"
                    )
                events.append(value)

        latencies = sorted(float(event.get("latency_ms", 0.0)) for event in events)
        status_counts = Counter(str(event.get("status", "unknown")) for event in events)
        stage_counts = Counter(str(event.get("stage", "unknown")) for event in events)
        phase_counts = Counter(str(event.get("phase", "unknown")) for event in events)
        accepted_calls = sum(1 for event in events if event.get("status") == "accepted")
        repaired_calls = sum(
            1
            for event in events
            if event.get("status") == "accepted" and event.get("is_repair") is True
        )
        locally_normalized_calls = sum(
            1
            for event in events
            if event.get("output_normalization") is not None
        )
        return {
            "trace_file": self.path.name,
            "content_recorded": self.include_content,
            "attempts": len(events),
            "accepted_calls": accepted_calls,
            "accepted_after_repair": repaired_calls,
            "accepted_after_local_normalization": locally_normalized_calls,
            "counts_by_status": dict(sorted(status_counts.items())),
            "counts_by_stage": dict(sorted(stage_counts.items())),
            "counts_by_phase": dict(sorted(phase_counts.items())),
            "latency_ms": {
                "total": round(sum(latencies), 3),
                "mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": round(latencies[-1], 3) if latencies else 0.0,
            },
            "characters": {
                "prompt": sum(int(event.get("prompt_chars", 0)) for event in events),
                "response": sum(
                    int(event.get("response_chars", 0)) for event in events
                ),
            },
        }

    def write_summary(self, path: str | Path) -> dict[str, Any]:
        value = self.summary()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return value


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, math.ceil(quantile * len(values)) - 1))
    return round(values[index], 3)
