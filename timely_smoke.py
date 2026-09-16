"""Run one or two real-Timely calibration sessions before a full experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from drap.agents import Analyst, Reasoner
from drap.config import LifecycleConfig, RetrievalConfig
from drap.data import load_dataset
from drap.embeddings import SentenceTransformerEmbedder
from drap.llm import TimelyLLMClient
from drap.memory import RuleMemory
from drap.metrics import ranking_metrics
from drap.pipeline import DRAPPipeline
from drap.regime import RegimeConfig, RegimeDetector
from drap.retrieval import RuleRetriever
from drap.tracing import LLMTraceRecorder


def _session_count(value: str) -> int:
    count = int(value)
    if count not in (1, 2):
        raise argparse.ArgumentTypeError("sessions must be 1 or 2")
    return count


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run at most two real-Timely calibration sessions with semantic "
            "embedding, progress display and complete trace artifacts."
        )
    )
    parser.add_argument("--train", required=True, help="Train split for regime fitting.")
    parser.add_argument(
        "--calibration",
        required=True,
        help="Dataset from which the smoke calibration sessions are selected.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--sessions", type=_session_count, default=1)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regime-clusters", type=int, default=3)
    parser.add_argument("--success-k", type=int, default=5)
    parser.add_argument(
        "--model",
        default=None,
        help="Timely model; defaults to MODEL_NAME in MODEL/.env.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access instead of requiring a cached model.",
    )
    parser.add_argument(
        "--redact-llm-trace-content",
        action="store_true",
        help="Keep timing/status/hash metadata but omit prompt and response text.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--duplicate-policy",
        choices=("stable-local", "llm-repair"),
        default="stable-local",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.sample_offset < 0:
        raise ValueError("sample-offset must be non-negative")

    train = load_dataset(args.train)
    calibration_pool = load_dataset(args.calibration)
    selected = calibration_pool[
        args.sample_offset : args.sample_offset + args.sessions
    ]
    if len(selected) != args.sessions:
        raise ValueError(
            "calibration dataset does not contain the requested session range"
        )
    if args.regime_clusters > len(train):
        raise ValueError("regime-clusters cannot exceed the train sample count")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    error_path = output / "errors.jsonl"
    event_path = output / "analyst_events.jsonl"
    prediction_path = output / "calibration_predictions.jsonl"
    error_path.unlink(missing_ok=True)
    event_path.write_text("", encoding="utf-8")
    prediction_path.write_text("", encoding="utf-8")

    embedder = SentenceTransformerEmbedder(
        args.embedding_model,
        local_files_only=not args.allow_model_download,
    )
    detector = RegimeDetector(
        embedder,
        RegimeConfig(n_clusters=args.regime_clusters, seed=args.seed),
    ).fit([sample.session for sample in train])
    retrieval_config = RetrievalConfig()
    lifecycle_config = LifecycleConfig()
    retriever = RuleRetriever(embedder, retrieval_config)
    memory = RuleMemory(
        output / "rules.json",
        retriever,
        lifecycle=lifecycle_config,
        utility=retrieval_config.utility,
        reset=True,
    )
    recorder = LLMTraceRecorder(
        output / "llm_traces.jsonl",
        include_content=not args.redact_llm_trace_content,
        reset=True,
    )
    client = TimelyLLMClient(model=args.model)
    pipeline = DRAPPipeline(
        Reasoner(
            client,
            trace_recorder=recorder,
            duplicate_policy=args.duplicate_policy.replace("-", "_"),
        ),
        Analyst(client, trace_recorder=recorder),
        memory,
        retriever,
        detector,
        success_k=args.success_k,
        error_path=error_path,
        analyst_event_path=event_path,
        calibration_prediction_path=prediction_path,
    )

    try:
        records = pipeline.calibrate(
            selected,
            show_progress=not args.no_progress,
            progress_description="Timely calibration smoke",
        )
    finally:
        recorder.write_summary(output / "llm_trace_summary.json")

    metrics = ranking_metrics(records)
    detector.save(output / "regime_detector.json")
    _write_json(output / "regime_diagnostics.json", detector.diagnostics)
    _write_json(output / "calibration_metrics.json", metrics)
    _write_json(
        output / "run_config.json",
        {
            "smoke_only": True,
            "train_path": str(args.train),
            "calibration_path": str(args.calibration),
            "sample_offset": args.sample_offset,
            "sessions": args.sessions,
            "sample_ids": [record.sample_id for record in records],
            "seed": args.seed,
            "regime_clusters": args.regime_clusters,
            "success_k": args.success_k,
            "duplicate_policy": args.duplicate_policy.replace("-", "_"),
            "retrieval": asdict(retrieval_config),
            "lifecycle": asdict(lifecycle_config),
            "llm": {
                "client": type(client).__name__,
                "model": client.model,
                "structured_output": client.structured_output,
            },
            "embedder": {
                "class": type(embedder).__name__,
                "model": embedder.model_name,
                "dimensions": embedder.dimensions,
                "local_files_only": embedder.local_files_only,
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output),
                "metrics": metrics,
                "trace_summary": recorder.summary(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
