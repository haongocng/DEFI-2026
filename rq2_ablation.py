"""Run one fixed RQ2 ablation on Bundle without a validation pass.

The configurations are hypotheses fixed before test evaluation.  Skipping
validation saves quota and is valid here because this script does not select or
tune a configuration.  The full dual-memory result is intentionally reused
from the existing main experiment rather than called again.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys


MODEL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_ROOT / "src"))

from drap.agents import Analyst, Reasoner
from drap.config import LifecycleConfig, RetrievalConfig
from drap.data import load_dataset
from drap.embeddings import SentenceTransformerEmbedder
from drap.llm import HeuristicLLMClient, TimelyLLMClient
from drap.memory import RuleMemory
from drap.metrics import ranking_metrics
from drap.pipeline import DRAPPipeline
from drap.prompts import load_base_prompt
from drap.regime import RegimeConfig, RegimeDetector
from drap.retrieval import RuleRetriever
from drap.tracing import LLMTraceRecorder


MODES = ("llm_only", "single_ltm", "dual_no_regime")


class PersistentSingleMemory(RuleMemory):
    """Single persistent rule pool represented by LTM with no transitions."""

    def add_rule(self, proposed, current_step, failed_rule_ids=()):
        decision = super().add_rule(proposed, current_step, failed_rule_ids)
        stored = self.by_id(proposed.id)
        if stored is not None and stored.active and stored.status == "active":
            stored.memory_type = "LTM"
        return decision

    def apply_lifecycle(self, current_step: int) -> None:
        # Keep utility auditable but disable promote/demote/archive transitions.
        self.current_step = max(self.current_step, current_step)
        for rule in self.active_rules():
            rule.last_utility = rule.utility_at(current_step, self.utility)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_records(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _configuration(mode: str) -> tuple[int, RetrievalConfig, type[RuleMemory], bool]:
    if mode == "llm_only":
        return (
            1,
            RetrievalConfig(
                stm_k=0,
                ltm_k=0,
                total_k=0,
                lambda_similarity=1.0,
                lambda_regime=0.0,
                lambda_utility=0.0,
            ),
            RuleMemory,
            False,
        )
    if mode == "single_ltm":
        return (
            3,
            RetrievalConfig(
                stm_k=0,
                ltm_k=5,
                total_k=5,
                lambda_similarity=0.65,
                lambda_regime=0.25,
                lambda_utility=0.10,
            ),
            PersistentSingleMemory,
            True,
        )
    if mode == "dual_no_regime":
        # Remove regime match and renormalize the remaining 0.65/0.10 weights.
        return (
            1,
            RetrievalConfig(
                stm_k=2,
                ltm_k=3,
                total_k=5,
                lambda_similarity=13.0 / 15.0,
                lambda_regime=0.0,
                lambda_utility=2.0 / 15.0,
            ),
            RuleMemory,
            True,
        )
    raise ValueError(f"Unknown mode: {mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one fixed Bundle RQ2 ablation")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--train", default="MODEL/datasets/bundle/train_50.json")
    parser.add_argument("--test", default="MODEL/datasets/bundle/test_seed_42.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--llm", choices=("heuristic", "timely"), default="timely")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-size", type=int, default=50)
    parser.add_argument("--success-k", type=int, default=3)
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--redact-llm-trace-content", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--duplicate-policy",
        choices=("stable-local", "llm-repair"),
        default="stable-local",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train = load_dataset(args.train)
    test = load_dataset(args.test)
    rng = random.Random(args.seed)
    calibration = rng.sample(train, min(args.calibration_size, len(train)))
    rng.shuffle(calibration)
    regime_clusters, retrieval_config, memory_class, calibrate_memory = (
        _configuration(args.mode)
    )
    embedder = SentenceTransformerEmbedder(
        args.embedding_model,
        local_files_only=not args.allow_model_download,
    )
    detector = RegimeDetector(
        embedder,
        RegimeConfig(n_clusters=regime_clusters, seed=args.seed),
    ).fit([sample.session for sample in train])
    retriever = RuleRetriever(embedder, retrieval_config)
    lifecycle = LifecycleConfig()
    memory = memory_class(
        output / "rules.json",
        retriever,
        lifecycle=lifecycle,
        utility=retrieval_config.utility,
        reset=True,
    )
    recorder = LLMTraceRecorder(
        output / "llm_traces.jsonl",
        include_content=not args.redact_llm_trace_content,
        reset=True,
    )
    client = (
        TimelyLLMClient(model=args.model)
        if args.llm == "timely"
        else HeuristicLLMClient()
    )
    analyst_event_path = output / "analyst_events.jsonl"
    calibration_prediction_path = output / "calibration_predictions.jsonl"
    analyst_event_path.write_text("", encoding="utf-8")
    calibration_prediction_path.write_text("", encoding="utf-8")
    pipeline = DRAPPipeline(
        Reasoner(
            client,
            base_prompt=load_base_prompt(),
            trace_recorder=recorder,
            duplicate_policy=args.duplicate_policy.replace("-", "_"),
        ),
        Analyst(client, trace_recorder=recorder),
        memory,
        retriever,
        detector,
        success_k=args.success_k,
        error_path=output / "errors.jsonl",
        analyst_event_path=analyst_event_path,
        calibration_prediction_path=calibration_prediction_path,
    )

    if calibrate_memory:
        calibration_records = pipeline.calibrate(
            calibration,
            show_progress=not args.no_progress,
            progress_description=f"RQ2 {args.mode} calibration",
        )
    else:
        calibration_records = []
        memory.save()

    test_metrics, test_records = pipeline.evaluate(
        test,
        phase="test",
        show_progress=not args.no_progress,
        progress_description=f"RQ2 {args.mode} test",
    )
    recorder.write_summary(output / "llm_trace_summary.json")
    detector.save(output / "regime_detector.json")
    _write_json(output / "regime_diagnostics.json", detector.diagnostics)
    _write_json(output / "calibration_metrics.json", ranking_metrics(calibration_records))
    _write_records(output / "test_predictions.jsonl", test_records)
    _write_json(output / "test_metrics.json", test_metrics)
    _write_json(
        output / "run_config.json",
        {
            "experiment": "RQ2 Bundle ablation",
            "mode": args.mode,
            "seed": args.seed,
            "train_path": args.train,
            "test_path": args.test,
            "validation_skipped": True,
            "configuration_fixed_before_test": True,
            "calibration_enabled": calibrate_memory,
            "calibration_size": len(calibration) if calibrate_memory else 0,
            "success_k": args.success_k,
            "regime_clusters": regime_clusters,
            "retrieval": asdict(retrieval_config),
            "lifecycle": asdict(lifecycle),
            "memory_class": memory_class.__name__,
            "llm": {
                "client": type(client).__name__,
                "model": getattr(client, "model", None),
                "structured_output": getattr(client, "structured_output", None),
                "duplicate_policy": args.duplicate_policy.replace("-", "_"),
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
                "mode": args.mode,
                "output": str(output),
                "test_metrics": test_metrics,
                "trace_summary": recorder.summary(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
