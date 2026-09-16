from __future__ import annotations

import json
import random
from dataclasses import asdict, replace
from pathlib import Path

from .agents import Analyst, Reasoner
from .config import ExperimentConfig, LifecycleConfig, RetrievalConfig
from .data import load_dataset
from .embeddings import Embedder
from .llm import LLMClient
from .memory import RuleMemory
from .metrics import aggregate_seed_metrics, ranking_metrics
from .pipeline import DRAPPipeline
from .prompts import load_base_prompt
from .regime import RegimeConfig, RegimeDetector
from .retrieval import RuleRetriever
from .tracing import LLMTraceRecorder


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_records(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _build_pipeline(
    *,
    train,
    calibration,
    run_dir: Path,
    client: LLMClient,
    embedder: Embedder,
    seed: int,
    regime_clusters: int,
    retrieval_config: RetrievalConfig,
    lifecycle_config: LifecycleConfig,
    success_k: int,
    base_prompt: str,
    include_regime_name: bool,
    show_progress: bool,
    progress_label: str,
    trace_llm: bool,
    trace_content: bool,
    duplicate_policy: str,
) -> tuple[DRAPPipeline, RegimeDetector, list]:
    run_dir.mkdir(parents=True, exist_ok=True)
    error_path = run_dir / "errors.jsonl"
    analyst_event_path = run_dir / "analyst_events.jsonl"
    calibration_prediction_path = run_dir / "calibration_predictions.jsonl"
    error_path.unlink(missing_ok=True)
    analyst_event_path.write_text("", encoding="utf-8")
    calibration_prediction_path.write_text("", encoding="utf-8")
    trace_recorder = (
        LLMTraceRecorder(
            run_dir / "llm_traces.jsonl",
            include_content=trace_content,
            reset=True,
        )
        if trace_llm
        else None
    )
    detector = RegimeDetector(
        embedder,
        RegimeConfig(
            n_clusters=regime_clusters,
            seed=seed,
            include_name_in_prompt=include_regime_name,
        ),
    ).fit([sample.session for sample in train])
    retriever = RuleRetriever(embedder, retrieval_config)
    memory = RuleMemory(
        run_dir / "rules.json",
        retriever,
        lifecycle=lifecycle_config,
        utility=retrieval_config.utility,
        reset=True,
    )
    pipeline = DRAPPipeline(
        Reasoner(
            client,
            base_prompt=base_prompt,
            include_regime_name=include_regime_name,
            trace_recorder=trace_recorder,
            duplicate_policy=duplicate_policy,
        ),
        Analyst(
            client,
            include_regime_name=include_regime_name,
            trace_recorder=trace_recorder,
        ),
        memory,
        retriever,
        detector,
        success_k=success_k,
        error_path=error_path,
        analyst_event_path=analyst_event_path,
        calibration_prediction_path=calibration_prediction_path,
    )
    calibration_records = pipeline.calibrate(
        calibration,
        show_progress=show_progress,
        progress_description=f"{progress_label} calibration",
    )
    return pipeline, detector, calibration_records


def _write_trace_summary(pipeline: DRAPPipeline, run_dir: Path) -> None:
    recorder = pipeline.reasoner.trace_recorder
    if recorder is not None:
        recorder.write_summary(run_dir / "llm_trace_summary.json")


def run_experiment(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    client: LLMClient,
    embedder: Embedder,
    seeds: tuple[int, ...] = (0, 42, 2023),
    calibration_size: int = 50,
    regime_k_grid: tuple[int, ...] = (3,),
    retrieval_k_grid: tuple[int, ...] = (5,),
    retrieval_config: RetrievalConfig | None = None,
    lifecycle_config: LifecycleConfig | None = None,
    success_k: int = 5,
    base_prompt_path: str | Path | None = None,
    include_regime_name: bool = False,
    candidate_configs: tuple[ExperimentConfig, ...] | None = None,
    show_progress: bool = False,
    trace_llm: bool = True,
    trace_content: bool = True,
    duplicate_policy: str = "stable_local",
    # Compatibility parameters from the base code.
    top_n: int | None = None,
    max_active_rules: int | None = None,
) -> dict:
    """Run one config directly, or select a grid on validation before testing."""
    del top_n
    if candidate_configs is None and (not regime_k_grid or not retrieval_k_grid):
        raise ValueError("regime_k_grid and retrieval_k_grid cannot be empty")

    train = load_dataset(train_path)
    validation = load_dataset(validation_path)
    test = load_dataset(test_path)
    base_retrieval = retrieval_config or RetrievalConfig()
    lifecycle = lifecycle_config or LifecycleConfig()
    if max_active_rules is not None:
        lifecycle = replace(
            lifecycle,
            max_stm_rules=max_active_rules,
            max_ltm_rules=max_active_rules,
        )
    if candidate_configs is None:
        candidate_configs = tuple(
            ExperimentConfig(
                name=f"regime_{regime_clusters}__rules_{total_k}",
                regime_clusters=regime_clusters,
                retrieval=replace(base_retrieval, total_k=total_k),
                lifecycle=lifecycle,
                include_regime_name=include_regime_name,
            )
            for regime_clusters in regime_k_grid
            for total_k in retrieval_k_grid
        )
    if not candidate_configs:
        raise ValueError("candidate_configs cannot be empty")
    names = [candidate.name for candidate in candidate_configs]
    if len(names) != len(set(names)):
        raise ValueError("candidate configuration names must be unique")
    if max(candidate.regime_clusters for candidate in candidate_configs) > len(train):
        raise ValueError("a regime cluster count exceeds the train sample count")
    base_prompt = load_base_prompt(base_prompt_path)
    root = Path(output_dir)
    seed_results: list[dict[str, float]] = []

    for seed in seeds:
        run_dir = root / f"seed_{seed}"
        rng = random.Random(seed)
        calibration = rng.sample(train, min(calibration_size, len(train)))
        rng.shuffle(calibration)

        single_configuration_fast_path = len(candidate_configs) == 1
        validation_reports: dict[str, dict[str, float]] = {}
        candidates_by_name = {candidate.name: candidate for candidate in candidate_configs}
        if single_configuration_fast_path:
            selected = candidate_configs[0]
            selected_name = selected.name
        else:
            choices: list[tuple[float, int, int, int, str]] = []
            for candidate_index, candidate in enumerate(candidate_configs):
                candidate_dir = run_dir / "selection" / candidate.name
                candidate_pipeline, _, _ = _build_pipeline(
                    train=train,
                    calibration=calibration,
                    run_dir=candidate_dir,
                    client=client,
                    embedder=embedder,
                    seed=seed,
                    regime_clusters=candidate.regime_clusters,
                    retrieval_config=candidate.retrieval,
                    lifecycle_config=candidate.lifecycle,
                    success_k=success_k,
                    base_prompt=base_prompt,
                    include_regime_name=candidate.include_regime_name,
                    show_progress=show_progress,
                    progress_label=f"seed {seed} select {candidate.name}",
                    trace_llm=trace_llm,
                    trace_content=trace_content,
                    duplicate_policy=duplicate_policy,
                )
                metrics, _ = candidate_pipeline.evaluate(
                    validation,
                    phase="validation_selection",
                    show_progress=show_progress,
                    progress_description=(
                        f"seed {seed} select {candidate.name} validation"
                    ),
                )
                _write_trace_summary(candidate_pipeline, candidate_dir)
                validation_reports[candidate.name] = metrics
                choices.append(
                    (
                        metrics.get("NDCG@10", 0.0),
                        -candidate.retrieval.total_k,
                        -candidate.regime_clusters,
                        -candidate_index,
                        candidate.name,
                    )
                )

            selected_name = max(choices)[4]
            selected = candidates_by_name[selected_name]

        # With multiple candidates, rebuild from empty memory so selection
        # memories are never reused for the final test. With one candidate this
        # is the first and only pipeline build.
        pipeline, detector, calibration_records = _build_pipeline(
            train=train,
            calibration=calibration,
            run_dir=run_dir,
            client=client,
            embedder=embedder,
            seed=seed,
            regime_clusters=selected.regime_clusters,
            retrieval_config=selected.retrieval,
            lifecycle_config=selected.lifecycle,
            success_k=success_k,
            base_prompt=base_prompt,
            include_regime_name=selected.include_regime_name,
            show_progress=show_progress,
            progress_label=f"seed {seed} final",
            trace_llm=trace_llm,
            trace_content=trace_content,
            duplicate_policy=duplicate_policy,
        )
        detector.save(run_dir / "regime_detector.json")
        _write_json(run_dir / "regime_diagnostics.json", detector.diagnostics)
        selected_validation_metrics, validation_records = pipeline.evaluate(
            validation,
            phase="validation",
            show_progress=show_progress,
            progress_description=f"seed {seed} final validation",
        )
        if single_configuration_fast_path:
            # Preserve the validation report shape without spending a separate
            # selection pass when there is nothing to select.
            validation_reports[selected_name] = selected_validation_metrics
        test_metrics, test_records = pipeline.evaluate(
            test,
            phase="test",
            show_progress=show_progress,
            progress_description=f"seed {seed} final test",
        )
        _write_trace_summary(pipeline, run_dir)
        seed_results.append(test_metrics)

        _write_json(
            run_dir / "calibration_metrics.json",
            ranking_metrics(calibration_records),
        )
        _write_json(
            run_dir / "validation_metrics.json",
            {
                "selection_mode": (
                    "single_configuration_fast_path"
                    if single_configuration_fast_path
                    else "validation_grid_search"
                ),
                "selection_runs": validation_reports,
                "selected_configuration": selected_name,
                "selected_validation_metrics": selected_validation_metrics,
                # Compatibility key retained for existing artifact readers.
                "rebuilt_selected_metrics": selected_validation_metrics,
            },
        )
        _write_json(run_dir / "test_metrics.json", test_metrics)
        _write_json(
            run_dir / "run_config.json",
            {
                "seed": seed,
                "execution": {
                    "single_configuration_fast_path": (
                        single_configuration_fast_path
                    ),
                    "selection_pass_performed": (
                        not single_configuration_fast_path
                    ),
                    "final_rebuild_performed": (
                        not single_configuration_fast_path
                    ),
                },
                "selected_configuration": asdict(selected),
                "success_k": success_k,
                "credit_assignment": {
                    "mode": "shared_outcome",
                    "per_rule_counterfactual": False,
                },
                "analyst_diagnosis": {
                    "required": True,
                    "used_for_retrieval": False,
                    "used_for_utility": False,
                },
                "llm": {
                    "client": type(client).__name__,
                    "model": getattr(client, "model", None),
                    "structured_output": getattr(
                        client, "structured_output", None
                    ),
                    "tracing_enabled": trace_llm,
                    "trace_content_recorded": trace_llm and trace_content,
                    "duplicate_policy": duplicate_policy,
                },
                "embedder": {
                    "class": type(embedder).__name__,
                    "model": getattr(embedder, "model_name", None),
                    "dimensions": getattr(embedder, "dimensions", None),
                    "local_files_only": getattr(
                        embedder, "local_files_only", None
                    ),
                },
                "candidate_configurations": [
                    asdict(candidate) for candidate in candidate_configs
                ],
            },
        )
        _write_records(run_dir / "calibration_predictions.jsonl", calibration_records)
        _write_records(run_dir / "validation_predictions.jsonl", validation_records)
        _write_records(run_dir / "test_predictions.jsonl", test_records)

    summary = {
        "seeds": list(seeds),
        "per_seed": seed_results,
        "aggregate": aggregate_seed_metrics(seed_results),
    }
    _write_json(root / "summary.json", summary)
    return summary
