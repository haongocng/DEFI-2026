from __future__ import annotations

import argparse
import json

from pathlib import Path

from .config import (
    ExperimentConfig,
    LifecycleConfig,
    RetrievalConfig,
    UtilityConfig,
)
from .embeddings import HashingEmbedder, SentenceTransformerEmbedder
from .experiment import run_experiment
from .llm import HeuristicLLMClient, TimelyLLMClient


def _int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run regime-aware dual-memory DRAP recommendation experiments"
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--llm",
        choices=("heuristic", "timely"),
        default="heuristic",
        help="heuristic is API-free smoke testing only; use timely for real experiments",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Timely model name; defaults to MODEL_NAME from the environment/.env.",
    )
    parser.add_argument("--seeds", type=_int_tuple, default=(0, 42, 2023))
    parser.add_argument("--calibration-size", type=int, default=50)
    parser.add_argument("--regime-k-grid", type=_int_tuple, default=(3,))
    parser.add_argument("--retrieval-k-grid", type=_int_tuple, default=(5,))
    parser.add_argument("--stm-k", type=int, default=2)
    parser.add_argument("--ltm-k", type=int, default=3)
    parser.add_argument("--success-k", type=int, default=5)
    parser.add_argument("--lambda-similarity", type=float, default=0.65)
    parser.add_argument("--lambda-regime", type=float, default=0.25)
    parser.add_argument("--lambda-utility", type=float, default=0.10)
    parser.add_argument("--minimum-retrieval-score", type=float, default=0.0)
    parser.add_argument("--utility-alpha", type=float, default=0.60)
    parser.add_argument("--utility-beta", type=float, default=0.20)
    parser.add_argument("--utility-gamma", type=float, default=0.20)
    parser.add_argument("--utility-kappa", type=float, default=5.0)
    parser.add_argument("--utility-tau", type=float, default=10.0)
    parser.add_argument("--minimum-rule-uses", type=int, default=3)
    parser.add_argument("--promotion-success", type=float, default=0.60)
    parser.add_argument("--promotion-utility", type=float, default=0.60)
    parser.add_argument("--demotion-utility", type=float, default=0.35)
    parser.add_argument("--deletion-utility", type=float, default=0.20)
    parser.add_argument("--stale-window", type=int, default=25)
    parser.add_argument("--max-stm-rules", type=int, default=100)
    parser.add_argument("--max-ltm-rules", type=int, default=100)
    parser.add_argument("--novelty-threshold", type=float, default=0.90)
    parser.add_argument("--base-prompt", default=None)
    parser.add_argument(
        "--include-regime-name",
        action="store_true",
        help="Also show the unverified human-readable regime label to the Reasoner.",
    )
    parser.add_argument(
        "--config-grid",
        default=None,
        help="JSON file containing complete validation candidate configurations.",
    )
    parser.add_argument(
        "--embedder",
        choices=("semantic", "hashing"),
        default="hashing",
        help="hashing is API-free smoke testing only; semantic is required for final results",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--embedding-local-files-only",
        action="store_true",
        help="Load an already-cached semantic model without Hugging Face network checks.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable calibration/validation/test progress bars.",
    )
    parser.add_argument(
        "--no-llm-tracing",
        action="store_true",
        help="Disable append-only per-attempt LLM trace artifacts.",
    )
    parser.add_argument(
        "--redact-llm-trace-content",
        action="store_true",
        help=(
            "Record timing, status and hashes but omit full prompts/responses from traces."
        ),
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("stable-local", "llm-repair"),
        default="stable-local",
        help=(
            "stable-local removes duplicate-only IDs deterministically without "
            "another API call; llm-repair preserves the legacy behavior"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    embedder = (
        SentenceTransformerEmbedder(
            args.embedding_model,
            local_files_only=args.embedding_local_files_only,
        )
        if args.embedder == "semantic"
        else HashingEmbedder()
    )
    client = (
        TimelyLLMClient(model=args.model)
        if args.llm == "timely"
        else HeuristicLLMClient()
    )
    utility = UtilityConfig(
        alpha=args.utility_alpha,
        beta=args.utility_beta,
        gamma=args.utility_gamma,
        kappa=args.utility_kappa,
        tau=args.utility_tau,
    )
    retrieval = RetrievalConfig(
        stm_k=args.stm_k,
        ltm_k=args.ltm_k,
        total_k=args.retrieval_k_grid[0],
        lambda_similarity=args.lambda_similarity,
        lambda_regime=args.lambda_regime,
        lambda_utility=args.lambda_utility,
        minimum_score=args.minimum_retrieval_score,
        utility=utility,
    )
    lifecycle = LifecycleConfig(
        minimum_uses=args.minimum_rule_uses,
        promotion_success=args.promotion_success,
        promotion_utility=args.promotion_utility,
        demotion_utility=args.demotion_utility,
        deletion_utility=args.deletion_utility,
        stale_window=args.stale_window,
        max_stm_rules=args.max_stm_rules,
        max_ltm_rules=args.max_ltm_rules,
        novelty_threshold=args.novelty_threshold,
    )
    candidate_configs: tuple[ExperimentConfig, ...] | None = None
    if args.config_grid:
        raw = json.loads(Path(args.config_grid).read_text(encoding="utf-8"))
        rows = raw.get("candidates", []) if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            raise ValueError("config-grid must contain a JSON list of candidates")
        candidate_configs = tuple(ExperimentConfig.from_dict(row) for row in rows)
    summary = run_experiment(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        output_dir=args.output,
        client=client,
        embedder=embedder,
        seeds=args.seeds,
        calibration_size=args.calibration_size,
        regime_k_grid=args.regime_k_grid,
        retrieval_k_grid=args.retrieval_k_grid,
        retrieval_config=retrieval,
        lifecycle_config=lifecycle,
        success_k=args.success_k,
        base_prompt_path=args.base_prompt,
        include_regime_name=args.include_regime_name,
        candidate_configs=candidate_configs,
        show_progress=not args.no_progress,
        trace_llm=not args.no_llm_tracing,
        trace_content=not args.redact_llm_trace_content,
        duplicate_policy=args.duplicate_policy.replace("-", "_"),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
