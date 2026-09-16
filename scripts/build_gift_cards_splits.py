"""Build deterministic, user-disjoint Gift Cards recommendation splits.

The legacy processed files do not retain user or item provenance and may place
the target title in the visible history.  This builder starts from the Amazon
review and metadata JSONL files, uses each eligible user's final interaction as
the target, and rejects samples where that target was already visible.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path
import random
import re
from typing import Iterable


MODEL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = MODEL_ROOT.parent
DEFAULT_REVIEW_PATH = (
    WORKSPACE_ROOT
    / "DEFi-2026/agent_recommend/dataset/normal_dataset/amazon/Gift_Cards.jsonl"
)
DEFAULT_META_PATH = (
    WORKSPACE_ROOT
    / "DEFi-2026/agent_recommend/dataset/normal_dataset/amazon/meta_Gift_Cards.jsonl"
)
DEFAULT_OUTPUT_DIR = MODEL_ROOT / "datasets/gift_cards_clean"


@dataclass(frozen=True)
class Product:
    parent_asin: str
    title: str
    normalized_title: str


@dataclass(frozen=True)
class Interaction:
    timestamp: int
    source_index: int
    product: Product


@dataclass(frozen=True)
class UserExample:
    user_id: str
    history: tuple[Interaction, ...]
    target: Interaction
    interacted_parent_asins: frozenset[str]
    interacted_titles: frozenset[str]


def _normalize_title(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value.casefold()


def _clean_title(value: object) -> str:
    title = html.unescape(str(value or ""))
    return re.sub(r"\s+", " ", title).strip()


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_products(meta_path: Path) -> dict[str, Product]:
    products: dict[str, Product] = {}
    for row in _read_jsonl(meta_path):
        parent_asin = str(row.get("parent_asin") or "").strip()
        title = _clean_title(row.get("title"))
        if not parent_asin or not title:
            continue
        products[parent_asin] = Product(
            parent_asin=parent_asin,
            title=title,
            normalized_title=_normalize_title(title),
        )
    if not products:
        raise ValueError(f"No usable products found in {meta_path}")
    return products


def _load_user_interactions(
    review_path: Path,
    products: dict[str, Product],
) -> tuple[dict[str, list[Interaction]], int]:
    users: dict[str, list[Interaction]] = defaultdict(list)
    skipped_missing_metadata = 0
    for source_index, row in enumerate(_read_jsonl(review_path)):
        user_id = str(row.get("user_id") or "").strip()
        parent_asin = str(row.get("parent_asin") or "").strip()
        product = products.get(parent_asin)
        if not user_id or product is None:
            skipped_missing_metadata += 1
            continue
        try:
            timestamp = int(row.get("timestamp"))
        except (TypeError, ValueError):
            continue
        users[user_id].append(
            Interaction(
                timestamp=timestamp,
                source_index=source_index,
                product=product,
            )
        )
    for interactions in users.values():
        interactions.sort(key=lambda item: (item.timestamp, item.source_index))
    return dict(users), skipped_missing_metadata


def _collapse_history(
    interactions: list[Interaction],
    *,
    max_history: int,
) -> tuple[Interaction, ...]:
    # Keep the latest occurrence of each visible title, then restore chronology.
    latest_by_title: dict[str, Interaction] = {}
    for interaction in interactions:
        latest_by_title[interaction.product.normalized_title] = interaction
    collapsed = sorted(
        latest_by_title.values(),
        key=lambda item: (item.timestamp, item.source_index),
    )
    return tuple(collapsed[-max_history:])


def _eligible_users(
    users: dict[str, list[Interaction]],
    *,
    max_history: int,
) -> tuple[list[UserExample], dict[str, int]]:
    examples: list[UserExample] = []
    counts = {
        "users_total": len(users),
        "excluded_fewer_than_two_events": 0,
        "excluded_repeated_target_parent_asin": 0,
        "excluded_repeated_target_title": 0,
        "excluded_empty_history_after_collapse": 0,
    }
    for user_id, interactions in users.items():
        if len(interactions) < 2:
            counts["excluded_fewer_than_two_events"] += 1
            continue
        target = interactions[-1]
        prior = interactions[:-1]
        prior_parent_asins = {item.product.parent_asin for item in prior}
        if target.product.parent_asin in prior_parent_asins:
            counts["excluded_repeated_target_parent_asin"] += 1
            continue
        prior_titles = {item.product.normalized_title for item in prior}
        if target.product.normalized_title in prior_titles:
            counts["excluded_repeated_target_title"] += 1
            continue
        history = _collapse_history(prior, max_history=max_history)
        if not history:
            counts["excluded_empty_history_after_collapse"] += 1
            continue
        examples.append(
            UserExample(
                user_id=user_id,
                history=history,
                target=target,
                interacted_parent_asins=frozenset(
                    item.product.parent_asin for item in interactions
                ),
                interacted_titles=frozenset(
                    item.product.normalized_title for item in interactions
                ),
            )
        )
    counts["eligible_users"] = len(examples)
    return examples, counts


def _negative_pool(products: dict[str, Product]) -> tuple[Product, ...]:
    # Candidate titles are what the LLM sees, so keep one representative item
    # per normalized title to prevent ambiguous duplicate-title candidates.
    representative: dict[str, Product] = {}
    for product in sorted(products.values(), key=lambda item: item.parent_asin):
        representative.setdefault(product.normalized_title, product)
    return tuple(representative.values())


def _observable_query(example: UserExample) -> tuple[tuple[str, ...], str]:
    """Fields visible to a ranker before candidate construction."""
    return (
        tuple(item.product.normalized_title for item in example.history),
        example.target.product.normalized_title,
    )


def _quote_title(title: str) -> str:
    return title.replace("\\", "\\\\").replace('"', r'\"')


def _format_items(titles: Iterable[str]) -> str:
    return ", ".join(
        f'{index}."{_quote_title(title)}"'
        for index, title in enumerate(titles, start=1)
    )


def _user_hash(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


def _build_row(
    example: UserExample,
    *,
    split: str,
    split_index: int,
    negative_products: tuple[Product, ...],
    negatives_per_sample: int,
    rng: random.Random,
) -> dict:
    eligible_negatives = [
        product
        for product in negative_products
        if product.parent_asin not in example.interacted_parent_asins
        and product.normalized_title not in example.interacted_titles
        and product.normalized_title != example.target.product.normalized_title
    ]
    if len(eligible_negatives) < negatives_per_sample:
        raise ValueError(
            f"User {example.user_id} has only {len(eligible_negatives)} negatives"
        )
    negatives = rng.sample(eligible_negatives, negatives_per_sample)
    candidates = [example.target.product, *negatives]
    rng.shuffle(candidates)
    target_index = next(
        index
        for index, product in enumerate(candidates, start=1)
        if product.parent_asin == example.target.product.parent_asin
    )
    history_titles = [item.product.title for item in example.history]
    candidate_titles = [item.title for item in candidates]
    return {
        "sample_id": f"gift_cards_{split}_{split_index:04d}",
        "user_hash": _user_hash(example.user_id),
        "target": example.target.product.title,
        "target_index": target_index,
        "target_parent_asin": example.target.product.parent_asin,
        "target_timestamp": example.target.timestamp,
        "candidate_parent_asins": [item.parent_asin for item in candidates],
        "input": (
            f"Current session interactions: [{_format_items(history_titles)}]\n"
            f"Candidate set: [{_format_items(candidate_titles)}]"
        ),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_splits(
    *,
    review_path: Path,
    meta_path: Path,
    output_dir: Path,
    seed: int,
    train_size: int,
    validation_size: int,
    test_size: int,
    negatives_per_sample: int,
    max_history: int,
    overwrite: bool,
) -> dict:
    output_paths = {
        "train": output_dir / f"train_{train_size}.json",
        "validation": output_dir / "valid.json",
        "test": output_dir / f"test_seed_{seed}.json",
        "manifest": output_dir / "manifest.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing files: {joined}")

    products = _load_products(meta_path)
    users, skipped_missing_metadata = _load_user_interactions(review_path, products)
    eligible, eligibility_counts = _eligible_users(users, max_history=max_history)
    required = train_size + validation_size + test_size
    if len(eligible) < required:
        raise ValueError(f"Need {required} eligible users, found {len(eligible)}")

    rng = random.Random(seed)
    eligible.sort(key=lambda item: item.user_id)
    rng.shuffle(eligible)
    selected: list[UserExample] = []
    seen_queries: set[tuple[tuple[str, ...], str]] = set()
    for example in eligible:
        query = _observable_query(example)
        if query in seen_queries:
            continue
        seen_queries.add(query)
        selected.append(example)
        if len(selected) == required:
            break
    if len(selected) < required:
        raise ValueError(
            f"Need {required} unique observable queries, found {len(selected)}"
        )
    split_examples = {
        "train": selected[:train_size],
        "validation": selected[train_size : train_size + validation_size],
        "test": selected[train_size + validation_size :],
    }
    negative_products = _negative_pool(products)
    rows: dict[str, list[dict]] = {}
    for split, examples in split_examples.items():
        rows[split] = [
            _build_row(
                example,
                split=split,
                split_index=index,
                negative_products=negative_products,
                negatives_per_sample=negatives_per_sample,
                rng=rng,
            )
            for index, example in enumerate(examples)
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_paths["train"], rows["train"])
    _write_json(output_paths["validation"], rows["validation"])
    _write_json(output_paths["test"], rows["test"])
    manifest = {
        "schema_version": 1,
        "dataset": "Amazon Reviews 2023 - Gift Cards",
        "seed": seed,
        "split_policy": (
            "one final interaction per user; user-disjoint and observable-query-"
            "disjoint splits"
        ),
        "target_policy": (
            "actual final event; reject if its parent_asin or normalized title "
            "appeared earlier"
        ),
        "history_policy": (
            "prior events only; keep latest occurrence per normalized title; "
            f"retain the most recent {max_history} unique titles"
        ),
        "candidate_policy": (
            f"one target plus {negatives_per_sample} deterministic random unseen "
            "products; unique normalized candidate titles"
        ),
        "sizes": {name: len(value) for name, value in rows.items()},
        "products_with_metadata": len(products),
        "unique_negative_pool_titles": len(negative_products),
        "skipped_reviews_missing_metadata": skipped_missing_metadata,
        "eligibility": eligibility_counts,
        "eligible_unique_observable_queries": len(
            {_observable_query(example) for example in eligible}
        ),
        "source": {
            "reviews": str(review_path),
            "reviews_sha256": _sha256(review_path),
            "metadata": str(meta_path),
            "metadata_sha256": _sha256(meta_path),
        },
        "files": {name: str(path) for name, path in output_paths.items()},
    }
    _write_json(output_paths["manifest"], manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build clean deterministic Gift Cards recommendation splits"
    )
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEW_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--negatives", type=int, default=19)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ("train_size", "validation_size", "test_size", "negatives"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.max_history <= 0:
        raise ValueError("max-history must be positive")
    manifest = build_splits(
        review_path=args.reviews,
        meta_path=args.metadata,
        output_dir=args.output_dir,
        seed=args.seed,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        negatives_per_sample=args.negatives,
        max_history=args.max_history,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
