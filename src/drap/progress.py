from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar


T = TypeVar("T")


def track(
    items: Iterable[T],
    *,
    total: int,
    description: str,
    enabled: bool,
) -> Iterable[T]:
    """Return a tqdm iterator when requested, otherwise preserve the iterable."""

    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "Progress display requires tqdm; install the semantic extra or tqdm"
        ) from exc
    return tqdm(items, total=total, desc=description, unit="session", dynamic_ncols=True)
