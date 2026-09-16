from __future__ import annotations

import json
import html
import re
from pathlib import Path
from typing import Any, Iterable

from .models import Candidate, DatasetSample, SessionFeedback, SessionInput


class DatasetFormatError(ValueError):
    pass


_ITEM_START = re.compile(r'(?:^|,\s*)(\d+)\.\s*(?:\\?")')


def _section(
    raw: str, labels: Iterable[str], next_labels: Iterable[str] = ()
) -> str:
    """Extract a labeled block without assuming item titles stay on one line."""
    lowered = raw.casefold()
    starts: list[tuple[int, int]] = []
    for label in labels:
        marker = label.casefold() + ":"
        position = lowered.find(marker)
        if position >= 0:
            starts.append((position, len(marker)))
    if not starts:
        return ""
    position, marker_length = min(starts)
    start = position + marker_length
    end = len(raw)
    for label in next_labels:
        next_position = lowered.find(label.casefold() + ":", start)
        if next_position >= 0:
            end = min(end, next_position)
    return raw[start:end].strip()


def _strip_outer_brackets(value: str) -> str:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        return value[1:-1]
    return value


def parse_numbered_items(value: str) -> tuple[Candidate, ...]:
    """Parse ``1.\"title\", 2.\"title\"`` without splitting title commas."""
    content = _strip_outer_brackets(value)
    starts = list(_ITEM_START.finditer(content))
    items: list[Candidate] = []

    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(content)
        text = content[match.end() : end].strip()
        if text.startswith(","):
            text = text[1:].lstrip()
        text = re.sub(r'(?:\\?")\s*,?\s*$', "", text).strip()
        text = text.replace(r'\"', '"').replace(r"\'", "'")
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            items.append(Candidate(id=int(match.group(1)), text=text))

    return tuple(items)


def _normalize_item(value: str) -> str:
    value = value.replace(r'\"', '"').replace(r"\'", "'")
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value).strip().casefold()
    return value.strip('"')


def _resolve_target_id(
    row: dict[str, Any], candidates: tuple[Candidate, ...], target_text: str
) -> int:
    normalized_target = _normalize_item(target_text)
    exact = [c.id for c in candidates if _normalize_item(c.text) == normalized_target]
    raw_index = row.get("target_index", row.get("target_id"))
    target_id: int | None = None
    if raw_index is not None:
        try:
            target_id = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise DatasetFormatError(f"Invalid target index: {raw_index!r}") from exc
        if not any(candidate.id == target_id for candidate in candidates):
            raise DatasetFormatError(
                f"Target index {target_id} is not present in the candidate set"
            )

    if len(exact) == 1:
        if target_id is not None and target_id != exact[0]:
            raise DatasetFormatError(
                f"Target text resolves to ID {exact[0]} but target_index is {target_id}"
            )
        return exact[0]
    if target_id is not None and (not exact or target_id in exact):
        return target_id

    raise DatasetFormatError(
        f"Target {target_text!r} is not uniquely present in the candidate set"
    )


def parse_sample(row: dict[str, Any], position: int = 0) -> DatasetSample:
    raw = str(row.get("input", ""))
    history_text = _section(
        raw, ("Current session interactions",), next_labels=("Candidate set",)
    )
    candidate_text = _section(raw, ("Candidate set",))
    history = tuple(item.text for item in parse_numbered_items(history_text))
    candidates = parse_numbered_items(candidate_text)

    if not candidates:
        raise DatasetFormatError("Candidate set is empty or malformed")
    candidate_ids = [candidate.id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise DatasetFormatError("Candidate IDs must be unique")

    target_text = html.unescape(
        str(row.get("target", row.get("target_text", ""))).strip()
    )
    target_text = re.sub(r"\s+", " ", target_text).strip()
    if not target_text:
        raise DatasetFormatError("Target text is missing")
    target_id = _resolve_target_id(row, candidates, target_text)

    sample_id = str(row.get("sample_id", row.get("id", position)))
    return DatasetSample(
        session=SessionInput(
            sample_id=sample_id,
            history=history,
            candidates=candidates,
            raw_input=raw,
        ),
        feedback=SessionFeedback(target_id=target_id, target_text=target_text),
    )


def load_dataset(path: str | Path, strict: bool = True) -> list[DatasetSample]:
    source = Path(path)
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise DatasetFormatError(f"Expected a JSON list in {source}")

    samples: list[DatasetSample] = []
    errors: list[str] = []
    for position, row in enumerate(rows):
        try:
            samples.append(parse_sample(row, position))
        except (DatasetFormatError, TypeError) as exc:
            errors.append(f"row {position}: {exc}")

    if errors and strict:
        preview = "\n".join(errors[:10])
        raise DatasetFormatError(
            f"Failed to parse {len(errors)}/{len(rows)} rows from {source}:\n{preview}"
        )
    return samples
