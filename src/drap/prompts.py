from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Sequence

from .models import (
    BehavioralRegime,
    RankedOutput,
    Rule,
    SessionFeedback,
    SessionInput,
)


DEFAULT_BASE_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompt" / "base_prompt.txt"


def load_base_prompt(path: str | Path | None = None) -> str:
    if path is not None:
        return Path(path).read_text(encoding="utf-8").strip()
    if DEFAULT_BASE_PROMPT_PATH.exists():
        return DEFAULT_BASE_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        resources.files("drap")
        .joinpath("resources/base_prompt.txt")
        .read_text(encoding="utf-8")
        .strip()
    )


def reasoner_prompt(
    session: SessionInput,
    rules: Sequence[Rule],
    regime: BehavioralRegime,
    *,
    base_prompt: str | None = None,
    include_regime_name: bool = False,
) -> str:
    candidates = "\n".join(f"{item.id}. {item.text}" for item in session.candidates)
    guidance = (
        "\n".join(
            f"- Applicable context: {rule.condition}\n  Guidance: {rule.instruction}"
            for rule in rules
        )
        if rules
        else "- No context-specific guidance was retrieved for this session."
    )
    regime_section = (
        "\n[BEHAVIORAL CONTEXT]\n"
        f"The current session behavior is best characterized as: {regime.name}.\n"
        "Treat this label as auxiliary evidence, not as a guaranteed fact.\n"
        if include_regime_name
        else ""
    )
    return f"""{base_prompt or load_base_prompt()}
{regime_section}

[RETRIEVED GUIDANCE]
{guidance}

[CURRENT SESSION INTERACTIONS — OLDEST TO NEWEST]
{session.context}

[CANDIDATE SET]
{candidates}
"""


def analyst_prompt(
    session: SessionInput,
    feedback: SessionFeedback,
    output: RankedOutput,
    retrieved_rules: Sequence[Rule],
    regime: BehavioralRegime,
    *,
    computed_target_rank: int | None = None,
    include_regime_name: bool = False,
) -> str:
    target_rank = (
        computed_target_rank
        if computed_target_rank is not None
        else output.ranked_candidate_ids.index(feedback.target_id) + 1
    )
    candidates = [{"id": c.id, "text": c.text} for c in session.candidates]
    guidance = [
        {"condition": rule.condition, "instruction": rule.instruction}
        for rule in retrieved_rules
    ]
    regime_section = (
        f"\n[BEHAVIORAL CONTEXT]\n{regime.name}\n"
        if include_regime_name
        else ""
    )
    return f"""You are the Analyst in a regime-aware recommendation system.
The Reasoner has already produced its ranking. Diagnose this failed prediction
and induce exactly one reusable corrective rule. Do not memorize the sample,
mention the ground-truth ID in the instruction, or create a rule that only works
for one product title.

The diagnosis is audit-only and may refer to the feedback item. The reusable
condition and instruction must use only generic product categories, behavioral
patterns, compatibility relationships, or general attributes. They must not
copy any brand/manufacturer name, model number/code, catalog-specific phrase, or
distinctive phrase from the ground-truth title. Do not include brand/model
examples in reusable fields.

[SESSION INTERACTIONS]
{session.context}
{regime_section}

[CANDIDATES]
{json.dumps(candidates, ensure_ascii=False)}

[RETRIEVED GUIDANCE USED BY THE REASONER]
{json.dumps(guidance, ensure_ascii=False)}

[REASONER RESULT]
Selected intent: {output.selected_intent}
Ranked candidate IDs: {list(output.ranked_candidate_ids)}

[GROUND-TRUTH FEEDBACK — REVEALED ONLY AFTER PREDICTION]
ID: {feedback.target_id}
Item: {feedback.target_text}
Computed target rank: {target_rank}

The computed target rank is authoritative. Do not confuse the candidate ID with
its rank in the Reasoner output.

The diagnosis must briefly explain the observed failure using evidence from this
session. It is audit metadata, not reusable guidance. The condition and
instruction must remain reusable and must not mention the ground-truth ID.

Return one JSON object and no additional text. All three values must be non-empty
JSON strings:
{{
  "diagnosis": "a brief evidence-based explanation of why the target was under-ranked",
  "condition": "a reusable category-level or behavioral pattern",
  "instruction": "a generic actionable reranking principle without brands or models"
}}
"""
