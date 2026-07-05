"""Phase 2B falsifier receipts with an explicit epistemic effect.

A receipt is the JSON record of one falsifier attack on one candidate, plus
what that attack means epistemically. The core rule: passing a check is not
proof. A falsifier that fails to kill a candidate contributes "supports" only
when the finding as a whole was committed; otherwise its effect is "none".

epistemic_effect values:
- supports    — the check passed AND the finding was committed (robust_relation)
- refutes     — the check killed the candidate with evidence actively against it
                (the finding classified as falsified_candidate → public "rejected")
- challenges  — the check killed the candidate, but the kill means "did not clear
                the bar" rather than "evidence against" (not_supported /
                inconclusive / functional_form_rejected / no_incremental_value /
                regime_dependent classifications)
- none        — the check passed but the finding was not committed, or the
                falsifier skipped this candidate
- unknown     — the attack record is missing the fields needed to judge it
"""
from __future__ import annotations

from typing import Any

SUPPORTS = "supports"
REFUTES = "refutes"
CHALLENGES = "challenges"
NONE = "none"
UNKNOWN = "unknown"

EPISTEMIC_EFFECTS = {SUPPORTS, REFUTES, CHALLENGES, NONE, UNKNOWN}

# Finding types whose kills carry evidence actively against the hypothesis.
_REFUTING_FINDING_TYPES = {"falsified_candidate"}

# Finding types that mean the candidate was committed.
_COMMITTED_FINDING_TYPES = {"robust_relation"}


def falsifier_receipt(attack: dict[str, Any], *, finding_type: str) -> dict[str, Any]:
    """Build the receipt for one falsifier attack given the finding's final type."""
    killed = attack.get("killed")
    detail = attack.get("detail", {}) or {}
    skipped = "skipped" in detail

    if killed is None:
        effect = UNKNOWN
        status = "unknown"
    elif skipped:
        effect = NONE
        status = "skipped"
    elif killed:
        effect = REFUTES if finding_type in _REFUTING_FINDING_TYPES else CHALLENGES
        status = "killed"
    else:
        effect = SUPPORTS if finding_type in _COMMITTED_FINDING_TYPES else NONE
        status = "passed"

    reason = detail.get("reason") or detail.get("error") or detail.get("skipped")
    receipt: dict[str, Any] = {
        "stage": attack.get("name", "unknown_falsifier"),
        "status": status,
        "killed": bool(killed) if killed is not None else None,
        "reason": reason,
        "epistemic_effect": effect,
    }
    if attack.get("metric") is not None:
        receipt["metric"] = attack["metric"]
    for key in ("score", "median", "spread", "baseline_score", "composite_score"):
        if detail.get(key) is not None:
            receipt[key] = detail[key]
    return receipt


def finding_receipts(finding: dict[str, Any], finding_type: str) -> list[dict[str, Any]]:
    """Receipts for every falsifier attack recorded on one finding."""
    return [
        falsifier_receipt(attack, finding_type=finding_type)
        for attack in finding.get("falsifications", []) or []
    ]


def summarize_effects(receipts: list[dict[str, Any]]) -> dict[str, int]:
    counts = {effect: 0 for effect in sorted(EPISTEMIC_EFFECTS)}
    for receipt in receipts:
        effect = receipt.get("epistemic_effect", UNKNOWN)
        counts[effect if effect in EPISTEMIC_EFFECTS else UNKNOWN] += 1
    return counts
