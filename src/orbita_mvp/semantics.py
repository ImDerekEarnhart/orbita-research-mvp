"""Public epistemic semantics for case findings.

The raw `claims.status` column reflects the internal support-engine lifecycle
(every claim is born ``provisional`` and only flips to ``committed`` when a
support chain closes). That value is **not** a faithful public verdict: a
refuted candidate keeps refute-only evidence yet never leaves ``provisional``,
so reading `claims.status` makes a falsified hypothesis look unresolved-but-alive.

This module derives the public verdict from the *finding type* the discovery
pipeline assigned, which already encodes whether the hypothesis survived
falsification, was rejected, or is a structural artifact. It also pulls the
hypothesis text and the individual check scores apart from the affirmative
candidate statement so a rejected finding is never displayed as if it were true.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Finding type -> public epistemic state
# ---------------------------------------------------------------------------

# Public states surfaced to API consumers and the dashboard.
COMMITTED = "committed"
REJECTED = "rejected"
ARTIFACT = "artifact"
PROVISIONAL = "provisional"
UNRESOLVED = "unresolved"

PUBLIC_STATES = {COMMITTED, REJECTED, ARTIFACT, PROVISIONAL, UNRESOLVED}

# Canonical mapping required by the spec. Legacy finding-type spellings are
# retained so older rows in the persistent ledger still resolve correctly.
FINDING_TYPE_TO_STATE: dict[str, str] = {
    # Survived all falsifiers and committed.
    "robust_relation": COMMITTED,
    # Killed by at least one falsifier.
    "falsified_candidate": REJECTED,
    # Structural / transform tautology, not a scientific hypothesis.
    "artifact": ARTIFACT,
    "structural_relation": ARTIFACT,
    # Survived but did not reach the commit bar.
    "promising_candidate": PROVISIONAL,
    "candidate_relation": PROVISIONAL,  # legacy spelling
    # Could not be tested (insufficient data, undefined metric).
    "untestable_candidate": UNRESOLVED,
    "unresolved_candidate": UNRESOLVED,  # legacy spelling
    # Deterministic data-profile findings keep their own bucket as artifacts.
    "data_quality": ARTIFACT,
    "data_error": ARTIFACT,
    "artifact_guard": ARTIFACT,
}

# Colors the dashboard uses per public state.
STATE_COLOR = {
    COMMITTED: "green",
    REJECTED: "red",
    ARTIFACT: "orange",
    PROVISIONAL: "yellow",
    UNRESOLVED: "gray",
}


def public_state(finding_type: str | None) -> str:
    """Map an internal finding type to its public epistemic state.

    Unknown finding types are conservatively reported as ``unresolved`` rather
    than silently treated as committed.
    """
    if not finding_type:
        return UNRESOLVED
    return FINDING_TYPE_TO_STATE.get(finding_type, UNRESOLVED)


def is_rejected(finding_type: str | None) -> bool:
    return public_state(finding_type) == REJECTED


# ---------------------------------------------------------------------------
# Hypothesis / verdict separation
# ---------------------------------------------------------------------------

_VERDICT_REASON = {
    COMMITTED: "Survived baseline, held-out, and cross-seed falsification and met the commit threshold.",
    REJECTED: "Rejected by at least one falsification check.",
    ARTIFACT: "Structural or transform relationship between columns; not an independent scientific finding.",
    PROVISIONAL: "Survived falsification but did not reach the commit threshold.",
    UNRESOLVED: "Could not be conclusively tested with the available data.",
}


def _check_score(falsifications: list[dict[str, Any]], name: str) -> float | None:
    for attack in falsifications:
        if attack.get("name") == name:
            detail = attack.get("detail", {}) or {}
            # Each falsifier reports its primary metric under a stable key.
            if name == "baseline":
                return detail.get("score")
            if name == "held_out":
                return detail.get("score")
            if name == "cross_seed":
                return detail.get("median")
            return attack.get("metric")
    return None


def _cross_seed_summary(falsifications: list[dict[str, Any]]) -> dict[str, Any] | None:
    for attack in falsifications:
        if attack.get("name") == "cross_seed":
            detail = attack.get("detail", {}) or {}
            return {
                "median": detail.get("median"),
                "spread": detail.get("spread"),
                "seeds": detail.get("seeds"),
                "min_median": detail.get("min_median"),
                "max_spread": detail.get("max_spread"),
                "killed": bool(attack.get("killed")),
            }
    return None


def derive_finding_record(
    finding: dict[str, Any],
    finding_type: str,
    *,
    influence_warning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Split an affirmative candidate into hypothesis + verdict + check scores.

    The returned record is what the storage layer persists alongside the claim
    link and what the API returns. ``hypothesis_text`` is the affirmative
    candidate statement; ``verdict`` is the public state; for rejected findings
    the affirmative text must be presented as a *candidate hypothesis*, never as
    an established conclusion.
    """
    candidate = finding.get("candidate", {}) or {}
    verdict = finding.get("verdict", {}) or {}
    falsifications = finding.get("falsifications", []) or []
    state = public_state(finding_type)

    passed = [a.get("name") for a in falsifications if not a.get("killed")]
    failed = [a.get("name") for a in falsifications if a.get("killed")]

    record: dict[str, Any] = {
        "hypothesis_text": candidate.get("statement", ""),
        "finding_type": finding_type,
        "verdict": state,
        "verdict_reason": _VERDICT_REASON.get(state, ""),
        "is_candidate_hypothesis": state in {REJECTED, ARTIFACT, PROVISIONAL, UNRESOLVED},
        "passed_checks": passed,
        "failed_checks": failed,
        "candidate_score": verdict.get("score"),
        "baseline_score": _check_score(falsifications, "baseline"),
        "held_out_score": _check_score(falsifications, "held_out"),
        "cross_seed_summary": _cross_seed_summary(falsifications),
        "final_status": finding.get("final_status"),
    }
    if influence_warning:
        record["influence_warning"] = influence_warning
    return record
