"""Regression tests for the claim-import and dashboard semantics overhaul.

These pin the exact inconsistent API response that motivated the change:

* finding_type=falsified_candidate must never surface verdict=provisional;
* raw vs log-transform pairs must be classified as artifacts, not science;
* run counts must report the real generated-candidate total, never zero;
* a leverage-dominated raw relation must carry an influence warning.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from orbita_mvp import ResearchMVP
from orbita_mvp.artifacts import detect_structural_relations
from orbita_mvp.influence import linear_influence_warning
from orbita_mvp.semantics import (
    FINDING_TYPE_TO_STATE,
    derive_finding_record,
    public_state,
)


# ---------------------------------------------------------------------------
# 1. Status normalization
# ---------------------------------------------------------------------------
def test_finding_type_to_public_state_mapping():
    assert public_state("robust_relation") == "committed"
    assert public_state("falsified_candidate") == "rejected"
    assert public_state("artifact") == "artifact"
    assert public_state("structural_relation") == "artifact"
    assert public_state("promising_candidate") == "provisional"
    assert public_state("untestable_candidate") == "unresolved"
    # A falsified candidate is never exposed as provisional.
    assert FINDING_TYPE_TO_STATE["falsified_candidate"] != "provisional"
    # Unknown finding types are conservative, never silently committed.
    assert public_state("something_new") == "unresolved"
    assert public_state(None) == "unresolved"


def test_derive_finding_record_separates_hypothesis_from_verdict():
    finding = {
        "candidate": {"id": "linear:x_y", "statement": "x and y show a stable positive linear association."},
        "verdict": {"status": "refuted", "score": 0.12},
        "falsifications": [
            {"name": "baseline", "killed": False, "metric": 0.2, "detail": {"score": 0.2, "baseline": 0.0}},
            {"name": "held_out", "killed": True, "metric": 0.05, "detail": {"score": 0.05, "minimum": 0.15}},
            {"name": "cross_seed", "killed": True, "metric": 0.04, "detail": {"median": 0.04, "spread": 0.5}},
        ],
        "final_status": "refuted",
    }
    record = derive_finding_record(finding, "falsified_candidate")
    assert record["verdict"] == "rejected"
    assert record["is_candidate_hypothesis"] is True
    # The affirmative statement is retained as a hypothesis, not a conclusion.
    assert record["hypothesis_text"].startswith("x and y")
    assert record["failed_checks"] == ["held_out", "cross_seed"]
    assert record["passed_checks"] == ["baseline"]
    assert record["baseline_score"] == 0.2
    assert record["held_out_score"] == 0.05
    assert record["cross_seed_summary"]["median"] == 0.04


# ---------------------------------------------------------------------------
# 2. Transform-artifact detection
# ---------------------------------------------------------------------------
def test_detect_log_transform_and_duplicate_and_unit_conversion():
    rng = np.random.default_rng(0)
    mass = np.geomspace(0.01, 1000, 30)
    df = pd.DataFrame({
        "mass_kg": mass,
        "log_mass": np.log10(mass),          # exact log transform -> artifact
        "mass_g": mass * 1000.0,             # unit conversion -> artifact
        "mass_copy": mass,                   # duplicate -> artifact
        "indep": rng.normal(size=mass.size),  # unrelated -> not an artifact
    })
    rel = detect_structural_relations(df)
    kinds = {v["kind"] for v in rel.values()}
    assert "log_transform" in kinds
    # mass_g and mass_copy are exact functions of mass_kg.
    structural_pairs = {tuple(sorted(v["columns"])) for v in rel.values()}
    assert ("log_mass", "mass_kg") in structural_pairs
    assert ("mass_g", "mass_kg") in structural_pairs
    assert ("mass_copy", "mass_kg") in structural_pairs
    # The independent column must not be flagged structural against mass.
    assert ("indep", "mass_kg") not in structural_pairs


def test_tight_real_law_is_not_flagged_as_artifact():
    # A near-perfect physical law (Kepler-like log-log) with real residual scatter
    # must survive as science, not be misclassified as a structural artifact.
    rng = np.random.default_rng(7)
    radius = np.geomspace(0.4, 40, 40)
    log_radius = np.log10(radius)
    log_period = 1.5 * log_radius + rng.normal(scale=0.01, size=radius.size)
    df = pd.DataFrame({"log_radius": log_radius, "log_period": log_period})
    rel = detect_structural_relations(df)
    structural_pairs = {tuple(sorted(v["columns"])) for v in rel.values()}
    assert ("log_period", "log_radius") not in structural_pairs


# ---------------------------------------------------------------------------
# 3. Influence warning
# ---------------------------------------------------------------------------
def test_influence_warning_on_high_leverage_point():
    # 19 clustered points plus one extreme observation that dominates the fit.
    x = np.concatenate([np.linspace(1.0, 5.0, 19), [500.0]])
    y = 2.0 * x + 1.0
    df = pd.DataFrame({"x": x, "y": y})
    warning = linear_influence_warning(df, "x", "y")
    assert warning is not None
    assert warning["warning"] == "high_leverage_dominance"
    assert warning["max_leverage"] > 0.5


def test_no_influence_warning_on_balanced_data():
    x = np.linspace(1.0, 20.0, 40)
    y = 2.0 * x + 1.0 + np.sin(x)
    df = pd.DataFrame({"x": x, "y": y})
    assert linear_influence_warning(df, "x", "y") is None


# ---------------------------------------------------------------------------
# 4 + 5. End-to-end: the exact inconsistent response must not recur
# ---------------------------------------------------------------------------
def _write_allometry_like(path: Path) -> None:
    """A power law with log columns, plus an unrelated noise column."""
    masses = [0.01 * (2.4 ** i) for i in range(22)]  # spans ~4 orders of magnitude
    rows = ["mass,rate,log_mass,log_rate,noise"]
    for i, m in enumerate(masses):
        rate = 3.0 * (m ** 0.75) * (1.0 + 0.05 * math.sin(i))  # real scatter in log space
        noise = ((i * 37) % 11) - 5
        rows.append(
            f"{m:.6f},{rate:.6f},{math.log10(m):.6f},{math.log10(rate):.6f},{noise}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_end_to_end_no_falsified_provisional_leak(tmp_path: Path):
    table = tmp_path / "allometry.csv"
    _write_allometry_like(table)
    with ResearchMVP(tmp_path / "o.db", tmp_path / "ws") as svc:
        case = svc.create_case(name="Allometry semantics", goal="")
        svc.add_file(case["id"], table)
        plan = svc.compile_case(case["id"])
        # Log columns must be detected as structural artifacts up front.
        structural = plan["plan"]["structural_relations"]
        assert structural, "expected log-transform artifacts to be detected"
        assert any(s["artifact_kind"] == "log_transform" for s in structural)

        svc.approve_plan(plan["id"], reviewer="tester")
        run = svc.run_case(case["id"], plan_id=plan["id"])
        assert run["status"] == "completed"

        claims = svc.store.case_claims(case["id"])
        counts = svc.store.case_claim_counts(case["id"])

        # THE BUG: a falsified candidate exposed as provisional. Must never happen.
        leaks = [
            c for c in claims
            if c["finding_type"] == "falsified_candidate" and c["verdict"] == "provisional"
        ]
        assert leaks == [], f"falsified candidates leaked as provisional: {len(leaks)}"

        # Every rejected finding is labelled a candidate hypothesis, never a conclusion.
        for c in claims:
            if c["verdict"] == "rejected":
                assert c["display_label"] == "Candidate hypothesis"

        # Counts must report a real generated-candidate total, never zero.
        assert counts["generated_candidates"] > 0
        assert counts["artifact_count"] >= 1
        assert counts["committed_count"] >= 1
        # Provisional must not absorb rejected findings.
        assert counts["rejected_count"] >= 1
        # Public verdict set is exactly the five spec states.
        assert set(c["verdict"] for c in claims) <= {
            "committed", "rejected", "artifact", "provisional", "unresolved"
        }
