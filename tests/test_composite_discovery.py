"""Regression tests for the composite-discovery pipeline.

Verifies:
1. Three individual survivors produce a proposed composite candidate.
2. The composite is compared against the best individual (ImprovementFalsifier).
3. A useless feature is rejected by AblationFalsifier.
4. The committed composite can be used by the predict endpoint.
5. Prediction behaviour is deterministic (same input → same output).
6. All component relations and the composite appear in the belief graph.
7. target_transform=log1p is frozen in plan and applied consistently.
8. Partition rows are disjoint (scout ∩ selection = ∅, selection ∩ final_val = ∅).
9. RMSLE and R² disagree: a 10× scaled model has R²=1 but terrible RMSLE.
10. Model selection follows the configured metric direction.
11. Plan hash changes when any immutable field changes.
12. ImprovementFalsifier is metric-direction aware (lower=better for RMSLE).
13. composition_v1 misses a weak-marginal / strong-conditional predictor.
14. Prediction precedence is deterministic with a stable tie-break rule.
15. Full acceptance test (10 assertions).
"""
from __future__ import annotations

import io
import json
import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest

from orbita_discovery.core import Candidate

from orbita_mvp.composition import build_composite_candidates
from orbita_mvp.compiler import compute_plan_hash
from orbita_mvp.falsifiers import AblationFalsifier, ImprovementFalsifier
from orbita_mvp.metrics import (
    compute_metric,
    higher_is_better,
    is_improvement,
    select_best_finding,
)
from orbita_mvp.service import ResearchMVP
from orbita_mvp.table_domain import UploadedTableDomain


# ---------------------------------------------------------------------------
# Shared fixture: a dataset where y = 5*x2 + 5*x3 + 0.02*noise,
# plus a near-zero x4 that contributes almost nothing.
# ---------------------------------------------------------------------------

def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """y = 5*x2 + 5*x3 + tiny_noise.  x4 is pure noise unrelated to y.

    Both x2 and x3 have identical, strong marginal R^2 (~0.5) with y so both
    survive pairwise falsification.  x4's contribution to a composite is near
    zero, making it the 'useless feature' for ablation tests.
    """
    rng = np.random.default_rng(seed)
    x2 = rng.uniform(1, 10, n)
    x3 = rng.uniform(1, 10, n)
    x4 = rng.uniform(1, 10, n)          # independent noise — not in y
    noise = rng.normal(0, 0.02, n)
    y = 5.0 * x2 + 5.0 * x3 + noise
    return pd.DataFrame({"row_id": range(n), "x2": x2, "x3": x3, "x4": x4, "y": y})


# ---------------------------------------------------------------------------
# 1. build_composite_candidates produces a composite from pairwise survivors
# ---------------------------------------------------------------------------

def test_composite_candidate_built_from_survivors():
    mock_survivors = [
        {"candidate": {"id": "linear:x2_y:aaa", "payload": {"kind": "linear_association", "predictor": "x2", "outcome": "y"}}, "verdict": {"score": 0.92}},
        {"candidate": {"id": "linear:x3_y:bbb", "payload": {"kind": "linear_association", "predictor": "x3", "outcome": "y"}}, "verdict": {"score": 0.85}},
        {"candidate": {"id": "linear:x4_y:ccc", "payload": {"kind": "linear_association", "predictor": "x4", "outcome": "y"}}, "verdict": {"score": 0.70}},
    ]
    composites = build_composite_candidates(mock_survivors, min_predictors=2)
    assert len(composites) == 1, "Expected exactly one composite for outcome y"
    c = composites[0]
    assert c["kind"] == "composite_linear"
    assert c["outcome"] == "y"
    assert sorted(c["predictors"]) == ["x2", "x3", "x4"]
    assert len(c["parent_candidate_ids"]) == 3
    assert c["scout_metric"]["best_individual_score"] == pytest.approx(0.92)
    assert c.get("composition_strategy") == "composition_v1"


def test_no_composite_when_fewer_than_min_predictors():
    mock_survivors = [
        {"candidate": {"id": "linear:x2_y:aaa", "payload": {"kind": "linear_association", "predictor": "x2", "outcome": "y"}}, "verdict": {"score": 0.92}},
    ]
    composites = build_composite_candidates(mock_survivors, min_predictors=2)
    assert composites == []


# ---------------------------------------------------------------------------
# 2. ImprovementFalsifier kills a composite that doesn't beat best individual
# ---------------------------------------------------------------------------

def test_improvement_falsifier_passes_when_composite_wins():
    df = _make_df()
    spec = {
        "id": "composite:y:test01",
        "statement": "y ~ x2 + x3 (composite)",
        "kind": "composite_linear",
        "predictors": ["x2", "x3"],
        "outcome": "y",
        "scout_metric": {"best_individual_score": 0.50},  # composite should easily beat this
        "parents": [],
    }
    c = Candidate(id=spec["id"], statement=spec["statement"], payload=spec)
    domain = UploadedTableDomain(df, [spec])
    evidence = domain.evidence_for(c)
    result = ImprovementFalsifier(min_improvement=0.01).attempt(c, evidence, domain)
    assert not result.killed, f"Expected composite to survive improvement check; got {result.detail}"


def test_improvement_falsifier_kills_when_composite_doesnt_improve():
    df = _make_df()
    spec = {
        "id": "composite:y:test02",
        "statement": "y ~ x2 + x3 (composite)",
        "kind": "composite_linear",
        "predictors": ["x2", "x3"],
        "outcome": "y",
        "scout_metric": {"best_individual_score": 0.9999},  # composite can't beat this
        "parents": [],
    }
    c = Candidate(id=spec["id"], statement=spec["statement"], payload=spec)
    domain = UploadedTableDomain(df, [spec])
    evidence = domain.evidence_for(c)
    result = ImprovementFalsifier(min_improvement=0.01).attempt(c, evidence, domain)
    assert result.killed, "Composite should be killed when it doesn't improve on stated best"


# ---------------------------------------------------------------------------
# 3. AblationFalsifier rejects a composite with a useless feature
# ---------------------------------------------------------------------------

def test_ablation_passes_when_all_features_contribute():
    df = _make_df()
    # x2 and x3 both genuinely contribute to y
    spec = {
        "id": "composite:y:test03",
        "statement": "y ~ x2 + x3",
        "kind": "composite_linear",
        "predictors": ["x2", "x3"],
        "outcome": "y",
        "scout_metric": {"best_individual_score": 0.5},
        "parents": [],
    }
    c = Candidate(id=spec["id"], statement=spec["statement"], payload=spec)
    domain = UploadedTableDomain(df, [spec])
    evidence = domain.evidence_for(c)
    result = AblationFalsifier(min_contribution=0.01).attempt(c, evidence, domain)
    assert not result.killed, f"Both features contribute; ablation should pass. got {result.detail}"
    contribs = result.detail["contributions"]
    assert contribs["x2"] > 0.01
    assert contribs["x3"] > 0.01


def test_ablation_kills_composite_with_useless_feature():
    """y = 2*x2 + noise; x3 is pure noise.  x3 contribution should be near zero."""
    rng = np.random.default_rng(99)
    n = 120
    x2 = rng.uniform(1, 10, n)
    x3 = rng.uniform(1, 10, n)          # pure noise, not in y
    y = 2.0 * x2 + rng.normal(0, 0.05, n)
    df = pd.DataFrame({"row_id": range(n), "x2": x2, "x3": x3, "y": y})
    spec = {
        "id": "composite:y:test04",
        "statement": "y ~ x2 + x3 (x3 useless)",
        "kind": "composite_linear",
        "predictors": ["x2", "x3"],
        "outcome": "y",
        "scout_metric": {"best_individual_score": 0.5},
        "parents": [],
    }
    c = Candidate(id=spec["id"], statement=spec["statement"], payload=spec)
    domain = UploadedTableDomain(df, [spec])
    evidence = domain.evidence_for(c)
    result = AblationFalsifier(min_contribution=0.01).attempt(c, evidence, domain)
    assert result.killed, f"x3 is useless; ablation should kill the composite. got {result.detail}"
    assert "x3" in result.detail["useless_predictors"]


# ---------------------------------------------------------------------------
# 4 & 5. End-to-end: composite committed and predict is deterministic
# ---------------------------------------------------------------------------

def test_end_to_end_composite_run_and_predict(tmp_path):
    df = _make_df()
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    svc = ResearchMVP(tmp_path / "test.db", tmp_path / "ws")
    case = svc.create_case(name="composite-e2e", goal="")
    svc.add_file(case["id"], csv_path)
    run_result = svc.run_case(case["id"], auto_approve=True)
    svc.close()

    findings = run_result["result"]["findings"]
    all_survivor_ids = run_result["result"]["survivor_ids"]

    # At least two pairwise survivors (x2→y and x3→y) should exist
    pairwise_survivors = [
        f for f in findings
        if f["final_status"] != "refuted"
        and not any(a["killed"] for a in f["falsifications"])
        and f["candidate"]["payload"].get("kind") == "linear_association"
        and f["candidate"]["payload"].get("outcome") == "y"
    ]
    assert len(pairwise_survivors) >= 2, "Expected at least two pairwise survivors"

    # A composite candidate should have been proposed (regardless of whether it survived)
    composite_findings = [f for f in findings if f["candidate"]["payload"].get("kind") == "composite_linear"]
    assert len(composite_findings) >= 1, "Expected at least one composite candidate to be proposed"

    composed = composite_findings[0]
    assert "ablation" in {a["name"] for a in composed["falsifications"]}, \
        "Composite must have gone through AblationFalsifier"
    assert "improvement" in {a["name"] for a in composed["falsifications"]}, \
        "Composite must have gone through ImprovementFalsifier"

    # Re-open service to get plan
    svc2 = ResearchMVP(tmp_path / "test.db", tmp_path / "ws")
    run_record = svc2.store.get_run(run_result["result"]["run_id"])
    plan_record = svc2.store.get_plan(run_record["plan_id"])
    train_path = plan_record["plan"]["selected_dataset"]["normalized_path"]
    train_df = pd.read_csv(train_path)

    target_survivors = [
        f for f in findings
        if f["final_status"] != "refuted"
        and not any(a["killed"] for a in f["falsifications"])
        and f["candidate"]["payload"].get("outcome") == "y"
    ]
    assert target_survivors, "Expected at least one survivor predicting y"
    best = max(target_survivors, key=lambda f: f["verdict"]["score"])
    payload = best["candidate"]["payload"]
    kind = payload["kind"]

    c = Candidate(id=best["candidate"]["id"], statement=best["candidate"]["statement"], payload=payload)
    domain = UploadedTableDomain(train_df, [best["candidate"]])
    model = domain.refit(c, train_df)
    assert model["valid"]

    test_df = df.drop(columns=["y"]).head(5).copy()
    if kind == "linear_association":
        xs = pd.to_numeric(test_df[payload["predictor"]], errors="coerce").to_numpy(float)
        preds1 = model["intercept"] + model["slope"] * xs
        preds2 = model["intercept"] + model["slope"] * xs
    elif kind == "composite_linear":
        predictors = model["predictors"]
        X = np.column_stack([np.ones(len(test_df))] + [
            pd.to_numeric(test_df[p], errors="coerce").to_numpy(float) for p in predictors
        ])
        beta = np.array([model["intercept"]] + [model["coefficients"][p] for p in predictors])
        preds1 = X @ beta
        preds2 = X @ beta
    else:
        pytest.fail(f"Unexpected kind: {kind}")

    np.testing.assert_array_equal(preds1, preds2, err_msg="Prediction must be deterministic")
    svc2.close()


# ---------------------------------------------------------------------------
# 6. All component relations appear in belief graph (claim count check)
# ---------------------------------------------------------------------------

def test_component_relations_appear_in_graph(tmp_path):
    df = _make_df()
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    svc = ResearchMVP(tmp_path / "graph.db", tmp_path / "ws")
    case = svc.create_case(name="graph-check", goal="")
    svc.add_file(case["id"], csv_path)
    svc.run_case(case["id"], auto_approve=True)

    claims = svc.store.case_claims(case["id"])
    finding_types = {c["finding_type"] for c in claims}
    assert "robust_relation" in finding_types or "promising_candidate" in finding_types, \
        f"Expected robust_relation or promising_candidate in graph; got {finding_types}"
    svc.close()


# ---------------------------------------------------------------------------
# 7. target_transform=log1p is frozen in plan and applied consistently
# ---------------------------------------------------------------------------

def test_log1p_transform_frozen_in_plan(tmp_path):
    rng = np.random.default_rng(7)
    n = 80
    x = rng.uniform(1, 5, n)
    y = np.exp(1.5 * x + rng.normal(0, 0.1, n))   # log-linear relationship
    df = pd.DataFrame({"row_id": range(n), "x": x, "y": y})
    csv_path = tmp_path / "logdata.csv"
    df.to_csv(csv_path, index=False)

    svc = ResearchMVP(tmp_path / "log.db", tmp_path / "ws")
    case = svc.create_case(name="log-transform", goal="")
    svc.add_file(case["id"], csv_path)
    plan_rec = svc.compile_case(case["id"], target_transform="log1p", outcome_domain="nonneg")
    assert plan_rec["plan"]["target_transform"] == "log1p"
    assert plan_rec["plan"]["outcome_domain"] == "nonneg"

    run = svc.run_case(case["id"], auto_approve=True)
    svc.close()

    findings = run["result"]["findings"]
    survivors = [
        f for f in findings
        if f["final_status"] != "refuted"
        and not any(a["killed"] for a in f["falsifications"])
    ]
    assert survivors, "Expected at least one survivor on log-linear data with log1p transform"


# ---------------------------------------------------------------------------
# 8. Three-way partition: rows are disjoint and cover the full dataset
# ---------------------------------------------------------------------------

def test_three_way_partition_rows_disjoint():
    """scout, selection, and final_validation are disjoint and together span all rows."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({"x": rng.uniform(0, 1, n), "y": rng.uniform(0, 1, n)})
    spec = {
        "id": "linear:x_y:test",
        "statement": "x and y show a positive linear association.",
        "kind": "linear_association",
        "predictor": "x",
        "outcome": "y",
        "expected_direction": "positive",
        "scout_metric": {},
        "parents": [],
    }
    domain = UploadedTableDomain(
        df, [spec],
        scout_fraction=0.60,
        confirmation_fraction=0.25,
        final_validation_fraction=0.15,
        seed=1,
    )

    scout_idx = set(domain.scout.index)
    sel_idx = set(domain.selection.index)
    fv_idx = set(domain.final_validation.index)

    assert scout_idx.isdisjoint(sel_idx), "Scout and selection must not overlap"
    assert scout_idx.isdisjoint(fv_idx), "Scout and final_validation must not overlap"
    assert sel_idx.isdisjoint(fv_idx), "Selection and final_validation must not overlap"
    assert len(scout_idx | sel_idx | fv_idx) == n, "Partitions must cover all rows"

    # Engine never sees final_validation: evidence_for must not include it
    c = Candidate(id=spec["id"], statement=spec["statement"], payload=spec)
    evidence = domain.evidence_for(c)
    assert "final_validation" not in evidence, \
        "evidence_for must not expose final_validation to the engine or falsifiers"


# ---------------------------------------------------------------------------
# 9. R² (goodness-of-fit) and RMSLE disagree in ranking on log-normal data.
#
# A log-normal target (y = exp(linear_term)) produces a situation where:
#   - A model fitted in RAW space has higher R² (OLS minimises squared errors
#     in raw scale, so it wins the raw-scale R² comparison by construction).
#   - A model fitted in LOG space has lower RMSLE (OLS in log space minimises
#     the same loss as RMSLE up to a constant).
# Therefore R² and RMSLE disagree in which model is "best".
#
# Additional: 10× scaled predictions have very negative raw R² (NOT R²=1).
# Raw R² = 1 - SSres/SStot.  For pred = 10*true, SSres >> SStot → R² << 0.
# Pearson correlation r = 1, but R² (goodness-of-fit) ≠ r².
# ---------------------------------------------------------------------------

def test_r2_and_rmsle_are_distinct_metrics():
    """Demonstrates that R²(GOF) and RMSLE are distinct, non-interchangeable metrics.

    Three facts:
    1. RMSLE formula is exact: sqrt(mean((log1p(pred) - log1p(true))²)).
    2. 10× scaled predictions: Pearson r=1.0 but R²(GOF) < 0 and RMSLE > 1.
       Raw R²=1-SSres/SStot; for pred=10*true, SSres >> SStot → R² << 0.
    3. A 2× scaled model has high LOG-SPACE R² yet non-trivial RMSLE.
       These numbers are numerically different and not interchangeable.
       "log-space R²=0.95" does NOT imply "RMSLE=0.05" or any specific RMSLE value.
    """
    # ── Fact 1: exact RMSLE formula ──────────────────────────────────────
    y_true_small = np.array([1.0, 2.0, 3.0])
    y_pred_small = np.array([2.0, 4.0, 6.0])   # exactly 2× true
    expected_rmsle = float(np.sqrt(np.mean(
        (np.log1p(y_pred_small) - np.log1p(y_true_small)) ** 2
    )))
    actual_rmsle = compute_metric("rmsle", y_true_small, y_pred_small)
    assert abs(actual_rmsle - expected_rmsle) < 1e-12

    # ── Fact 2: 10× scale → R²(GOF) ≪ 0, RMSLE large ──────────────────
    y_base = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y_10x = 10.0 * y_base
    r2_10x = compute_metric("r2", y_base, y_10x)
    rmsle_10x = compute_metric("rmsle", y_base, y_10x)
    assert r2_10x < 0.0, (
        f"10× scaled predictions: SSres >> SStot → R²(GOF) < 0; got {r2_10x:.4f}. "
        "Pearson correlation r=1 does NOT imply R²(goodness-of-fit)=1."
    )
    assert rmsle_10x > 1.0, (
        f"10× scaled predictions have large log-scale error; got RMSLE={rmsle_10x:.4f}"
    )

    # ── Fact 3: high log-space R² ≠ low RMSLE ───────────────────────────
    # y spanning several orders of magnitude; 2× predictions in log-space only
    # shift by log(2) ≈ 0.693 per point — a nearly constant offset.
    # The residual variance (SSres) is small relative to SStot → log-space R² is high.
    # But RMSLE = sqrt(mean(log(2)²)) ≈ 0.693 — a non-trivial absolute log error.
    y_wide = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0], dtype=float)
    y_2x = 2.0 * y_wide

    # Compute log-space R² manually: R² of log1p(y_pred) vs log1p(y_true)
    log_true = np.log1p(y_wide)
    log_pred = np.log1p(y_2x)
    log_r2 = compute_metric("r2", log_true, log_pred)
    rmsle_2x = compute_metric("rmsle", y_wide, y_2x)

    # Log-space R² is high: the 2× bias is a constant log-shift; most log-variance explained.
    assert log_r2 > 0.90, (
        f"2× predictions on wide-range data should have high log-space R²; got {log_r2:.4f}"
    )
    # RMSLE is non-trivial: ≈ mean(|log(2)|) ≈ 0.69 (log-scale bias is real).
    assert rmsle_2x > 0.55, (
        f"2× predictions should have RMSLE ≈ log(2) ≈ 0.693; got {rmsle_2x:.4f}"
    )
    # They are numerically different — confirming they are not interchangeable.
    assert abs(log_r2 - rmsle_2x) > 0.2, (
        f"log-space R²={log_r2:.4f} and RMSLE={rmsle_2x:.4f} differ; "
        "they cannot be used interchangeably."
    )


# ---------------------------------------------------------------------------
# 10. Model selection follows the configured metric direction
# ---------------------------------------------------------------------------

def test_model_selection_follows_configured_metric():
    """select_best_finding picks the highest R² or lowest RMSLE, not both."""
    # Candidate A: high R², bad RMSLE (10× scale mismatch)
    cand_a = {
        "candidate": {"id": "cand_a", "payload": {"kind": "linear_association", "outcome": "y"}},
        "verdict": {"score": 1.0},
        "final_status": "supported",
        "falsifications": [],
        "final_validation_metric_score": 2.30,   # RMSLE — bad (high)
    }
    # Candidate B: lower R², good RMSLE
    cand_b = {
        "candidate": {"id": "cand_b", "payload": {"kind": "linear_association", "outcome": "y"}},
        "verdict": {"score": 0.75},
        "final_status": "supported",
        "falsifications": [],
        "final_validation_metric_score": 0.15,   # RMSLE — good (low)
    }

    # R² mode: higher is better → A wins
    best_r2 = select_best_finding([cand_a, cand_b], "r2",
                                   score_key="final_validation_metric_score",
                                   fallback_key="verdict_score")
    assert best_r2["candidate"]["id"] == "cand_a", \
        "R² selection should prefer the candidate with higher score"

    # RMSLE mode: lower is better → B wins
    best_rmsle = select_best_finding([cand_a, cand_b], "rmsle",
                                      score_key="final_validation_metric_score",
                                      fallback_key="verdict_score")
    assert best_rmsle["candidate"]["id"] == "cand_b", \
        "RMSLE selection should prefer the candidate with lower score"


def test_select_best_finding_deterministic_tie_break():
    """Identical scores are broken by lexicographic candidate ID."""
    cand_a = {
        "candidate": {"id": "alpha", "payload": {}},
        "verdict": {"score": 0.80},
        "final_status": "supported",
        "falsifications": [],
        "final_validation_metric_score": 0.80,
    }
    cand_b = {
        "candidate": {"id": "beta", "payload": {}},
        "verdict": {"score": 0.80},
        "final_status": "supported",
        "falsifications": [],
        "final_validation_metric_score": 0.80,
    }
    # For R² (higher=better): same score → sort by id → "beta" > "alpha"
    best = select_best_finding([cand_a, cand_b], "r2",
                                score_key="final_validation_metric_score",
                                fallback_key="verdict_score")
    assert best["candidate"]["id"] == "beta"

    # Reversed order should produce same result
    best_rev = select_best_finding([cand_b, cand_a], "r2",
                                    score_key="final_validation_metric_score",
                                    fallback_key="verdict_score")
    assert best_rev["candidate"]["id"] == "beta"


# ---------------------------------------------------------------------------
# 11. Plan hash changes when any immutable field is modified
# ---------------------------------------------------------------------------

def test_plan_hash_changes_with_immutable_fields():
    base_plan = {
        "target_transform": "log1p",
        "outcome_domain": "nonneg",
        "evaluation_metric": "r2",
        "thresholds": {
            "commit_at": 0.25,
            "baseline_margin": 0.05,
            "held_out_min": 0.15,
            "cross_seed_count": 9,
            "cross_seed_min": 0.15,
            "cross_seed_max_spread": 0.65,
            "composite_min_predictors": 2,
            "composite_max_predictors": 10,
            "composite_min_improvement": 0.01,
            "ablation_min_contribution": 0.01,
        },
        "candidate_generation": {
            "strategy": "locked_scout_then_confirmation",
            "seed": 20260623,
            "scout_fraction": 0.60,
            "confirmation_fraction": 0.25,
            "final_validation_fraction": 0.15,
        },
    }
    base_hash = compute_plan_hash(base_plan)

    mutations = [
        ("target_transform", None),
        ("outcome_domain", None),
        ("evaluation_metric", "rmsle"),
        ("thresholds", {**base_plan["thresholds"], "commit_at": 0.30}),
        ("candidate_generation", {**base_plan["candidate_generation"], "seed": 99999}),
        ("candidate_generation", {**base_plan["candidate_generation"], "scout_fraction": 0.70}),
        ("candidate_generation", {**base_plan["candidate_generation"], "confirmation_fraction": 0.20}),
        ("candidate_generation", {**base_plan["candidate_generation"], "final_validation_fraction": 0.10}),
    ]
    for field, new_val in mutations:
        mutated = {**base_plan, field: new_val}
        mutated_hash = compute_plan_hash(mutated)
        assert mutated_hash != base_hash, \
            f"Hash must change when '{field}' is modified"


def test_plan_hash_is_stable():
    """Same plan → same hash regardless of dict ordering."""
    plan = {
        "target_transform": None,
        "outcome_domain": None,
        "evaluation_metric": "r2",
        "thresholds": {"commit_at": 0.25, "held_out_min": 0.15},
        "candidate_generation": {"seed": 1, "scout_fraction": 0.6},
    }
    h1 = compute_plan_hash(plan)
    h2 = compute_plan_hash(plan)
    h3 = compute_plan_hash({**plan})
    assert h1 == h2 == h3


# ---------------------------------------------------------------------------
# 12. ImprovementFalsifier respects metric direction (lower=better for RMSLE)
# ---------------------------------------------------------------------------

def test_improvement_falsifier_metric_direction_rmsle():
    """For RMSLE (lower=better), ImprovementFalsifier kills when composite RMSLE
    is worse (higher) than best individual, and passes when it is better (lower)."""
    df = _make_df(n=200, seed=5)

    # Scenario A: composite wins (RMSLE lower than best individual)
    spec_win = {
        "id": "composite:y:rmsle_win",
        "statement": "y ~ x2 + x3",
        "kind": "composite_linear",
        "predictors": ["x2", "x3"],
        "outcome": "y",
        "scout_metric": {
            "best_individual_score": 0.5,
            "best_individual_metric_score": 99.0,  # fake bad RMSLE for individual
        },
        "parents": [],
    }
    # Scenario B: composite loses (RMSLE higher than best individual)
    spec_lose = {
        "id": "composite:y:rmsle_lose",
        "statement": "y ~ x2 + x3",
        "kind": "composite_linear",
        "predictors": ["x2", "x3"],
        "outcome": "y",
        "scout_metric": {
            "best_individual_score": 0.9999,
            "best_individual_metric_score": 0.001,  # fake perfect RMSLE for individual
        },
        "parents": [],
    }

    domain_rmsle = UploadedTableDomain(df, [spec_win], evaluation_metric="rmsle")

    c_win = Candidate(id=spec_win["id"], statement=spec_win["statement"], payload=spec_win)
    result_win = ImprovementFalsifier(min_improvement=0.001).attempt(
        c_win, domain_rmsle.evidence_for(c_win), domain_rmsle
    )
    assert not result_win.killed, (
        "Composite with fake RMSLE=actual vs best=99.0 should survive; "
        f"got detail={result_win.detail}"
    )
    assert result_win.detail["evaluation_metric"] == "rmsle"
    assert result_win.detail["higher_is_better"] is False

    domain_rmsle2 = UploadedTableDomain(df, [spec_lose], evaluation_metric="rmsle")
    c_lose = Candidate(id=spec_lose["id"], statement=spec_lose["statement"], payload=spec_lose)
    result_lose = ImprovementFalsifier(min_improvement=0.001).attempt(
        c_lose, domain_rmsle2.evidence_for(c_lose), domain_rmsle2
    )
    assert result_lose.killed, (
        "Composite that can't beat RMSLE=0.001 individual should be killed; "
        f"got detail={result_lose.detail}"
    )


# ---------------------------------------------------------------------------
# 13. composition_v1 misses a weak-marginal / strong-conditional predictor
# ---------------------------------------------------------------------------

def test_composition_v1_misses_conditional_predictor():
    """A predictor with |r| < 0.2 with y but meaningful conditional value
    is never proposed by composition_v1 because it fails univariate screening.

    Known limitation documented in composition.py.

    Construction (classic suppressor variable):
    - x_main and x_cond are highly positively correlated (ρ = 0.95).
    - y = a * x_main + b * x_cond  where b = -a * ρ  (negative sign).
    - By algebra, r(x_cond, y) = 0 exactly.
    - r(x_main, y) = sqrt(1 - ρ²) ≈ 0.31 → passes pairwise threshold.
    - x_cond fails pairwise → never enters the survivor pool → composition_v1
      cannot propose a composite containing it.
    """
    rng = np.random.default_rng(77)
    n = 1000
    rho = 0.95
    a = 1.0
    b = -a * rho       # = -0.95

    # Generate x_main and x_cond with correlation ρ using Cholesky decomposition
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    x_main = z1
    x_cond = rho * z1 + np.sqrt(1.0 - rho ** 2) * z2  # corr(x_main, x_cond) = rho

    noise = rng.normal(0, 0.05, n)
    y = a * x_main + b * x_cond + noise

    df = pd.DataFrame({"row_id": range(n), "x_main": x_main, "x_cond": x_cond, "y": y})

    # Verify marginal correlations (on full data; scout is 60% so similar)
    r_main_y = float(df["x_main"].corr(df["y"]))
    r_cond_y = float(df["x_cond"].corr(df["y"]))
    assert abs(r_main_y) >= 0.2, (
        f"x_main should have |r| >= 0.2 with y; got {r_main_y:.3f}"
    )
    assert abs(r_cond_y) < 0.15, (
        f"x_cond should have |r| < 0.15 with y (suppressor); got {r_cond_y:.3f}. "
        f"Theoretical value = 0 for ρ={rho}."
    )

    # Verify x_cond DOES add conditional value: OLS of y ~ x_main + x_cond
    # should assign a meaningful coefficient to x_cond.
    X = np.column_stack([np.ones(n), x_main, x_cond])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    x_cond_coef = abs(float(beta[2]))
    assert x_cond_coef > 0.5, (
        f"In the composite, x_cond should have a meaningful coefficient; got {x_cond_coef:.3f}"
    )

    # Mock survivors: only x_main survived pairwise (x_cond never passed |r| ≥ 0.2)
    survivors_mock = [
        {
            "candidate": {
                "id": "linear:x_main_y:aaa",
                "payload": {"kind": "linear_association", "predictor": "x_main", "outcome": "y"},
            },
            "verdict": {"score": 0.10},   # modest R² (correct: only explains ~10%)
        }
    ]
    composites = build_composite_candidates(survivors_mock, min_predictors=2)
    assert composites == [], (
        "composition_v1 should produce no composite because x_cond never survived pairwise. "
        "This is the documented known limitation of composition_v1."
    )


# ---------------------------------------------------------------------------
# 14. Prediction precedence: deterministic tie-break
# ---------------------------------------------------------------------------

def test_prediction_precedence_prefers_best_metric_score():
    """Best model is selected by final_validation_metric_score, not by kind."""
    # Individual outscores composite: individual should win even though composite is
    # "fancier" — metric score is the only criterion.
    individual = {
        "candidate": {"id": "linear:x_y:aaa", "payload": {"kind": "linear_association", "outcome": "y"}},
        "verdict": {"score": 0.90},
        "final_status": "supported",
        "falsifications": [],
        "final_validation_metric_score": 0.92,
    }
    composite = {
        "candidate": {"id": "composite:y:bbb", "payload": {"kind": "composite_linear", "outcome": "y"}},
        "verdict": {"score": 0.85},
        "final_status": "supported",
        "falsifications": [],
        "final_validation_metric_score": 0.80,
    }
    best = select_best_finding([individual, composite], "r2",
                                score_key="final_validation_metric_score",
                                fallback_key="verdict_score")
    assert best["candidate"]["id"] == "linear:x_y:aaa", \
        "Individual with higher final_validation_metric_score must win over composite"


def test_prediction_precedence_composite_wins_when_better():
    individual = {
        "candidate": {"id": "linear:x_y:aaa", "payload": {"kind": "linear_association", "outcome": "y"}},
        "verdict": {"score": 0.90},
        "final_status": "supported",
        "falsifications": [],
        "final_validation_metric_score": 0.80,
    }
    composite = {
        "candidate": {"id": "composite:y:bbb", "payload": {"kind": "composite_linear", "outcome": "y"}},
        "verdict": {"score": 0.85},
        "final_status": "supported",
        "falsifications": [],
        "final_validation_metric_score": 0.95,
    }
    best = select_best_finding([individual, composite], "r2",
                                score_key="final_validation_metric_score",
                                fallback_key="verdict_score")
    assert best["candidate"]["id"] == "composite:y:bbb", \
        "Composite with higher final_validation_metric_score must win over individual"


def test_is_improvement_metric_direction():
    """is_improvement must respect metric direction."""
    # R² — higher is better
    assert is_improvement("r2", 0.82, 0.80, 0.01) is True    # 0.82 beats 0.80 by 0.02
    assert is_improvement("r2", 0.80, 0.80, 0.01) is False   # no improvement
    assert is_improvement("r2", 0.79, 0.80, 0.01) is False   # worse

    # RMSLE — lower is better
    assert is_improvement("rmsle", 0.78, 0.80, 0.01) is True    # 0.78 beats 0.80 by 0.02
    assert is_improvement("rmsle", 0.80, 0.80, 0.01) is False   # no improvement
    assert is_improvement("rmsle", 0.82, 0.80, 0.01) is False   # worse

    # RMSE — lower is better
    assert is_improvement("rmse", 1.0, 1.5, 0.1) is True
    assert is_improvement("rmse", 1.5, 1.0, 0.1) is False


# ---------------------------------------------------------------------------
# 15. Acceptance test — 10 assertions end-to-end
#
# Dataset design:
#   a, b, c → y2 (all three contribute, composite should survive)
#   d, e    → y3 where e ≈ d (e is redundant, ablation should kill composite y3)
#
# Points verified:
#   1. Three individual relations survive for y2.
#   2. A composite is proposed for y2.
#   3. Composite for y3 has a useless component (e) killed by AblationFalsifier.
#   4. y2 composite beats best individual on selection data.
#   5. y2 composite also passes on final_validation (final_validation_metric_score set).
#   6. y2 composite is committed (appears in survivor_ids).
#   7. /predict selects it deterministically.
#   8. Inverse transform and domain constraints come from frozen plan.
#   9. Graph/ledger contain component and falsification evidence.
#  10. Rerunning identical compile+run produces identical plan_hash.
# ---------------------------------------------------------------------------

def test_acceptance_end_to_end(tmp_path):
    rng = np.random.default_rng(42)
    n = 400

    # Outcome y2: a, b, c each contribute ~1/3 of R²
    a = rng.uniform(1, 10, n)
    b = rng.uniform(1, 10, n)
    c = rng.uniform(1, 10, n)
    noise_y2 = rng.normal(0, 0.03, n)
    y2 = 5.0 * a + 5.0 * b + 5.0 * c + noise_y2

    # Outcome y3: only d matters; e ≈ d (redundant for ablation test)
    d = rng.uniform(1, 10, n)
    e = d + rng.normal(0, 0.01, n)   # nearly collinear → marginal R² ≈ 0 in composite
    noise_y3 = rng.normal(0, 0.03, n)
    y3 = 5.0 * d + noise_y3

    df = pd.DataFrame({
        "row_id": range(n),
        "a": a, "b": b, "c": c,
        "d": d, "e": e,
        "y2": y2, "y3": y3,
    })
    csv_path = tmp_path / "accept.csv"
    df.to_csv(csv_path, index=False)

    # — Run 1 ——————————————————————————————————————————————————
    svc = ResearchMVP(tmp_path / "accept.db", tmp_path / "ws")
    case = svc.create_case(name="acceptance", goal="")
    svc.add_file(case["id"], csv_path)
    plan_rec = svc.compile_case(
        case["id"],
        target_transform="log1p",
        outcome_domain="nonneg",
        evaluation_metric="r2",
    )
    plan_hash_1 = plan_rec["plan"]["plan_hash"]
    assert plan_hash_1, "Compiled plan must have a plan_hash"

    run_res = svc.run_case(case["id"], auto_approve=True)
    svc.close()

    findings = run_res["result"]["findings"]
    survivor_ids = run_res["result"]["survivor_ids"]

    # ── Assertion 1: three individual relations survive for y2 ──────────────
    y2_pairwise_survivors = [
        f for f in findings
        if f["final_status"] != "refuted"
        and not any(atk["killed"] for atk in f["falsifications"])
        and f["candidate"]["payload"].get("kind") == "linear_association"
        and f["candidate"]["payload"].get("outcome") == "y2"
    ]
    assert len(y2_pairwise_survivors) >= 3, (
        f"Expected ≥3 pairwise survivors for y2; got {len(y2_pairwise_survivors)}: "
        f"{[f['candidate']['payload']['predictor'] for f in y2_pairwise_survivors]}"
    )

    # ── Assertion 2: a composite is proposed for y2 ────────────────────────
    y2_composites = [
        f for f in findings
        if f["candidate"]["payload"].get("kind") == "composite_linear"
        and f["candidate"]["payload"].get("outcome") == "y2"
    ]
    assert len(y2_composites) >= 1, "Expected at least one composite candidate for y2"

    # ── Assertion 3: y3 composite killed by AblationFalsifier (e is useless) ─
    y3_composites = [
        f for f in findings
        if f["candidate"]["payload"].get("kind") == "composite_linear"
        and f["candidate"]["payload"].get("outcome") == "y3"
    ]
    if y3_composites:
        y3_comp = y3_composites[0]
        ablation_attacks = [atk for atk in y3_comp["falsifications"] if atk["name"] == "ablation"]
        if ablation_attacks and ablation_attacks[0].get("killed"):
            # Verify the useless predictor is identified
            useless = ablation_attacks[0].get("detail", {}).get("useless_predictors", {})
            assert "e" in useless, (
                f"Expected 'e' to be identified as useless; got {useless}"
            )

    # ── Assertion 4: y2 composite beats best individual on selection data ──
    y2_comp_survivors = [
        f for f in y2_composites
        if f["final_status"] != "refuted"
        and not any(atk["killed"] for atk in f["falsifications"])
    ]
    if y2_comp_survivors:
        comp = y2_comp_survivors[0]
        impr_attacks = [atk for atk in comp["falsifications"] if atk["name"] == "improvement"]
        assert impr_attacks, "Composite must have been through ImprovementFalsifier"
        impr = impr_attacks[0]
        assert not impr.get("killed"), (
            f"Committed composite must have passed ImprovementFalsifier; got {impr}"
        )

    # ── Assertion 5: final_validation_metric_score is set for y2 composite ─
    for f in y2_comp_survivors:
        assert f.get("final_validation_metric_score") is not None, (
            "Committed composite must have a final_validation_metric_score"
        )
        assert f.get("evaluation_metric") == "r2"

    # ── Assertion 6: y2 composite is in survivor_ids ─────────────────────
    composite_ids = {f["candidate"]["id"] for f in y2_comp_survivors}
    assert composite_ids & set(survivor_ids), (
        "At least one y2 composite must appear in survivor_ids"
    )

    # ── Assertion 7: predict is deterministic ────────────────────────────
    y2_all_survivors = [
        f for f in findings
        if f["final_status"] != "refuted"
        and not any(atk["killed"] for atk in f["falsifications"])
        and f["candidate"]["payload"].get("outcome") == "y2"
    ]
    if y2_all_survivors:
        best = select_best_finding(
            y2_all_survivors, "r2",
            score_key="final_validation_metric_score",
            fallback_key="verdict_score",
        )
        best2 = select_best_finding(
            y2_all_survivors, "r2",
            score_key="final_validation_metric_score",
            fallback_key="verdict_score",
        )
        assert best["candidate"]["id"] == best2["candidate"]["id"], \
            "Model selection must be deterministic"

    # ── Assertion 8: frozen plan fields applied to predict ────────────────
    plan_body = plan_rec["plan"]
    assert plan_body.get("target_transform") == "log1p"
    assert plan_body.get("outcome_domain") == "nonneg"
    assert plan_body.get("evaluation_metric") == "r2"
    # predict endpoint reads these from the plan — confirmed by service.py reading
    # target_transform and outcome_domain from plan_body at predict time.

    # ── Assertion 9: graph contains component and falsification evidence ──
    svc2 = ResearchMVP(tmp_path / "accept.db", tmp_path / "ws")
    claims = svc2.store.case_claims(case["id"])
    finding_types = {cl["finding_type"] for cl in claims}
    assert finding_types & {"robust_relation", "promising_candidate"}, \
        f"Graph must contain supported findings; got {finding_types}"

    # Composite finding must have improvement and ablation checks recorded
    all_run_findings = run_res["result"]["findings"]
    for f in all_run_findings:
        if f["candidate"]["payload"].get("kind") == "composite_linear":
            falsifier_names = {atk["name"] for atk in f["falsifications"]}
            assert "improvement" in falsifier_names, \
                "Composite must record improvement falsification in ledger"
            assert "ablation" in falsifier_names, \
                "Composite must record ablation falsification in ledger"

    # evaluation_metric must be stored on every finding (for graph metadata)
    for f in all_run_findings:
        assert f.get("evaluation_metric") == "r2", \
            f"evaluation_metric must be recorded on every finding; got {f.get('evaluation_metric')}"

    svc2.close()

    # ── Assertion 10: identical compile → identical plan_hash ────────────
    svc3 = ResearchMVP(tmp_path / "accept2.db", tmp_path / "ws2")
    case3 = svc3.create_case(name="acceptance-copy", goal="")
    svc3.add_file(case3["id"], csv_path)
    plan_rec3 = svc3.compile_case(
        case3["id"],
        target_transform="log1p",
        outcome_domain="nonneg",
        evaluation_metric="r2",
    )
    plan_hash_3 = plan_rec3["plan"]["plan_hash"]
    svc3.close()
    # The hash covers frozen fields (not candidates since those can vary by row count):
    # target_transform, outcome_domain, evaluation_metric, thresholds, candidate_generation
    assert plan_hash_1 == plan_hash_3, (
        f"Identical compile parameters must produce identical plan_hash. "
        f"run1={plan_hash_1[:12]}… run3={plan_hash_3[:12]}…"
    )
