"""Tests for fix/metric-aware-ablation-v0.2.1.

Covers:
 1. AblationFalsifier uses selection partition, not scout
 2. AblationFalsifier metric-aware: R² direction (higher-is-better)
 3. AblationFalsifier metric-aware: RMSLE direction (lower-is-better)
 4. AblationFalsifier detail records evaluation_metric, full_score, partition
 5. ablation_contributions in refit() is R²-diagnostic-only (renamed key)
 6. plan_hash changes when ablation_metric is modified
 7. ablation_metric field is propagated to compiled plan
 8. ablation_min_absolute_improvement appears in plan thresholds
 9. composition_v1_1_backward_elimination removes worst failing predictor
10. backward elimination produces reduced candidate with elimination_ledger
11. model artifact is serialized and verifiable
12. model artifact fails on hash tamper
13. staging acceptance: 3-predictor dataset, one weak predictor eliminated, reduced composite survives
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from orbita_discovery.core import Candidate

from orbita_mvp.compiler import compute_plan_hash
from orbita_mvp.composition import build_backward_eliminated_composites, build_composite_candidates
from orbita_mvp.falsifiers import AblationFalsifier
from orbita_mvp.model_artifact import (
    load_model_artifact,
    save_model_artifact,
    serialize_deployment_artifact,
)
from orbita_mvp.table_domain import UploadedTableDomain


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_strong_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """y = 4*x2 + 4*x3 + tiny noise; x4 is pure noise unrelated to y."""
    rng = np.random.default_rng(seed)
    x2 = rng.uniform(1, 10, n)
    x3 = rng.uniform(1, 10, n)
    x4 = rng.uniform(1, 10, n)
    y = 4.0 * x2 + 4.0 * x3 + rng.normal(0, 0.05, n)
    return pd.DataFrame({"row_id": range(n), "x2": x2, "x3": x3, "x4": x4, "y": y})


def _composite_spec(predictors, outcome="y", score=0.5, metric_score=None):
    pid = "_".join(sorted(predictors))
    spec = {
        "id": f"composite:{outcome}:{pid}",
        "statement": f"{outcome} ~ {' + '.join(predictors)}",
        "kind": "composite_linear",
        "predictors": sorted(predictors),
        "outcome": outcome,
        "scout_metric": {"best_individual_score": score},
        "parents": [],
    }
    if metric_score is not None:
        spec["scout_metric"]["best_individual_metric_score"] = metric_score
    return spec


# ---------------------------------------------------------------------------
# Test 1: AblationFalsifier uses selection partition, not scout
# ---------------------------------------------------------------------------

def test_ablation_uses_selection_not_scout():
    """Evidence passed to AblationFalsifier must be evidence['confirmation'] (selection)."""
    df = _make_strong_df(n=300)
    spec = _composite_spec(["x2", "x3"])
    c = Candidate(id=spec["id"], statement=spec["statement"], payload=spec)
    domain = UploadedTableDomain(df, [spec])
    evidence = domain.evidence_for(c)

    # Confirm evidence['confirmation'] IS the selection partition
    assert "confirmation" in evidence, "evidence_for must include 'confirmation' key"
    assert evidence["confirmation"] is domain.selection, \
        "evidence['confirmation'] must be the selection partition"

    result = AblationFalsifier(min_contribution=0.01).attempt(c, evidence, domain)
    assert result.detail.get("partition") == "selection", \
        f"AblationFalsifier must record partition='selection'; got {result.detail.get('partition')!r}"


# ---------------------------------------------------------------------------
# Test 2: AblationFalsifier metric-aware, R² direction
# ---------------------------------------------------------------------------

def test_ablation_r2_direction_passes_strong_predictors():
    """Both x2 and x3 strongly predict y; both contributions must be > 0.01 under R²."""
    df = _make_strong_df()
    spec = _composite_spec(["x2", "x3"])
    c = Candidate(id=spec["id"], statement=spec["statement"], payload=spec)
    domain = UploadedTableDomain(df, [spec], evaluation_metric="r2")
    evidence = domain.evidence_for(c)
    result = AblationFalsifier(min_contribution=0.01).attempt(c, evidence, domain)

    assert not result.killed, f"Both predictors contribute; should pass. detail={result.detail}"
    contribs = result.detail["contributions"]
    assert contribs["x2"] > 0.01, f"x2 R² contribution {contribs['x2']:.4f} should be > 0.01"
    assert contribs["x3"] > 0.01, f"x3 R² contribution {contribs['x3']:.4f} should be > 0.01"
    assert result.detail["evaluation_metric"] == "r2"
    assert result.detail["higher_is_better"] is True


def test_ablation_r2_kills_noise_predictor():
    """x4 is pure noise; its R² contribution should be < 0.01."""
    rng = np.random.default_rng(10)
    n = 200
    x2 = rng.uniform(1, 10, n)
    x4 = rng.uniform(1, 10, n)
    y = 4.0 * x2 + rng.normal(0, 0.05, n)
    df = pd.DataFrame({"row_id": range(n), "x2": x2, "x4": x4, "y": y})

    spec = _composite_spec(["x2", "x4"])
    c = Candidate(id=spec["id"], statement=spec["statement"], payload=spec)
    domain = UploadedTableDomain(df, [spec], evaluation_metric="r2")
    evidence = domain.evidence_for(c)
    result = AblationFalsifier(min_contribution=0.01).attempt(c, evidence, domain)

    assert result.killed, f"x4 is noise; ablation should kill composite. detail={result.detail}"
    assert "x4" in result.detail["useless_predictors"]


# ---------------------------------------------------------------------------
# Test 3: AblationFalsifier metric-aware, RMSLE direction
# ---------------------------------------------------------------------------

def test_ablation_rmsle_direction_passes_strong_predictors():
    """Under RMSLE, contribution = drop_rmsle - full_rmsle (positive = useful)."""
    rng = np.random.default_rng(42)
    n = 300
    x2 = rng.uniform(1, 5, n)
    x3 = rng.uniform(1, 5, n)
    y = np.exp(0.5 * x2 + 0.5 * x3 + rng.normal(0, 0.05, n))
    df = pd.DataFrame({"row_id": range(n), "x2": x2, "x3": x3, "y": y})

    spec = _composite_spec(["x2", "x3"], metric_score=9.0)
    c = Candidate(id=spec["id"], statement=spec["statement"], payload=spec)
    domain = UploadedTableDomain(df, [spec], evaluation_metric="rmsle",
                                  target_transform="log1p")
    evidence = domain.evidence_for(c)
    result = AblationFalsifier(min_contribution=0.001).attempt(c, evidence, domain)

    assert not result.killed, f"Both predictors contribute; should pass. detail={result.detail}"
    assert result.detail["evaluation_metric"] == "rmsle"
    assert result.detail["higher_is_better"] is False
    contribs = result.detail["contributions"]
    # Both contributions must be positive (positive = removing predictor worsens RMSLE)
    assert contribs["x2"] > 0.0, f"x2 RMSLE contribution should be positive; got {contribs['x2']}"
    assert contribs["x3"] > 0.0, f"x3 RMSLE contribution should be positive; got {contribs['x3']}"


def test_ablation_rmsle_kills_noise_predictor():
    """Under RMSLE, a pure-noise predictor has contribution ≈ 0 and is killed."""
    rng = np.random.default_rng(77)
    n = 200
    x2 = rng.uniform(1, 5, n)
    x4 = rng.uniform(1, 5, n)  # noise
    y = np.exp(1.0 * x2 + rng.normal(0, 0.05, n))
    df = pd.DataFrame({"row_id": range(n), "x2": x2, "x4": x4, "y": y})

    spec = _composite_spec(["x2", "x4"], metric_score=9.0)
    c = Candidate(id=spec["id"], statement=spec["statement"], payload=spec)
    domain = UploadedTableDomain(df, [spec], evaluation_metric="rmsle",
                                  target_transform="log1p")
    evidence = domain.evidence_for(c)
    result = AblationFalsifier(min_contribution=0.01).attempt(c, evidence, domain)

    assert result.killed, f"x4 is noise; RMSLE ablation should kill composite. detail={result.detail}"
    assert "x4" in result.detail["useless_predictors"]


# ---------------------------------------------------------------------------
# Test 4: AblationFalsifier detail records required fields
# ---------------------------------------------------------------------------

def test_ablation_detail_fields():
    """Result detail must include: evaluation_metric, higher_is_better, full_score, partition."""
    df = _make_strong_df()
    spec = _composite_spec(["x2", "x3"])
    c = Candidate(id=spec["id"], statement=spec["statement"], payload=spec)
    domain = UploadedTableDomain(df, [spec], evaluation_metric="rmsle",
                                  target_transform="log1p")
    evidence = domain.evidence_for(c)
    result = AblationFalsifier(min_contribution=0.001).attempt(c, evidence, domain)

    d = result.detail
    assert "evaluation_metric" in d, "detail must include evaluation_metric"
    assert "higher_is_better" in d, "detail must include higher_is_better"
    assert "full_score" in d, "detail must include full_score"
    assert "partition" in d, "detail must include partition"
    assert d["partition"] == "selection"
    assert d["evaluation_metric"] == "rmsle"
    assert d["higher_is_better"] is False
    assert "per_predictor" in d, "detail must include per_predictor list"
    assert "min_contribution_threshold" in d


# ---------------------------------------------------------------------------
# Test 5: ablation_contributions in refit() is R²-diagnostic-only (renamed key)
# ---------------------------------------------------------------------------

def test_refit_ablation_key_is_diagnostic_only():
    """refit() must use 'ablation_contributions_r2_diagnostic', not 'ablation_contributions'."""
    df = _make_strong_df()
    spec = _composite_spec(["x2", "x3"])
    c = Candidate(id=spec["id"], statement=spec["statement"], payload=spec)
    domain = UploadedTableDomain(df, [spec], evaluation_metric="rmsle")
    model = domain.refit(c, df)
    assert model.get("valid"), "refit should succeed"
    assert "ablation_contributions_r2_diagnostic" in model, \
        "refit must use key 'ablation_contributions_r2_diagnostic' for R² diagnostic"
    assert "ablation_contributions" not in model, \
        "old key 'ablation_contributions' must not exist in refit() output"


# ---------------------------------------------------------------------------
# Test 6: plan_hash changes when ablation_metric is modified
# ---------------------------------------------------------------------------

def test_plan_hash_changes_with_ablation_metric():
    """ablation_metric is in the v0.3 immutable field set; changing it must change plan_hash."""
    from orbita_mvp.compiler import PLAN_SCHEMA_V03
    base_plan = {
        "schema_version": PLAN_SCHEMA_V03,
        "target_transform": "log1p",
        "outcome_domain": "nonneg",
        "evaluation_metric": "rmsle",
        "ablation_metric": "rmsle",
        "composition_strategy": "composition_v1_1_backward_elimination",
        "thresholds": {"commit_at": 0.25, "ablation_min_contribution": 0.01,
                        "ablation_min_absolute_improvement": 0.01,
                        "ablation_min_relative_improvement": None},
        "candidate_generation": {"seed": 20260623, "scout_fraction": 0.6,
                                   "confirmation_fraction": 0.25,
                                   "final_validation_fraction": 0.15},
    }
    h1 = compute_plan_hash(base_plan)
    h2 = compute_plan_hash({**base_plan, "ablation_metric": "r2"})
    h3 = compute_plan_hash({**base_plan, "ablation_metric": None})
    assert h1 != h2, "Hash must change when ablation_metric changes from rmsle to r2"
    assert h1 != h3, "Hash must change when ablation_metric changes from rmsle to None"
    assert h2 != h3, "Hash for ablation_metric=r2 must differ from None"


# ---------------------------------------------------------------------------
# Test 7: ablation_metric field is propagated to compiled plan
# ---------------------------------------------------------------------------

def test_compiled_plan_has_ablation_metric(tmp_path):
    """compile_case must set ablation_metric in the plan and include it in plan_hash."""
    from orbita_mvp.service import ResearchMVP

    rng = np.random.default_rng(1)
    n = 80
    x = rng.uniform(1, 5, n)
    y = 2.0 * x + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"row_id": range(n), "x": x, "y": y})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    with ResearchMVP(tmp_path / "am.db", tmp_path / "ws") as svc:
        case = svc.create_case(name="ablation-metric-test", goal="")
        svc.add_file(case["id"], csv_path)
        plan_rec = svc.compile_case(case["id"], evaluation_metric="rmsle")

    plan = plan_rec["plan"]
    assert "ablation_metric" in plan, "compiled plan must include ablation_metric field"
    assert plan["ablation_metric"] == "rmsle", \
        f"ablation_metric should default to evaluation_metric ('rmsle'); got {plan['ablation_metric']!r}"
    assert plan["plan_hash"], "plan must have a plan_hash"
    # Recomputing hash must match
    assert compute_plan_hash(plan) == plan["plan_hash"], \
        "plan_hash must match recomputed value"


# ---------------------------------------------------------------------------
# Test 8: ablation_min_absolute_improvement appears in plan thresholds
# ---------------------------------------------------------------------------

def test_plan_thresholds_include_ablation_fields(tmp_path):
    """Compiled plan thresholds must include ablation_min_absolute_improvement."""
    from orbita_mvp.service import ResearchMVP

    rng = np.random.default_rng(2)
    n = 80
    df = pd.DataFrame({"row_id": range(n), "x": rng.uniform(1, 5, n),
                        "y": rng.uniform(1, 5, n)})
    csv_path = tmp_path / "d.csv"
    df.to_csv(csv_path, index=False)

    with ResearchMVP(tmp_path / "th.db", tmp_path / "ws") as svc:
        case = svc.create_case(name="thresholds-test", goal="")
        svc.add_file(case["id"], csv_path)
        plan_rec = svc.compile_case(case["id"])

    thresholds = plan_rec["plan"]["thresholds"]
    assert "ablation_min_absolute_improvement" in thresholds, \
        "thresholds must include ablation_min_absolute_improvement"
    assert "ablation_min_relative_improvement" in thresholds, \
        "thresholds must include ablation_min_relative_improvement"


# ---------------------------------------------------------------------------
# Test 9: composition_v1_1_backward_elimination removes worst failing predictor
# ---------------------------------------------------------------------------

def test_backward_elimination_removes_noise_predictor():
    """y = 4*x2 + 4*x3; x4 is noise. Backward elimination must remove x4."""
    df = _make_strong_df(n=300)
    full_spec = {
        "id": "composite:y:test_be",
        "statement": "y ~ x2 + x3 + x4",
        "kind": "composite_linear",
        "composition_strategy": "composition_v1",
        "predictors": ["x2", "x3", "x4"],
        "outcome": "y",
        "parent_candidate_ids": ["p1", "p2", "p3"],
        "scout_metric": {
            "parent_scores": {"x2": 0.9, "x3": 0.9, "x4": 0.02},
            "best_individual_score": 0.9,
        },
        "parents": ["p1", "p2", "p3"],
    }
    domain = UploadedTableDomain(df, [full_spec], evaluation_metric="r2")
    reduced = build_backward_eliminated_composites(
        [full_spec], domain, min_contribution=0.01, min_predictors=2
    )

    assert len(reduced) == 1, f"Expected one reduced candidate; got {len(reduced)}"
    rc = reduced[0]
    assert "x4" not in rc["predictors"], \
        f"x4 (noise) must be eliminated; remaining: {rc['predictors']}"
    assert "x2" in rc["predictors"] and "x3" in rc["predictors"], \
        f"x2 and x3 must survive; remaining: {rc['predictors']}"
    assert rc["composition_strategy"] == "composition_v1_1_backward_elimination"
    assert len(rc["elimination_ledger"]) >= 1, "elimination_ledger must have at least one entry"
    assert rc["eliminated_predictors"] == ["x4"]


# ---------------------------------------------------------------------------
# Test 10: backward elimination records elimination_ledger with required fields
# ---------------------------------------------------------------------------

def test_backward_elimination_ledger_fields():
    """Each elimination_ledger entry must record: step, eliminated, contribution, metric, partition."""
    df = _make_strong_df(n=300)
    full_spec = {
        "id": "composite:y:be_ledger",
        "statement": "y ~ x2 + x3 + x4",
        "kind": "composite_linear",
        "composition_strategy": "composition_v1",
        "predictors": ["x2", "x3", "x4"],
        "outcome": "y",
        "parent_candidate_ids": [],
        "scout_metric": {"parent_scores": {"x2": 0.9, "x3": 0.9, "x4": 0.02},
                          "best_individual_score": 0.9},
        "parents": [],
    }
    domain = UploadedTableDomain(df, [full_spec], evaluation_metric="r2")
    reduced = build_backward_eliminated_composites([full_spec], domain, min_contribution=0.01)

    assert reduced, "Expected at least one reduced candidate"
    for entry in reduced[0]["elimination_ledger"]:
        assert "step" in entry
        assert "eliminated" in entry
        assert "contribution" in entry
        assert "full_score" in entry
        assert "metric" in entry
        assert "partition" in entry
        assert entry["partition"] == "selection"


# ---------------------------------------------------------------------------
# Test 11: model artifact is serialized, saved, loaded, and verifiable
# ---------------------------------------------------------------------------

def test_model_artifact_serialize_load_verify(tmp_path):
    """Serialize an artifact, save it, load it, verify SHA-256 integrity."""
    rng = np.random.default_rng(5)
    n = 100
    x = rng.uniform(1, 5, n)
    y = 2.5 * x + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"row_id": range(n), "x": x, "y": y})
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    plan = {
        "target_transform": None,
        "outcome_domain": None,
        "evaluation_metric": "r2",
        "ablation_metric": "r2",
        "plan_hash": "abc123",
        "selected_dataset": {
            "normalized_path": str(csv_path),
            "sha256": "fake_sha",
        },
    }
    finding = {
        "candidate": {
            "id": "linear:x_y:test",
            "statement": "x and y show a positive linear association.",
            "payload": {
                "kind": "linear_association",
                "predictor": "x",
                "outcome": "y",
                "expected_direction": "positive",
                "scout_metric": {},
                "parents": [],
            },
        },
        "final_status": "supported",
        "falsifications": [],
    }

    artifact = serialize_deployment_artifact(
        run_id="run_test",
        plan=plan,
        finding=finding,
        normalized_path=csv_path,
        selection_artifact_id="sel:y:run_test",
        final_validation_score=0.95,
        production_commit="deadbeef",
    )
    assert artifact["schema_version"] == "orbita-deployment-artifact/0.1"
    assert artifact["artifact_kind"] == "deployment"
    assert "artifact_sha256" in artifact
    assert artifact["run_id"] == "run_test"
    assert artifact["selected_model_id"] == "linear:x_y:test"
    assert "intercept" in artifact
    assert "x" in artifact["coefficients"]

    saved_path = save_model_artifact(artifact, tmp_path / "run_dir", kind="deployment")
    assert saved_path.exists()

    loaded = load_model_artifact(saved_path)
    assert loaded["artifact_sha256"] == artifact["artifact_sha256"]
    assert loaded["intercept"] == artifact["intercept"]


# ---------------------------------------------------------------------------
# Test 12: model artifact fails on hash tamper
# ---------------------------------------------------------------------------

def test_model_artifact_tamper_detection(tmp_path):
    """Modifying any artifact field must cause load_model_artifact to raise."""
    rng = np.random.default_rng(6)
    n = 80
    x = rng.uniform(1, 5, n)
    y = 2.0 * x + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"row_id": range(n), "x": x, "y": y})
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    plan = {
        "target_transform": None, "outcome_domain": None,
        "evaluation_metric": "r2", "ablation_metric": "r2",
        "plan_hash": "abc", "selected_dataset": {"normalized_path": str(csv_path), "sha256": "x"},
    }
    finding = {
        "candidate": {
            "id": "linear:x_y:tamper",
            "statement": "test",
            "payload": {"kind": "linear_association", "predictor": "x", "outcome": "y",
                         "expected_direction": "positive", "scout_metric": {}, "parents": []},
        },
        "final_status": "supported",
        "falsifications": [],
    }

    artifact = serialize_deployment_artifact(
        run_id="run_tamper",
        plan=plan,
        finding=finding,
        normalized_path=csv_path,
        selection_artifact_id="sel:y:run_tamper",
        final_validation_score=0.90,
    )
    saved_path = save_model_artifact(artifact, tmp_path / "run_dir", kind="deployment")

    # Tamper with intercept
    tampered = artifact.copy()
    tampered["intercept"] = 9999.0
    saved_path.write_text(json.dumps(tampered, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity check FAILED"):
        load_model_artifact(saved_path)


# ---------------------------------------------------------------------------
# Test 13: Staging acceptance — 3-predictor dataset, weak predictor eliminated
# ---------------------------------------------------------------------------

def test_acceptance_3predictor_backward_elimination_e2e(tmp_path):
    """End-to-end: y = 5*a + 5*b; c ≈ a (collinear proxy — high marginal r with y, near-zero conditional).
    c must pass univariate pairwise screening (|r(c,y)| ≥ 0.2) so it enters composite_v1.
    In the composite, c's conditional contribution is near-zero → backward elimination removes it.
    With composition_v1_1_backward_elimination:
    - Full composite a+b+c is proposed (composition_v1 candidate)
    - Reduced composite a+b is proposed (backward-elimination candidate)
    - Reduced composite survives all falsifiers
    - AblationFalsifier for reduced composite passes (a and b both contribute)
    """
    from orbita_mvp.service import ResearchMVP

    rng = np.random.default_rng(42)
    n = 400
    a = rng.uniform(1, 10, n)
    b = rng.uniform(1, 10, n)
    # c is correlated with a (ρ ≈ 0.85) so r(c,y) ≈ 0.85 * r(a,y) → passes pairwise threshold.
    # But in composite a+b+c, c is near-collinear with a → marginal contribution ≈ 0.
    c = 0.85 * a + 0.15 * rng.uniform(0, 1, n)
    y = 5.0 * a + 5.0 * b + rng.normal(0, 0.03, n)

    df = pd.DataFrame({"row_id": range(n), "a": a, "b": b, "c": c, "y": y})
    csv_path = tmp_path / "staging.csv"
    df.to_csv(csv_path, index=False)

    with ResearchMVP(tmp_path / "staging.db", tmp_path / "ws") as svc:
        case = svc.create_case(name="staging-backward-elim", goal="")
        svc.add_file(case["id"], csv_path)
        plan_rec = svc.compile_case(
            case["id"],
            evaluation_metric="r2",
            target_transform=None,
        )
        assert plan_rec["plan"]["composition_strategy"] == "composition_v1_1_backward_elimination"
        run_res = svc.run_case(case["id"], auto_approve=True)

    findings = run_res["result"]["findings"]
    composite_findings = [
        f for f in findings
        if f["candidate"]["payload"].get("kind") == "composite_linear"
    ]
    assert composite_findings, "Expected at least one composite finding"

    # At least one composite should use backward-elimination strategy
    be_composites = [
        f for f in composite_findings
        if f["candidate"]["payload"].get("composition_strategy") == "composition_v1_1_backward_elimination"
    ]
    assert be_composites, (
        "Expected at least one composite with composition_v1_1_backward_elimination strategy"
    )

    # The reduced (a+b) composite should not contain c
    for f in be_composites:
        preds = f["candidate"]["payload"].get("predictors", [])
        assert "c" not in preds, (
            f"Backward-eliminated composite must not contain noise predictor 'c'; "
            f"got predictors={preds}"
        )

    # At least one backward-eliminated composite must survive (not be refuted)
    be_survivors = [
        f for f in be_composites
        if f["final_status"] != "refuted"
        and not any(atk["killed"] for atk in f["falsifications"])
    ]
    assert be_survivors, (
        "Expected at least one backward-eliminated composite to survive all falsifiers"
    )

    # Verify ablation passed for the surviving reduced composite
    for f in be_survivors:
        abl = next((atk for atk in f["falsifications"] if atk["name"] == "ablation"), None)
        assert abl is not None, "Surviving composite must have AblationFalsifier entry"
        assert not abl.get("killed"), (
            f"AblationFalsifier must PASS for reduced composite {f['candidate']['id']}; "
            f"detail={abl.get('detail')}"
        )
        assert abl["detail"].get("partition") == "selection", \
            "AblationFalsifier must record partition='selection'"

    # Verify model artifact was serialized for the selected model
    model_artifacts = run_res["result"].get("model_artifacts", {})
    assert model_artifacts, "run result must include model_artifacts"
    for outcome_col, art_info in model_artifacts.items():
        assert "model_artifact_path" in art_info or "error" in art_info, (
            f"model_artifacts[{outcome_col!r}] must have model_artifact_path or error"
        )
        if "model_artifact_path" in art_info:
            ap = pathlib.Path(art_info["model_artifact_path"])
            assert ap.exists(), f"Model artifact file must exist at {ap}"
            loaded = load_model_artifact(ap)
            assert loaded["run_id"] == run_res["result"]["run_id"]
