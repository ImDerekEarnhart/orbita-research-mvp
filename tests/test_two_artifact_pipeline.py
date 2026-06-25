"""Tests for the two-phase frozen artifact pipeline.

Protocol invariants:
  1. FV uses stored selection coefficients (no refit on scout at FV time)
  2. _score_from_artifact performs no fitting
  3. Mutating scout rows after artifact creation cannot change FV predictions
  4. Mutating selection rows after artifact creation cannot change FV predictions
  5. Deployment refit happens only after FV scoring is recorded
  6. Selection and deployment artifact IDs and hashes are distinct
  7. FV data never enters model fitting
"""
from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from orbita_discovery.core import Candidate

from orbita_mvp.model_artifact import (
    _artifact_sha256,
    model_from_artifact,
    save_model_artifact,
    serialize_deployment_artifact,
    serialize_selection_artifact,
)
from orbita_mvp.table_domain import UploadedTableDomain


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 10, n)
    y = 2.5 * x + 1.0 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x": x, "y": y})


def _make_domain(df: pd.DataFrame) -> UploadedTableDomain:
    cand = [{"id": "c_xy", "statement": "x → y", "payload": {"kind": "linear_association", "predictor": "x", "outcome": "y"}, "parents": []}]
    return UploadedTableDomain(df, cand, scout_fraction=0.6, confirmation_fraction=0.25, final_validation_fraction=0.15, seed=20260623)


def _make_candidate() -> Candidate:
    return Candidate(id="c_xy", statement="x → y", payload={"kind": "linear_association", "predictor": "x", "outcome": "y"})


def _make_plan() -> dict:
    return {
        "schema_version": "orbita-research-plan/0.3",
        "plan_hash": "abc123",
        "target_transform": None,
        "outcome_domain": None,
        "evaluation_metric": "r2",
    }


def _make_finding(candidate_dict: dict | None = None) -> dict:
    if candidate_dict is None:
        candidate_dict = {"id": "c_xy", "statement": "x → y", "payload": {"kind": "linear_association", "predictor": "x", "outcome": "y"}, "parents": []}
    return {"candidate": candidate_dict}


# ---------------------------------------------------------------------------
# Test 1: FV uses stored selection coefficients (not a fresh scout refit)
# ---------------------------------------------------------------------------

def test_fv_uses_stored_selection_coefficients():
    """FV score is derived from selection artifact coefficients, not a fresh refit."""
    df = _make_df()
    domain = _make_domain(df)
    finding = _make_finding()
    plan = _make_plan()
    c = _make_candidate()

    # Build selection artifact (scout-fitted)
    artifact = serialize_selection_artifact(
        run_id="run_test_1",
        plan=plan,
        finding=finding,
        domain=domain,
    )

    # Reconstruct model from artifact — these are the STORED coefficients
    model_from_stored = model_from_artifact(artifact, finding["candidate"]["payload"])

    # Compute the FV score using stored model (no refit)
    fv_rows = domain.final_validation
    score_from_stored = domain.score_metric(c, model_from_stored, fv_rows)

    # Independently refit on scout, score on FV — should be the same (within float tolerance)
    model_refitted = domain.refit(c, domain.scout)
    score_from_refit = domain.score_metric(c, model_refitted, fv_rows)

    # They should agree since artifact was fitted on the same scout rows
    assert math.isclose(score_from_stored, score_from_refit, rel_tol=1e-10), (
        f"Stored: {score_from_stored}, Refit: {score_from_refit} — "
        "selection artifact coefficients must match a fresh scout refit"
    )

    # Key requirement: the stored coefficients match what was used for selection
    assert math.isclose(artifact["intercept"], model_refitted["intercept"], rel_tol=1e-10)
    assert math.isclose(artifact["coefficients"]["x"], model_refitted["slope"], rel_tol=1e-10)


# ---------------------------------------------------------------------------
# Test 2: _score_from_artifact (i.e. model_from_artifact) performs no fitting
# ---------------------------------------------------------------------------

def test_score_from_artifact_performs_no_fitting(monkeypatch):
    """model_from_artifact must NOT call numpy.linalg.lstsq."""
    import numpy.linalg as nla

    fitting_attempted = []

    original_lstsq = nla.lstsq

    def _spy_lstsq(*args, **kwargs):
        fitting_attempted.append(True)
        return original_lstsq(*args, **kwargs)

    monkeypatch.setattr(nla, "lstsq", _spy_lstsq)

    df = _make_df()
    domain = _make_domain(df)
    finding = _make_finding()
    plan = _make_plan()
    c = _make_candidate()

    # Build artifact (this WILL call lstsq — record count before)
    artifact = serialize_selection_artifact(
        run_id="run_test_2",
        plan=plan,
        finding=finding,
        domain=domain,
    )
    before_count = len(fitting_attempted)

    # model_from_artifact and the subsequent score_metric MUST NOT call lstsq
    model = model_from_artifact(artifact, finding["candidate"]["payload"])
    _ = domain.score_metric(c, model, domain.final_validation)

    after_count = len(fitting_attempted)
    assert after_count == before_count, (
        f"lstsq was called {after_count - before_count} extra time(s) during "
        "model_from_artifact / score_metric — no fitting allowed at FV time"
    )


# ---------------------------------------------------------------------------
# Test 3: Mutating scout rows after artifact creation cannot change FV predictions
# ---------------------------------------------------------------------------

def test_scout_mutation_after_artifact_cannot_change_fv_predictions():
    """Changing scout rows after selection artifact is frozen cannot alter FV scores."""
    df = _make_df()
    domain = _make_domain(df)
    finding = _make_finding()
    plan = _make_plan()
    c = _make_candidate()

    artifact = serialize_selection_artifact(
        run_id="run_test_3",
        plan=plan,
        finding=finding,
        domain=domain,
    )

    # Record FV score from stored artifact
    model_before = model_from_artifact(artifact, finding["candidate"]["payload"])
    score_before = domain.score_metric(c, model_before, domain.final_validation)

    # Corrupt ALL scout rows in place
    domain.scout["x"] = domain.scout["x"] * 999.0 + 5000.0

    # FV score from stored artifact must be unchanged
    model_after = model_from_artifact(artifact, finding["candidate"]["payload"])
    score_after = domain.score_metric(c, model_after, domain.final_validation)

    assert math.isclose(score_before, score_after, rel_tol=1e-14), (
        f"FV score changed from {score_before} to {score_after} after scout corruption. "
        "Stored artifact coefficients must be independent of scout data state."
    )


# ---------------------------------------------------------------------------
# Test 4: Mutating selection rows after artifact creation cannot change FV predictions
# ---------------------------------------------------------------------------

def test_selection_mutation_after_artifact_cannot_change_fv_predictions():
    """Changing selection rows after selection artifact is frozen cannot alter FV scores."""
    df = _make_df()
    domain = _make_domain(df)
    finding = _make_finding()
    plan = _make_plan()
    c = _make_candidate()

    artifact = serialize_selection_artifact(
        run_id="run_test_4",
        plan=plan,
        finding=finding,
        domain=domain,
    )

    model_before = model_from_artifact(artifact, finding["candidate"]["payload"])
    score_before = domain.score_metric(c, model_before, domain.final_validation)

    # Corrupt ALL confirmation (selection) rows in place
    domain.confirmation["x"] = domain.confirmation["x"] * -999.0

    model_after = model_from_artifact(artifact, finding["candidate"]["payload"])
    score_after = domain.score_metric(c, model_after, domain.final_validation)

    assert math.isclose(score_before, score_after, rel_tol=1e-14), (
        f"FV score changed from {score_before} to {score_after} after selection partition corruption. "
        "Stored artifact coefficients must be independent of selection data state."
    )


# ---------------------------------------------------------------------------
# Test 5: Deployment artifact is only created after FV scoring is recorded
# ---------------------------------------------------------------------------

def test_deployment_artifact_created_only_after_fv_scoring(tmp_path):
    """The two-artifact sequence: selection → FV → deployment."""
    df = _make_df(n=300)
    domain = _make_domain(df)
    finding = _make_finding()
    plan = _make_plan()
    c = _make_candidate()

    # Step A: selection artifact (before FV)
    sel_art = serialize_selection_artifact(
        run_id="run_test_5",
        plan=plan,
        finding=finding,
        domain=domain,
    )
    assert sel_art["artifact_kind"] == "selection"
    assert sel_art["training_partition"] == "scout"

    # Step B: FV scoring using stored selection coefficients (no refit)
    model = model_from_artifact(sel_art, finding["candidate"]["payload"])
    fv_score = domain.score_metric(c, model, domain.final_validation)
    assert fv_score is not None and math.isfinite(fv_score)

    # Step C: deployment artifact only NOW, after FV score is known
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    dep_art = serialize_deployment_artifact(
        run_id="run_test_5",
        plan=plan,
        finding=finding,
        normalized_path=csv_path,
        selection_artifact_id=sel_art["selection_artifact_id"],
        final_validation_score=fv_score,
    )
    assert dep_art["artifact_kind"] == "deployment"
    assert dep_art["training_partition"] == "all_rows"
    # FV score is embedded as report-only metadata
    assert dep_art["final_validation_score_from_selection_artifact"] == fv_score
    # Deployment artifact references the selection artifact
    assert dep_art["selection_artifact_id"] == sel_art["selection_artifact_id"]


# ---------------------------------------------------------------------------
# Test 6: Selection and deployment artifact IDs and hashes are distinct
# ---------------------------------------------------------------------------

def test_selection_and_deployment_artifacts_have_distinct_ids_and_hashes(tmp_path):
    """Selection artifact and deployment artifact must have different IDs and SHA-256 hashes."""
    df = _make_df(n=300)
    domain = _make_domain(df)
    finding = _make_finding()
    plan = _make_plan()
    c = _make_candidate()

    sel_art = serialize_selection_artifact(
        run_id="run_test_6",
        plan=plan,
        finding=finding,
        domain=domain,
    )

    model = model_from_artifact(sel_art, finding["candidate"]["payload"])
    fv_score = domain.score_metric(c, model, domain.final_validation)

    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    dep_art = serialize_deployment_artifact(
        run_id="run_test_6",
        plan=plan,
        finding=finding,
        normalized_path=csv_path,
        selection_artifact_id=sel_art["selection_artifact_id"],
        final_validation_score=fv_score,
    )

    # IDs must differ
    assert sel_art["selection_artifact_id"] != dep_art["model_artifact_id"], (
        "Selection and deployment artifact IDs must be distinct"
    )

    # SHA-256 hashes must differ
    assert sel_art["artifact_sha256"] != dep_art["artifact_sha256"], (
        "Selection and deployment artifact SHA-256 hashes must be distinct"
    )

    # Schema versions must differ
    assert sel_art["schema_version"] != dep_art["schema_version"]

    # Different artifact_kind markers
    assert sel_art["artifact_kind"] == "selection"
    assert dep_art["artifact_kind"] == "deployment"


# ---------------------------------------------------------------------------
# Test 7: FV data never enters model fitting
# ---------------------------------------------------------------------------

def test_fv_data_never_enters_model_fitting(monkeypatch):
    """The final_validation rows must never appear as input to lstsq."""
    import numpy.linalg as nla

    fitted_arrays: list[np.ndarray] = []
    original_lstsq = nla.lstsq

    def _capture_lstsq(A, b, *args, **kwargs):
        fitted_arrays.append(A.copy())
        return original_lstsq(A, b, *args, **kwargs)

    monkeypatch.setattr(nla, "lstsq", _capture_lstsq)

    df = _make_df(n=300)
    domain = _make_domain(df)
    finding = _make_finding()
    plan = _make_plan()

    # Selection artifact creation (uses scout rows — calls lstsq)
    sel_art = serialize_selection_artifact(
        run_id="run_test_7",
        plan=plan,
        finding=finding,
        domain=domain,
    )

    # FV rows
    fv_x = domain.final_validation["x"].to_numpy()

    # Verify that no captured lstsq call was fed the FV rows
    for call_matrix in fitted_arrays:
        # The design matrix has shape (n_rows, 2) — intercept column + x column
        if call_matrix.shape[1] >= 2:
            fitted_x = call_matrix[:, 1]  # x values (column 1 after intercept)
        else:
            fitted_x = call_matrix[:, 0]
        overlap = np.intersect1d(np.round(fv_x, 10), np.round(fitted_x, 10))
        assert len(overlap) == 0, (
            f"Final-validation x-values appeared in a fitting call: {overlap[:3]}. "
            "FV data must never be passed to lstsq."
        )
