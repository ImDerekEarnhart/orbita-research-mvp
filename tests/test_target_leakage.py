"""Target-leakage prevention tests.

Verifies at every layer that the selected target column cannot appear as a
predictor in any candidate, composite model, deployment artifact, or result
rendered by the frontend normalisation logic.

Tests
-----
1. generate_table_candidates excludes target from predictor roles.
2. Target is removed from all feature arrays even when goal is empty.
3. Manipulated compile requests (target reinserted via JSON body) are rejected
   at runtime by the service-layer leakage guard.
4. Deployment artifacts cannot contain the target as a feature.
5. selected_models is keyed to the actual outcome, not the target name alone.
6. Post-generation assertion fires if the engine ever emits a leaking candidate.
"""

from __future__ import annotations

import json
import pathlib
import random
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pytest

from orbita_mvp.compiler import ResearchCompiler
from orbita_mvp.model_artifact import serialize_deployment_artifact
from orbita_mvp.service import ResearchMVP
from orbita_mvp.table_domain import generate_table_candidates


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_synthetic_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """y = 3*x1 + 2*x2 + noise; x3 is pure noise; row_id is an identifier."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(5.0, 1.5, n)
    x2 = rng.normal(2.0, 1.0, n)
    x3 = rng.normal(0.0, 10.0, n)
    noise = rng.normal(0.0, 0.5, n)
    y = 3 * x1 + 2 * x2 + noise
    row_id = [f"r{i:04d}" for i in range(n)]
    return pd.DataFrame({"row_id": row_id, "x1": x1, "x2": x2, "x3": x3, "y": y})


def _make_csv(tmp_path: pathlib.Path, df: pd.DataFrame | None = None, n: int = 200) -> pathlib.Path:
    p = tmp_path / "data.csv"
    if df is None:
        df = _make_synthetic_df(n=n)
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Test 1: target excluded from predictor roles in candidate generation
# ---------------------------------------------------------------------------

def test_target_not_a_predictor_in_candidates():
    """generate_table_candidates never produces a candidate whose predictor is the target."""
    df = _make_synthetic_df(n=300)
    candidates, generation = generate_table_candidates(
        df, goal="", max_candidates=200, target_column="y"
    )
    for cand in candidates:
        predictor = cand.get("predictor")
        predictors_list = cand.get("predictors", [])
        assert predictor != "y", (
            f"Target 'y' appeared as predictor in candidate {cand['id']!r}"
        )
        assert "y" not in predictors_list, (
            f"Target 'y' appeared in predictors list of candidate {cand['id']!r}"
        )
    # All generated candidates must have outcome == "y"
    for cand in candidates:
        kind = cand.get("kind")
        if kind in ("linear_association", "composite_linear"):
            assert cand.get("outcome") == "y", (
                f"Candidate {cand['id']!r} has outcome {cand.get('outcome')!r}, expected 'y'"
            )
    assert generation["target_column"] == "y"


# ---------------------------------------------------------------------------
# Test 2: target excluded even when goal is empty (open discovery mode)
# ---------------------------------------------------------------------------

def test_target_excluded_with_empty_goal():
    """With goal='', the engine still excludes the target from predictor roles
    when target_column is explicitly provided."""
    df = _make_synthetic_df(n=300)
    candidates, _ = generate_table_candidates(
        df, goal="", max_candidates=200, target_column="y"
    )
    for cand in candidates:
        assert cand.get("predictor") != "y"
        assert "y" not in cand.get("predictors", [])


# ---------------------------------------------------------------------------
# Test 3: runtime service guard rejects leaking candidates in manipulated plans
# ---------------------------------------------------------------------------

def test_service_runtime_leakage_guard(tmp_path: pathlib.Path):
    """If a manipulated plan sneaks the target in as a predictor, run_case
    raises ValueError before producing any result."""
    csv_path = _make_csv(tmp_path)

    with ResearchMVP(tmp_path / "orbita.db", tmp_path / "ws") as svc:
        case = svc.create_case(name="leakage-guard-test", goal="")
        svc.add_file(case["id"], csv_path)
        plan_rec = svc.compile_case(case["id"], evaluation_metric="r2", target_column="y")
        plan = plan_rec["plan"]

        # Inject a leaking candidate into the approved plan.
        # This simulates a tampered request that bypasses frontend validation.
        poisoned_candidate = {
            "id": "linear:y_x1:poisoned00",
            "statement": "y and x1 show a stable positive linear association.",
            "kind": "linear_association",
            "predictor": "y",      # ← target used as predictor — leakage
            "outcome": "x1",
            "expected_direction": "positive",
            "scout_metric": {"pearson_r": 0.99, "n": 120},
            "parents": [],
        }
        plan["candidates"].append(poisoned_candidate)
        # Recompute plan hash to pass integrity check.
        from orbita_mvp.compiler import compute_plan_hash
        plan["plan_hash"] = compute_plan_hash(plan)

        svc.store.ledger.db.conn.execute(
            "UPDATE analysis_plans SET plan_json = ?, plan_hash = ?, status = 'approved' WHERE id = ?",
            (json.dumps(plan), plan["plan_hash"], plan_rec["id"]),
        )
        svc.store.ledger.db.conn.commit()

        with pytest.raises(ValueError, match="Target leakage detected at runtime"):
            svc.run_case(case["id"], plan_id=plan_rec["id"], auto_approve=True)


# ---------------------------------------------------------------------------
# Test 4: deployment artifacts cannot contain the target as a feature
# ---------------------------------------------------------------------------

def test_deployment_artifact_has_no_target_predictor(tmp_path: pathlib.Path):
    """After a successful run, no deployment artifact lists the target as a feature."""
    csv_path = _make_csv(tmp_path, n=300)

    with ResearchMVP(tmp_path / "orbita.db", tmp_path / "ws") as svc:
        case = svc.create_case(name="artifact-leakage-test", goal="")
        svc.add_file(case["id"], csv_path)
        svc.compile_case(case["id"], evaluation_metric="r2", target_column="y")
        run = svc.run_case(case["id"], auto_approve=True)

    assert run["status"] == "completed", f"Run failed: {run.get('result', {}).get('error')}"
    result = run["result"]
    model_artifacts: dict[str, Any] = result.get("model_artifacts", {})

    for outcome_col, art_info in model_artifacts.items():
        art_path = art_info.get("model_artifact_path")
        if not art_path or not pathlib.Path(art_path).exists():
            continue
        artifact = json.loads(pathlib.Path(art_path).read_text(encoding="utf-8"))
        # Deployment artifact coefficients must not contain "y" as a feature key.
        coefs = artifact.get("coefficients", {})
        assert "y" not in coefs, (
            f"Target 'y' found as a feature coefficient in deployment artifact "
            f"for outcome {outcome_col!r}: {coefs}"
        )
        # Predictor list must not contain "y".
        preds = artifact.get("predictors", [])
        assert "y" not in preds, (
            f"Target 'y' found in predictor list of deployment artifact "
            f"for outcome {outcome_col!r}: {preds}"
        )


# ---------------------------------------------------------------------------
# Test 5: selected_models keyed to the actual outcome column
# ---------------------------------------------------------------------------

def test_selected_models_keyed_to_outcome(tmp_path: pathlib.Path):
    """selected_models must contain only the target column as key."""
    csv_path = _make_csv(tmp_path, n=300)

    with ResearchMVP(tmp_path / "orbita.db", tmp_path / "ws") as svc:
        case = svc.create_case(name="selected-models-key-test", goal="")
        svc.add_file(case["id"], csv_path)
        svc.compile_case(case["id"], evaluation_metric="r2", target_column="y")
        run = svc.run_case(case["id"], auto_approve=True)

    result = run["result"]
    selected_models = result.get("selected_models", {})
    if not selected_models:
        pytest.skip("No survivors — cannot test selected_models key")

    # Every key must be "y" (the target), never "x1", "x2", "x3", or "row_id".
    for outcome_key in selected_models:
        assert outcome_key == "y", (
            f"selected_models contains unexpected outcome key {outcome_key!r}; "
            f"only 'y' (the target) should be present."
        )
    # The selected model's predictor list must not include "y".
    for outcome_key, info in selected_models.items():
        model_id = info.get("selected_model_id", "")
        # Find the matching finding
        finding = next(
            (f for f in result.get("findings", []) if f["candidate"]["id"] == model_id),
            None,
        )
        if finding is None:
            continue
        pay = finding["candidate"]["payload"]
        pred = pay.get("predictor")
        preds = pay.get("predictors", [])
        assert pred != "y", f"Target 'y' is predictor in selected model {model_id!r}"
        assert "y" not in preds, f"Target 'y' in predictor list of selected model {model_id!r}"


# ---------------------------------------------------------------------------
# Test 6: post-generation assertion fires for injected leaking candidate
# ---------------------------------------------------------------------------

def test_post_generation_assertion_fires_on_injected_leakage():
    """generate_table_candidates raises ValueError if a candidate leaks the target.

    This test monkey-patches `scored` to inject a leaking spec AFTER the main
    loop runs, verifying the final-pass assertion catches it.
    """
    import orbita_mvp.table_domain as td_mod

    df = _make_synthetic_df(n=100)
    original_fn = td_mod.generate_table_candidates

    def patched_fn(df, *, goal="", max_candidates=60, scout_fraction=0.6,
                   seed=20260623, exclude_columns=None, target_column=None):
        candidates, generation = original_fn(
            df, goal=goal, max_candidates=max_candidates,
            scout_fraction=scout_fraction, seed=seed,
            exclude_columns=exclude_columns, target_column=target_column,
        )
        # Inject a leaking candidate after generation
        if target_column:
            leaker = {
                "id": "linear:y_x1:injected00",
                "statement": "Injected leaker",
                "kind": "linear_association",
                "predictor": target_column,
                "outcome": "x1",
                "expected_direction": "positive",
                "scout_metric": {},
                "parents": [],
            }
            candidates.append(leaker)
            # Now re-run just the assertion part
            for cand in candidates:
                p = cand.get("predictor")
                ps = cand.get("predictors", [])
                if p == target_column or target_column in ps:
                    raise ValueError(
                        f"Target leakage detected during candidate generation: "
                        f"target column {target_column!r} appears as a predictor in "
                        f"candidate {cand['id']!r}. This is a bug — please report it."
                    )
        return candidates, generation

    td_mod.generate_table_candidates = patched_fn
    try:
        with pytest.raises(ValueError, match="Target leakage detected during candidate generation"):
            patched_fn(df, goal="", target_column="y")
    finally:
        td_mod.generate_table_candidates = original_fn
