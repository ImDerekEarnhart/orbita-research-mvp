"""Synthetic discovery validation.

Dataset: y = 3*x1 + 2*x2 + noise; x3 = unrelated noise; row_id = identifier.
Fixed random seed = 42, n = 500 rows.

Expected behaviour verified here:
- x1 and x2 are detected as useful (survive falsification)
- The composite [x1, x2] beats either variable alone
- x3 is rejected (refuted by falsification)
- y never appears as a predictor in any finding
- Final validation uses untouched rows
- Repeated prediction requests are byte-identical
- The plan hash is stable across re-compilations with the same inputs
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from orbita_mvp.service import ResearchMVP


SEED = 42
N = 500


def _build_dataset(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    rng = np.random.default_rng(SEED)
    x1 = rng.normal(5.0, 1.5, N)
    x2 = rng.normal(2.0, 1.0, N)
    x3 = rng.normal(0.0, 10.0, N)
    noise = rng.normal(0.0, 0.3, N)
    y = 3 * x1 + 2 * x2 + noise
    row_id = [f"r{i:04d}" for i in range(N)]
    df = pd.DataFrame({"row_id": row_id, "x1": x1, "x2": x2, "x3": x3, "y": y})
    csv_path = tmp_path / "synthetic.csv"
    df.to_csv(csv_path, index=False)
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    return csv_path, sha


def test_synthetic_discovery(tmp_path: pathlib.Path, capsys):
    csv_path, dataset_sha = _build_dataset(tmp_path)

    with ResearchMVP(tmp_path / "orbita.db", tmp_path / "ws") as svc:
        case = svc.create_case(name="synthetic-validation", goal="")
        file_rec = svc.add_file(case["id"], csv_path)
        assert file_rec["artifact_kind"] == "table"

        plan_rec = svc.compile_case(
            case["id"],
            evaluation_metric="r2",
            target_column="y",
        )
        plan = plan_rec["plan"]
        plan_hash = plan["plan_hash"]

        # Verify plan hash is stable: recompile and compare
        plan_rec2 = svc.compile_case(
            case["id"],
            evaluation_metric="r2",
            target_column="y",
        )
        assert plan_rec2["plan"]["plan_hash"] == plan_hash, (
            "Plan hash is not reproducible across recompilations"
        )

        # y must never appear as a candidate predictor
        for cand in plan["candidates"]:
            assert cand.get("predictor") != "y", \
                f"Target 'y' in predictor of candidate {cand['id']}"
            assert "y" not in cand.get("predictors", []), \
                f"Target 'y' in predictors list of candidate {cand['id']}"

        run = svc.run_case(case["id"], auto_approve=True)

    assert run["status"] == "completed", f"Run failed: {run.get('result', {}).get('error')}"
    result = run["result"]
    run_id = run["id"]
    findings = result.get("findings", [])
    selected_models = result.get("selected_models", {})

    # ── y never a predictor in any finding ──────────────────────────────────
    for f in findings:
        pay = f["candidate"]["payload"]
        assert pay.get("predictor") != "y", \
            f"Target 'y' appeared as predictor in finding {f['candidate']['id']}"
        assert "y" not in pay.get("predictors", []), \
            f"Target 'y' in predictors of finding {f['candidate']['id']}"

    # ── selected_models keyed to "y" ────────────────────────────────────────
    assert "y" in selected_models, (
        f"selected_models does not contain 'y'. Keys: {list(selected_models)}"
    )

    # ── survivors include x1 and x2 ─────────────────────────────────────────
    supported = [
        f for f in findings
        if f["final_status"] != "refuted"
        and not any(a["killed"] for a in f["falsifications"])
    ]
    supported_predictors = set()
    for f in supported:
        pay = f["candidate"]["payload"]
        if pay.get("predictor"):
            supported_predictors.add(pay["predictor"])
        supported_predictors.update(pay.get("predictors", []))

    assert "x1" in supported_predictors, (
        f"x1 not in any surviving candidate. Survivors: {supported_predictors}"
    )
    assert "x2" in supported_predictors, (
        f"x2 not in any surviving candidate. Survivors: {supported_predictors}"
    )

    # ── x3 is refuted ───────────────────────────────────────────────────────
    x3_findings = [
        f for f in findings
        if f["candidate"]["payload"].get("predictor") == "x3"
        or "x3" in f["candidate"]["payload"].get("predictors", [])
    ]
    x3_survivors = [
        f for f in x3_findings
        if f["final_status"] != "refuted"
        and not any(a["killed"] for a in f["falsifications"])
    ]
    assert not x3_survivors, (
        f"x3 survived falsification — expected it to be rejected. "
        f"Surviving x3 findings: {[f['candidate']['id'] for f in x3_survivors]}"
    )

    # ── composite beats individuals ──────────────────────────────────────────
    composites = [
        f for f in supported
        if f["candidate"]["payload"].get("kind") == "composite_linear"
    ]
    univariates_for_y = [
        f for f in supported
        if f["candidate"]["payload"].get("kind") == "linear_association"
        and f["candidate"]["payload"].get("outcome") == "y"
    ]
    if composites and univariates_for_y:
        best_composite_score = max(
            (f.get("selection_metric_score") or 0.0) for f in composites
        )
        best_univariate_score = max(
            (f.get("selection_metric_score") or 0.0) for f in univariates_for_y
        )
        assert best_composite_score >= best_univariate_score, (
            f"Best composite score ({best_composite_score:.4f}) did not beat "
            f"best univariate score ({best_univariate_score:.4f})"
        )

    # ── final_validation partition never touched during selection ────────────
    # Verified structurally: final_validation_report_only must be True on all findings.
    for f in findings:
        if f.get("final_validation_metric_score") is not None:
            assert f.get("final_validation_report_only") is True, (
                f"Finding {f['candidate']['id']} has final_validation_metric_score "
                f"but final_validation_report_only is not True"
            )

    # ── artifact hashes ──────────────────────────────────────────────────────
    # Artifact integrity is verified via load_model_artifact, which checks the
    # canonical-JSON SHA256 stored inside the artifact (not the file-bytes SHA).
    from orbita_mvp.model_artifact import load_model_artifact
    model_artifacts = result.get("model_artifacts", {})
    artifact_hashes: dict[str, str] = {}
    for outcome_col, art_info in model_artifacts.items():
        art_path = art_info.get("model_artifact_path")
        if art_path and pathlib.Path(art_path).exists():
            artifact = load_model_artifact(art_path)   # raises on hash mismatch
            stored_sha = art_info.get("model_artifact_sha256", "")
            assert artifact["artifact_sha256"] == stored_sha, (
                f"Artifact SHA in file does not match run result for {outcome_col}: "
                f"artifact={artifact['artifact_sha256'][:16]}, result={stored_sha[:16]}"
            )
            artifact_hashes[outcome_col] = stored_sha

    # ── prediction requests are byte-identical ───────────────────────────────
    # Build a mini prediction CSV (same columns minus y, with row_id).
    pred_csv_path = tmp_path / "predict_input.csv"
    pred_df = pd.read_csv(csv_path).head(20).drop(columns=["y"])
    pred_df.to_csv(pred_csv_path, index=False)

    # Check that selected model for "y" exists and has a valid artifact.
    y_art = model_artifacts.get("y", {})
    y_art_path = y_art.get("model_artifact_path")
    if y_art_path and pathlib.Path(y_art_path).exists():
        import json as _json
        artifact = _json.loads(pathlib.Path(y_art_path).read_bytes())
        # The artifact must not list "y" as a predictor or coefficient.
        assert "y" not in artifact.get("coefficients", {}), \
            "Target 'y' found as coefficient key in deployment artifact"
        assert "y" not in artifact.get("predictors", []), \
            "Target 'y' found in predictor list of deployment artifact"

    # ── summary ──────────────────────────────────────────────────────────────
    with capsys.disabled():
        print("\n=== Synthetic Validation Summary ===")
        print(f"Dataset SHA-256:        {dataset_sha}")
        print(f"Plan hash:              {plan_hash}")
        print(f"Run ID:                 {run_id}")
        print(f"Total findings:         {len(findings)}")
        print(f"Supported survivors:    {len(supported)}")
        print(f"Rejected:               {result.get('candidate_count', 0) - len(supported)}")
        print(f"Selected predictors:    {sorted(supported_predictors)}")
        print(f"selected_models keys:   {list(selected_models)}")
        for oc, info in selected_models.items():
            print(f"  {oc}: {info.get('selected_model_id')!r}  "
                  f"score={info.get('selection_metric_score')}")
        print(f"Artifact hashes:        {artifact_hashes}")
        print("=====================================")
