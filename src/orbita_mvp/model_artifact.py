"""Two-phase frozen model artifacts for the Orbita discovery pipeline.

Selection artifact
------------------
Created immediately after model selection is frozen, before final-validation
exposure.  Fitted on the **scout** partition only — the same data used by
ImprovementFalsifier during selection.  Contains the exact coefficients that
were implicitly evaluated during the selection phase.

The final-validation score is computed by applying these stored coefficients
to the held-out final-validation rows.  No new fitting occurs during
final-validation scoring.

Deployment artifact
-------------------
Created *only after* the report-only final-validation score has been recorded.
Refits the same frozen structure (predictors, transform, outcome-domain rule)
once on all available training rows (the full normalized CSV).  This is the
artifact loaded by ``/predict``; it is deliberately distinct from the selection
artifact and carries a reference to the selection artifact it was derived from.

The final-validation score remains explicitly associated with the selection
artifact, not the deployment artifact.

Security note
-------------
Column names are stored as-is.  Semantic meanings must not be inferred or
added here or in any code that reads these artifacts.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "scipy"):
        try:
            import importlib.metadata
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_sha256(artifact: dict[str, Any]) -> str:
    canonical = json.dumps(
        {k: v for k, v in artifact.items() if k != "artifact_sha256"},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _coefficients_from_model(model: dict, payload: dict) -> tuple[float, dict[str, float], list[str]]:
    """Extract (intercept, coefficients, predictor_order) from a refit model dict."""
    kind = payload["kind"]
    intercept = float(model["intercept"])
    if kind in {"linear_association", "binary_indicator"}:
        predictor = str(payload["predictor"])
        return intercept, {predictor: float(model["slope"])}, [predictor]
    elif kind == "composite_linear":
        order = list(model["predictors"])
        coefs = {p: float(model["coefficients"][p]) for p in order}
        return intercept, coefs, order
    elif kind == "nonlinear_association":
        params = model.get("params", {}) or {}
        order = [k for k in params if k != "intercept"]
        coefs = {k: float(params[k]) for k in order}
        return intercept, coefs, order
    raise ValueError(f"Artifact serialization not supported for kind={kind!r}")


def model_from_artifact(artifact: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a model dict (compatible with UploadedTableDomain) from an artifact.

    No fitting is performed.  The stored coefficients are used directly.
    """
    kind = artifact["kind"]
    intercept = artifact["intercept"]
    if kind in {"linear_association", "binary_indicator"}:
        predictor = str(payload.get("predictor", artifact["predictor_order"][0]))
        return {
            "kind": kind, "valid": True,
            "intercept": intercept,
            "slope": artifact["coefficients"][predictor],
            "target_transform": artifact.get("target_transform"),
        }
    elif kind == "composite_linear":
        return {
            "kind": kind, "valid": True,
            "intercept": intercept,
            "coefficients": artifact["coefficients"],
            "predictors": artifact["predictor_order"],
            "target_transform": artifact.get("target_transform"),
        }
    elif kind == "nonlinear_association":
        return {
            "kind": kind, "valid": True,
            "form": artifact.get("form") or payload.get("form"),
            "intercept": intercept,
            "params": {"intercept": intercept, **artifact["coefficients"]},
        }
    return {"kind": kind, "valid": False}


def serialize_selection_artifact(
    *,
    run_id: str,
    plan: dict[str, Any],
    finding: dict[str, Any],
    domain: Any,
    production_commit: str | None = None,
) -> dict[str, Any]:
    """Fit on scout partition and serialize selection artifact.

    MUST be called before final-validation rows are touched.
    Uses ``domain.scout`` — the same data available during ImprovementFalsifier.
    """
    from orbita_discovery.core import Candidate

    candidate = finding["candidate"]
    payload = candidate["payload"]
    kind = payload["kind"]
    outcome = str(payload.get("outcome", ""))

    c = Candidate(id=candidate["id"], statement=candidate["statement"], payload=payload)
    model = domain.refit(c, domain.scout)
    if not model.get("valid"):
        raise ValueError(
            f"Selection artifact serialization failed: refit on scout returned invalid model "
            f"for candidate {candidate['id']}"
        )

    intercept, coefficients, predictor_order = _coefficients_from_model(model, payload)
    target_transform = plan.get("target_transform") or None
    outcome_domain = plan.get("outcome_domain") or None

    inverse_transform = {
        "log1p": "expm1(x) — i.e., exp(x) - 1", None: "identity",
    }.get(target_transform, str(target_transform))
    domain_rule = {
        "nonneg": "clip predictions to ≥ 0", None: "no clipping",
    }.get(outcome_domain, str(outcome_domain))

    artifact: dict[str, Any] = {
        "schema_version": "orbita-selection-artifact/0.1",
        "artifact_kind": "selection",
        "selection_artifact_id": f"sel:{outcome}:{run_id}",
        "run_id": run_id,
        "plan_hash": plan.get("plan_hash", ""),
        "selected_model_id": candidate["id"],
        "outcome": outcome,
        "kind": kind,
        "form": payload.get("form"),
        "predictor_order": predictor_order,
        "intercept": intercept,
        "coefficients": coefficients,
        "target_transform": target_transform,
        "inverse_transform": inverse_transform,
        "outcome_domain_rule": domain_rule,
        "training_partition": "scout",
        "training_row_count": len(domain.scout),
        "fitting_method": "numpy.linalg.lstsq (rcond=None) on scout partition",
        "library_versions": _library_versions(),
        "production_commit": production_commit or _git_commit(),
    }
    artifact["artifact_sha256"] = _artifact_sha256(artifact)
    return artifact


def serialize_deployment_artifact(
    *,
    run_id: str,
    plan: dict[str, Any],
    finding: dict[str, Any],
    normalized_path: str | Path,
    selection_artifact_id: str,
    final_validation_score: float | None,
    production_commit: str | None = None,
) -> dict[str, Any]:
    """Refit on full training CSV and serialize deployment artifact.

    MUST be called only after final-validation scoring has been recorded.
    The ``selection_artifact_id`` links this deployment artifact back to the
    selection artifact whose coefficients were used for final-validation scoring.
    ``final_validation_score`` is embedded for traceability but is report-only.
    """
    import pandas as pd
    from orbita_discovery.core import Candidate
    from .table_domain import UploadedTableDomain

    candidate = finding["candidate"]
    payload = candidate["payload"]
    kind = payload["kind"]
    outcome = str(payload.get("outcome", ""))

    train_df = pd.read_csv(normalized_path)
    row_count = len(train_df)
    target_transform = plan.get("target_transform") or None
    outcome_domain = plan.get("outcome_domain") or None
    evaluation_metric = plan.get("evaluation_metric", "r2")

    domain = UploadedTableDomain(train_df, [candidate],
                                 target_transform=target_transform,
                                 evaluation_metric=evaluation_metric)
    c = Candidate(id=candidate["id"], statement=candidate["statement"], payload=payload)
    model = domain.refit(c, train_df)
    if not model.get("valid"):
        raise ValueError(
            f"Deployment artifact serialization failed: refit on full CSV returned invalid "
            f"model for candidate {candidate['id']}"
        )

    intercept, coefficients, predictor_order = _coefficients_from_model(model, payload)

    inverse_transform = {
        "log1p": "expm1(x) — i.e., exp(x) - 1", None: "identity",
    }.get(target_transform, str(target_transform))
    domain_rule = {
        "nonneg": "clip predictions to ≥ 0", None: "no clipping",
    }.get(outcome_domain, str(outcome_domain))

    normalized_sha256 = _sha256_file(normalized_path)

    artifact: dict[str, Any] = {
        "schema_version": "orbita-deployment-artifact/0.1",
        "artifact_kind": "deployment",
        "model_artifact_id": f"dep:{outcome}:{run_id}",
        "selection_artifact_id": selection_artifact_id,
        "final_validation_score_from_selection_artifact": final_validation_score,
        "run_id": run_id,
        "plan_hash": plan.get("plan_hash", ""),
        "selected_model_id": candidate["id"],
        "outcome": outcome,
        "kind": kind,
        "form": payload.get("form"),
        "predictor_order": predictor_order,
        "intercept": intercept,
        "coefficients": coefficients,
        "target_transform": target_transform,
        "inverse_transform": inverse_transform,
        "outcome_domain_rule": domain_rule,
        "training_partition": "all_rows",
        "normalized_file_sha256": normalized_sha256,
        "normalized_path": str(normalized_path),
        "row_count": row_count,
        "fitting_method": "numpy.linalg.lstsq (rcond=None) on full training set",
        "library_versions": _library_versions(),
        "production_commit": production_commit or _git_commit(),
    }
    artifact["artifact_sha256"] = _artifact_sha256(artifact)
    return artifact


def save_model_artifact(artifact: dict[str, Any], run_dir: Path, *, kind: str = "") -> Path:
    """Write artifact to ``{run_dir}/artifacts/{outcome}_{kind}_artifact.json``."""
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    outcome = artifact.get("outcome", "unknown")
    suffix = f"_{kind}" if kind else ""
    path = artifacts_dir / f"{outcome}{suffix}_artifact.json"
    path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    return path


def load_model_artifact(artifact_path: Path | str) -> dict[str, Any]:
    """Load and verify a model artifact.  Raises ``ValueError`` on hash mismatch."""
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            "This historical run predates serialized deployment artifacts and cannot "
            "be used for new inference. Create a new case and run under plan schema 0.3."
        )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    stored_sha = artifact.get("artifact_sha256", "")
    computed_sha = _artifact_sha256(artifact)
    if stored_sha != computed_sha:
        raise ValueError(
            f"Model artifact integrity check FAILED at {path}. "
            f"Stored SHA-256: {stored_sha[:16]}…  Computed: {computed_sha[:16]}…"
        )
    return artifact
