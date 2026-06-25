"""Frozen deployment artifact for committed models.

After model selection is frozen (before final validation), the engine serializes
one artifact per outcome column containing all information needed to generate
predictions without refitting.  The ``/predict`` endpoint MUST load this artifact
and fail explicitly if it is missing.

Artifact layout
---------------
``{run_dir}/artifacts/{outcome}_model_artifact.json``

Security note
-------------
Column names are stored as-is.  The semantic meaning of any column is NOT recorded
here and MUST NOT be inferred or added.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


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


def serialize_model_artifact(
    *,
    run_id: str,
    plan: dict[str, Any],
    finding: dict[str, Any],
    training_df_path: str | Path,
    normalized_path: str | Path,
    production_commit: str | None = None,
) -> dict[str, Any]:
    """Compute and return the full model artifact dict.

    Parameters
    ----------
    finding:
        A surviving finding dict (from ``run_case()`` results).
    training_df_path:
        Path to the original uploaded CSV.
    normalized_path:
        Path to the normalized (re-serialized) CSV used for training.
    """
    import pandas as pd
    from orbita_discovery.core import Candidate
    from .table_domain import UploadedTableDomain

    candidate = finding["candidate"]
    payload = candidate["payload"]
    kind = payload["kind"]
    outcome = payload.get("outcome", "")

    # Refit on the full normalized CSV to get final coefficients
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
            f"Artifact serialization failed: refit on full training data returned invalid model "
            f"for candidate {candidate['id']}"
        )

    # Build coefficient dict based on kind
    intercept: float = float(model["intercept"])
    coefficients: dict[str, float] = {}
    predictor_order: list[str] = []

    if kind == "linear_association":
        predictor_order = [str(payload["predictor"])]
        coefficients = {str(payload["predictor"]): float(model["slope"])}
    elif kind == "composite_linear":
        predictor_order = list(model["predictors"])
        coefficients = {p: float(model["coefficients"][p]) for p in predictor_order}
    else:
        raise ValueError(f"Artifact serialization not supported for kind={kind!r}")

    # Inverse transform name (human-readable)
    inverse_transform = {
        "log1p": "expm1(x) — i.e., exp(x) - 1",
        None: "identity",
    }.get(target_transform, str(target_transform))

    # Outcome domain clipping rule
    domain_rule = {
        "nonneg": "clip predictions to ≥ 0",
        None: "no clipping",
    }.get(outcome_domain, str(outcome_domain))

    # File hashes
    training_sha256 = _sha256_file(training_df_path)
    normalized_sha256 = _sha256_file(normalized_path)

    artifact: dict[str, Any] = {
        "schema_version": "orbita-model-artifact/0.1",
        "model_artifact_id": f"artifact:{outcome}:{run_id}",
        "run_id": run_id,
        "plan_hash": plan.get("plan_hash", ""),
        "selected_model_id": candidate["id"],
        "outcome": outcome,
        "kind": kind,
        "predictor_order": predictor_order,
        "intercept": intercept,
        "coefficients": coefficients,
        "target_transform": target_transform,
        "inverse_transform": inverse_transform,
        "outcome_domain_rule": domain_rule,
        "training_file_sha256": training_sha256,
        "normalized_file_sha256": normalized_sha256,
        "normalized_path": str(normalized_path),
        "row_count": row_count,
        "fitting_method": "numpy.linalg.lstsq (rcond=None) on full training set",
        "library_versions": _library_versions(),
        "production_commit": production_commit or _git_commit(),
    }

    # Compute self-referential SHA-256 after all other fields are set
    canonical = json.dumps(
        {k: v for k, v in artifact.items() if k != "artifact_sha256"},
        sort_keys=True, separators=(",", ":"), default=str
    )
    artifact["artifact_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return artifact


def save_model_artifact(artifact: dict[str, Any], run_dir: Path) -> Path:
    """Write artifact to disk; return the path."""
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    outcome = artifact.get("outcome", "unknown")
    path = artifacts_dir / f"{outcome}_model_artifact.json"
    path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    return path


def load_model_artifact(artifact_path: Path | str) -> dict[str, Any]:
    """Load and verify a model artifact.  Raises on hash mismatch."""
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            "The run result may predate artifact serialization; re-run to generate."
        )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    stored_sha = artifact.get("artifact_sha256", "")
    canonical = json.dumps(
        {k: v for k, v in artifact.items() if k != "artifact_sha256"},
        sort_keys=True, separators=(",", ":"), default=str
    )
    computed_sha = hashlib.sha256(canonical.encode()).hexdigest()
    if stored_sha != computed_sha:
        raise ValueError(
            f"Model artifact integrity check FAILED at {path}. "
            f"Stored SHA-256: {stored_sha[:16]}…  Computed: {computed_sha[:16]}…"
        )
    return artifact
