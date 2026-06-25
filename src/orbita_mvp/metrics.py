"""Metric computation and direction utilities for Orbita discovery evaluation.

Supported metrics
-----------------
r2     : R² coefficient of determination.  Higher is better.
rmse   : Root mean squared error.           Lower is better.
mae    : Mean absolute error.               Lower is better.
rmsle  : Root mean squared log error.       Lower is better.
         Requires nonnegative actuals and predictions; clips both to ≥ 0
         before computing log1p so the formula is always well-defined.

Do NOT equate log-space R² with RMSLE.  A model fitted with ``log1p``
target transform will have a high log-space R² iff log(y+1) is linear
in the predictors.  RMSLE measures prediction accuracy in the original
space after the inverse transform is applied.  They are related but not
identical: a model can improve log-space R² while worsening RMSLE
(e.g. when the inverse transform magnifies scale errors).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

SUPPORTED_METRICS: frozenset[str] = frozenset({"r2", "rmse", "mae", "rmsle"})

# Metrics where a *larger* value is better.
_HIGHER_IS_BETTER: frozenset[str] = frozenset({"r2"})

# The score that a null (constant-mean) model achieves under each metric.
# Used by ImprovementFalsifier when no baseline is available.
NULL_SCORE: dict[str, float] = {
    "r2": 0.0,
    "rmse": float("inf"),
    "mae": float("inf"),
    "rmsle": float("inf"),
}


def higher_is_better(metric: str) -> bool:
    """Return True when a larger score is better (R²), False for error metrics."""
    _validate(metric)
    return metric in _HIGHER_IS_BETTER


def validate_metric(metric: str) -> str:
    """Raise ValueError for unknown metrics; return the metric unchanged."""
    _validate(metric)
    return metric


def _validate(metric: str) -> None:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unknown evaluation_metric {metric!r}. "
            f"Supported: {sorted(SUPPORTED_METRICS)}"
        )


# ---------------------------------------------------------------------------
# Core formula implementations
# ---------------------------------------------------------------------------

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 1e-15:
        return 0.0
    score = 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom
    return max(-1.0, min(1.0, score))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exact RMSLE formula.  Clips both arrays to ≥ 0 before log1p."""
    y_true = np.clip(y_true, 0.0, None)
    y_pred = np.clip(y_pred, 0.0, None)
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    return float(np.sqrt(np.mean(log_diff ** 2)))


_FORMULA = {"r2": _r2, "rmse": _rmse, "mae": _mae, "rmsle": _rmsle}


def compute_metric(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute *metric* on finite (y_true, y_pred) pairs.

    Pairs where either value is non-finite are excluded.  Returns NaN when
    fewer than 3 finite pairs remain.
    """
    _validate(metric)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t, y_p = y_true[mask], y_pred[mask]
    if len(y_t) < 3:
        return float("nan")
    return _FORMULA[metric](y_t, y_p)


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def is_improvement(
    metric: str,
    new_score: float,
    baseline: float,
    min_delta: float,
) -> bool:
    """Return True when *new_score* beats *baseline* by at least *min_delta*.

    Direction is metric-aware: for R² "beats" means *larger*; for error
    metrics it means *smaller*.
    """
    _validate(metric)
    if not math.isfinite(new_score) or not math.isfinite(baseline):
        return False
    if higher_is_better(metric):
        return (new_score - baseline) >= min_delta
    else:
        return (baseline - new_score) >= min_delta


def select_best_finding(
    findings: list[dict[str, Any]],
    metric: str,
    *,
    score_key: str = "final_validation_metric_score",
    fallback_key: str = "verdict_score",
) -> dict[str, Any]:
    """Return the finding with the best score under *metric*.

    Looks for *score_key* on each finding, falls back to *fallback_key*
    (defaults to the R² verdict score stored there).  Tie-break is
    deterministic: lexicographic candidate ID.

    Raises ValueError for an empty list.
    """
    if not findings:
        raise ValueError("select_best_finding called with an empty list")
    _validate(metric)
    hib = higher_is_better(metric)

    def sort_key(f: dict[str, Any]):
        score = f.get(score_key)
        if score is None or not math.isfinite(float(score)):
            score = f.get(fallback_key, f.get("verdict", {}).get("score", 0.0))
        score = float(score)
        signed = score if hib else -score
        return (signed, f["candidate"]["id"])

    return max(findings, key=sort_key)
