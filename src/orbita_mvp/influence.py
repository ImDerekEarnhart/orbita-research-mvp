"""High-leverage influence diagnostics for linear findings.

A relation can clear every falsification threshold and still be an artifact of
one or two extreme observations. In the allometry dataset the raw
body-mass/metabolic-rate relation is dominated by the blue whale (100,000 kg);
without it the apparent linear law is far weaker. This module flags such
findings so the dashboard never shows a leverage-dominated raw relation as if it
were as solid as the log-space allometric result.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def linear_influence_warning(
    df: pd.DataFrame,
    predictor: str,
    outcome: str,
) -> dict[str, Any] | None:
    """Return an influence warning for outcome ~ predictor, or None if robust.

    Uses ordinary least-squares leverage (hat values) and Cook's distance. A
    finding is flagged when a single observation has Cook's distance > 1 (the
    classic "highly influential" threshold) or leverage above 0.5, or when
    dropping the single highest-leverage point cuts R^2 by more than half.
    """
    if predictor not in df.columns or outcome not in df.columns:
        return None
    pair = df[[predictor, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(pair)
    if n < 6:
        return None
    x = pair[predictor].to_numpy(float)
    y = pair[outcome].to_numpy(float)
    if np.ptp(x) <= 1e-15:
        return None

    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    p = 2  # intercept + slope
    dof = n - p
    if dof <= 0:
        return None
    mse = float(resid @ resid) / dof
    r2_full = _r2(y, X @ beta)

    # Hat (leverage) values from the diagonal of X (XᵀX)⁻¹ Xᵀ.
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    hat = np.einsum("ij,jk,ik->i", X, xtx_inv, X)

    cooks = np.zeros(n)
    if mse > 1e-30:
        denom = (1.0 - hat) ** 2
        safe = denom > 1e-12
        cooks[safe] = (resid[safe] ** 2 / (p * mse)) * (hat[safe] / denom[safe])

    max_cooks = float(np.max(cooks))
    max_leverage = float(np.max(hat))
    top = int(np.argmax(hat))

    # Refit without the single highest-leverage point.
    keep = np.ones(n, dtype=bool)
    keep[top] = False
    Xk, yk = X[keep], y[keep]
    beta_k, *_ = np.linalg.lstsq(Xk, yk, rcond=None)
    r2_reduced = _r2(yk, Xk @ beta_k)

    r2_drop = None
    if math.isfinite(r2_full) and math.isfinite(r2_reduced):
        r2_drop = r2_full - r2_reduced

    dominated = (
        max_cooks > 1.0
        or max_leverage > 0.5
        or (r2_drop is not None and r2_full > 0.3 and r2_drop > 0.5 * r2_full)
    )
    if not dominated:
        return None

    return {
        "warning": "high_leverage_dominance",
        "message": (
            "This relation passes the thresholds but is dominated by one or more "
            "high-leverage observations; treat its strength with caution relative "
            "to transform-stable findings."
        ),
        "max_cooks_distance": round(max_cooks, 4),
        "max_leverage": round(max_leverage, 4),
        "leverage_threshold": round(2.0 * p / n, 4),
        "dominant_row_index": int(pair.index[top]),
        "r2_full": None if not math.isfinite(r2_full) else round(r2_full, 4),
        "r2_without_dominant": None if not math.isfinite(r2_reduced) else round(r2_reduced, 4),
    }


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-15:
        return float("nan")
    return 1.0 - float(np.sum((y - pred) ** 2)) / denom
