"""Subgroup-reversal / regime-dependence detection (Simpson's paradox guard).

A pooled linear association can point one way while the relationship inside every
major subgroup points the other way (Simpson's paradox). Committing the pooled
directional claim as a universal law is then wrong. This module inspects a
bounded set of eligible categorical conditioning variables Z for a
predictor->outcome pair and reports a reversal when the pooled direction
conflicts with a *stable* opposite direction inside the major subgroups.

Design constraints (bounded, general, not dataset-specific):
  * only categorical Z with a manageable number of groups are considered;
  * each analysed group must meet a configurable minimum sample size;
  * within-group direction must be bootstrap-stable to count;
  * a reversal is declared only when the major subgroups (covering the bulk of
    the data) agree on a direction opposite to the pooled one — so a genuine
    relationship that holds in every subgroup is never flagged.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _slope(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.ptp(x) <= 1e-12:
        return None
    return float(np.polyfit(x, y, 1)[0])


def _sign(v: float) -> int:
    return 1 if v > 0 else -1 if v < 0 else 0


def _bootstrap_sign_stability(x: np.ndarray, y: np.ndarray, ref_sign: int, *, seed: int, iters: int = 200) -> float:
    rng = np.random.default_rng(seed)
    n = len(x)
    agree = 0
    total = 0
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        s = _slope(x[idx], y[idx])
        if s is None or s == 0:
            continue
        total += 1
        if _sign(s) == ref_sign:
            agree += 1
    return agree / total if total else 0.0


def detect_subgroup_reversal(
    df: pd.DataFrame,
    predictor: str,
    outcome: str,
    categorical_columns: list[str],
    *,
    min_group_n: int = 25,
    max_groups: int = 6,
    min_sign_stability: float = 0.9,
    majority_fraction: float = 0.6,
    seed: int = 424242,
) -> dict[str, Any] | None:
    """Return a reversal report for the strongest conditioning variable, or None.

    The report contains the conditioning variable, pooled estimate/direction,
    per-group estimates + sample sizes + stability, a human reason, and the
    scoped per-group claims to persist.
    """
    if predictor not in df.columns or outcome not in df.columns:
        return None
    x_all = pd.to_numeric(df[predictor], errors="coerce").to_numpy(float)
    y_all = pd.to_numeric(df[outcome], errors="coerce").to_numpy(float)
    pooled = _slope(x_all, y_all)
    if pooled is None or pooled == 0:
        return None
    pooled_sign = _sign(pooled)

    best: dict[str, Any] | None = None
    best_covered = -1
    for z in categorical_columns:
        if z in (predictor, outcome) or z not in df.columns:
            continue
        groups = df[z].astype(str)
        if groups.nunique() > max_groups or groups.nunique() < 2:
            continue
        qualifying: list[dict[str, Any]] = []
        for gval, sub_idx in groups.groupby(groups).groups.items():
            sub = df.loc[sub_idx]
            xs = pd.to_numeric(sub[predictor], errors="coerce").to_numpy(float)
            ys = pd.to_numeric(sub[outcome], errors="coerce").to_numpy(float)
            n = int((np.isfinite(xs) & np.isfinite(ys)).sum())
            if n < min_group_n:
                continue
            s = _slope(xs, ys)
            if s is None:
                continue
            stab = _bootstrap_sign_stability(xs, ys, _sign(s), seed=seed)
            qualifying.append({
                "group": str(gval),
                "slope": round(s, 6),
                "direction": "positive" if s > 0 else "negative",
                "sign_stability": round(stab, 4),
                "n": n,
            })
        if len(qualifying) < 2:
            continue

        covered = sum(g["n"] for g in qualifying)
        opposing = [g for g in qualifying if _sign(g["slope"]) == -pooled_sign and g["sign_stability"] >= min_sign_stability]
        opposing_n = sum(g["n"] for g in opposing)
        # Clean reversal: every qualifying major group runs opposite the pooled
        # direction and is stable, and they cover a majority of the analysed data.
        all_opposite_and_stable = all(
            _sign(g["slope"]) == -pooled_sign and g["sign_stability"] >= min_sign_stability
            for g in qualifying
        )
        if opposing and all_opposite_and_stable and opposing_n >= majority_fraction * covered:
            if opposing_n > best_covered:
                best_covered = opposing_n
                pooled_dir = "positive" if pooled_sign > 0 else "negative"
                group_dirs = ", ".join(f"{g['group']}: {g['direction']}" for g in qualifying)
                scoped = [
                    {
                        "group_col": z,
                        "group_value": g["group"],
                        "direction": g["direction"],
                        "n": g["n"],
                        "sign_stability": g["sign_stability"],
                        "statement": (
                            f"Within {z}={g['group']}, {predictor} and {outcome} have a "
                            f"{g['direction']} association."
                        ),
                    }
                    for g in qualifying
                ]
                best = {
                    "conditioning_variable": z,
                    "pooled_slope": round(pooled, 6),
                    "pooled_direction": pooled_dir,
                    "groups": qualifying,
                    "reason": (
                        f"The pooled direction ({pooled_dir}) reverses inside every major subgroup "
                        f"of {z} ({group_dirs}). This is a subgroup reversal / Simpson's paradox, so "
                        f"the universal directional claim is not committed; scoped per-subgroup claims "
                        f"are recorded instead."
                    ),
                    "scoped_claims": scoped,
                }
    return best
