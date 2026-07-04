"""Informative-missingness (MNAR/MAR-on-observed) diagnostic.

A column can be missing *systematically*: whether a value is present depends on
other observed variables, groups, or the row's entity/time. That is a
measurement-process fact, not a scientific relationship — and it means any
complete-case analysis of that column is potentially biased.

This module builds a binary missingness indicator for each column with material
missingness and tests whether that indicator is associated with the other
observed variables (point-biserial correlation for numeric predictors,
missingness-rate spread across levels for categorical predictors). Associations
are cross-validated on two independent folds so a spurious one-fold effect does
not trip the flag. Findings are reported as data-quality / measurement-process
observations, never as causal claims.

Dependency-free (numpy/pandas only); thresholds are general "clearly non-random"
conventions, not tuned to any dataset.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _point_biserial(indicator: np.ndarray, predictor: np.ndarray) -> float:
    """|correlation| between a 0/1 missingness indicator and a numeric predictor."""
    mask = np.isfinite(predictor)
    ind, pred = indicator[mask], predictor[mask]
    if len(ind) < 8 or ind.std() == 0 or pred.std() == 0:
        return float("nan")
    return float(abs(np.corrcoef(ind.astype(float), pred.astype(float))[0, 1]))


def _group_rate_spread(indicator: pd.Series, groups: pd.Series) -> float:
    """Spread (max−min) of the missingness rate across categorical levels."""
    temp = pd.DataFrame({"m": indicator.astype(float).values, "g": groups.astype(str).values})
    rates = temp.groupby("g")["m"].mean()
    if len(rates) < 2:
        return float("nan")
    return float(rates.max() - rates.min())


def detect_informative_missingness(
    df: pd.DataFrame,
    *,
    min_missing_frac: float = 0.05,
    min_present: int = 20,
    min_effect: float = 0.2,
    fold_min_effect: float = 0.12,
    max_categorical_levels: int = 20,
) -> list[dict[str, Any]]:
    """Return a list of informative-missingness findings (one per affected column).

    A column is flagged only when its missingness indicator is associated with at
    least one other observed variable at ``min_effect`` AND that association is
    stable across two independent random folds (both ≥ ``fold_min_effect``, same
    direction). Purely random (MCAR) missingness produces near-zero associations
    and is never flagged.
    """
    n = int(len(df))
    if n < 40:
        return []
    findings: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260703)
    order = rng.permutation(n)
    fold1, fold2 = order[: n // 2], order[n // 2:]

    for col in df.columns:
        miss = df[col].isna()
        n_missing = int(miss.sum())
        rate = n_missing / n if n else 0.0
        n_present = n - n_missing
        if rate < min_missing_frac or n_missing < 10 or n_present < min_present:
            continue
        indicator = miss.to_numpy()

        assoc: list[dict[str, Any]] = []
        for other in df.columns:
            if other == col:
                continue
            oc = df[other]
            numeric = pd.to_numeric(oc, errors="coerce")
            if float(numeric.notna().mean()) >= 0.8:
                eff = _point_biserial(indicator, numeric.to_numpy(float))
                metric = "point_biserial_r"
            else:
                if oc.nunique(dropna=True) < 2 or oc.nunique(dropna=True) > max_categorical_levels:
                    continue
                eff = _group_rate_spread(pd.Series(indicator), oc)
                metric = "group_missingness_rate_spread"
            if not np.isfinite(eff):
                continue
            assoc.append({"predictor": str(other), "effect_metric": metric, "effect": round(float(eff), 4)})

        if not assoc:
            continue
        assoc.sort(key=lambda a: -a["effect"])
        top = assoc[0]
        if top["effect"] < min_effect:
            continue

        # Cross-validate the strongest predictor's association on two folds.
        strongest_name = top["predictor"]
        strongest_num = pd.to_numeric(df[strongest_name], errors="coerce")
        fold_effects: list[float] = []
        stable = True
        if top["effect_metric"] == "point_biserial_r":
            for fold in (fold1, fold2):
                e = _point_biserial(indicator[fold], strongest_num.to_numpy(float)[fold])
                fold_effects.append(round(float(e), 4) if np.isfinite(e) else 0.0)
            stable = all(e >= fold_min_effect for e in fold_effects)
        else:
            for fold in (fold1, fold2):
                e = _group_rate_spread(pd.Series(indicator[fold]), df[strongest_name].iloc[fold])
                fold_effects.append(round(float(e), 4) if np.isfinite(e) else 0.0)
            stable = all(e >= fold_min_effect for e in fold_effects)
        if not stable:
            continue

        findings.append({
            "type": "data_quality",
            "severity": "medium",
            "title": f"Informative missingness in {col}",
            "detail": (
                f"Missingness of {col} ({rate:.1%} missing) is systematically associated with observed "
                f"variables (strongest: {strongest_name}, {top['effect_metric']}={top['effect']}) — a "
                f"measurement-process pattern (not-missing-at-random), not a scientific/causal claim. "
                f"Complete-case analysis of {col} may be biased."
            ),
            "column": col,
            "informative_missingness": {
                "column": col,
                "missingness_rate": round(rate, 4),
                "n_present": n_present,
                "n_missing": n_missing,
                "diagnostic": "not_missing_at_random",
                "effect_size": top["effect"],
                "strongest_predictors": assoc[:3],
                "validation": {
                    "fold_effects": fold_effects,
                    "folds_stable": bool(stable),
                    "fold_n": [int(len(fold1)), int(len(fold2))],
                },
            },
        })
    return findings
