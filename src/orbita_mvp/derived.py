"""Bounded multivariable derived-variable / target-leakage detection.

Some columns are not independent measurements but *constructed indices*: a
target that is (near-)deterministically reconstructed from a small subset of
other columns — e.g. a weighted score, a signed contrast, or a low-order
product. These are artifacts, not discoveries, but the pairwise near-copy
detector (which only catches a single near-identity column) misses them.

This module performs a bounded search: for each numeric target it forward-selects
a small predictor subset on the *scout* partition and validates the
reconstruction on an *untouched held-out* partition. A target is flagged only
when the held-out reconstruction is extremely tight (near-zero residual), uses
≥2 predictors, is stable across refits, and beats the best single ordinary
predictor by a real margin. This deliberately does NOT flag genuine scientific
relationships (which keep irreducible residual) or single-variable laws such as
Kepler/power laws (no multivariable margin over the best single predictor).
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[mask], pred[mask]
    if len(y) < 3:
        return float("nan")
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-15:
        return float("nan")
    return 1.0 - float(np.sum((y - pred) ** 2)) / denom


def _fit_eval(train: pd.DataFrame, test: pd.DataFrame, target: str, feats: list[str],
              multiplicative: bool = False) -> tuple[float, np.ndarray] | None:
    """Fit affine (optionally + squares/pairwise products) on train, R² on test."""
    def design(frame: pd.DataFrame) -> np.ndarray | None:
        base = [pd.to_numeric(frame[f], errors="coerce").to_numpy(float) for f in feats]
        cols = [np.ones(len(frame))] + base
        if multiplicative:
            for f in base:
                cols.append(f * f)
            for a, b in combinations(base, 2):
                cols.append(a * b)
        X = np.column_stack(cols)
        return X

    ytr = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
    Xtr = design(train)
    m = np.isfinite(ytr) & np.all(np.isfinite(Xtr), axis=1)
    if m.sum() < len(feats) + 3:
        return None
    beta, *_ = np.linalg.lstsq(Xtr[m], ytr[m], rcond=None)
    yte = pd.to_numeric(test[target], errors="coerce").to_numpy(float)
    Xte = design(test)
    mt = np.isfinite(yte) & np.all(np.isfinite(Xte), axis=1)
    if mt.sum() < 3:
        return None
    return _r2(yte[mt], Xte[mt] @ beta), beta


def _forward_select(scout: pd.DataFrame, target: str, pool: list[str], max_predictors: int) -> list[str]:
    """Greedily add the predictor that most improves scout R² (bounded search)."""
    chosen: list[str] = []
    remaining = list(pool)
    best_r2 = -np.inf
    while remaining and len(chosen) < max_predictors:
        scored = []
        for f in remaining:
            res = _fit_eval(scout, scout, target, chosen + [f])
            if res is not None and np.isfinite(res[0]):
                scored.append((res[0], f))
        if not scored:
            break
        scored.sort(reverse=True)
        gain = scored[0][0] - best_r2
        if gain <= 1e-6 and chosen:
            break
        best_r2 = scored[0][0]
        chosen.append(scored[0][1])
        remaining.remove(scored[0][1])
    return chosen


def _encode_categoricals(
    scout: pd.DataFrame,
    heldout: pd.DataFrame,
    categorical_columns: list[str],
    max_levels: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str]]:
    """Bounded one-hot/binary encoding of low-cardinality categorical predictors.

    Levels are taken from the union of scout+heldout so both frames get the same
    dummy columns. High-cardinality columns (> ``max_levels`` levels) are skipped,
    keeping IDs / repeated entities out. Returns augmented frames, the new dummy
    column names, and a ``dummy -> source column`` map for reporting.
    """
    aug_scout = scout.copy()
    aug_heldout = heldout.copy()
    dummy_pool: list[str] = []
    dummy_source: dict[str, str] = {}
    for col in categorical_columns:
        if col not in scout.columns or col not in heldout.columns:
            continue
        levels = pd.unique(pd.concat([scout[col], heldout[col]], ignore_index=True).dropna().astype(str))
        if len(levels) < 2 or len(levels) > max_levels:
            continue
        # Drop the first level to avoid perfect collinearity (reference category).
        for lvl in sorted(levels)[1:]:
            dummy = f"{col}=={lvl}"
            aug_scout[dummy] = (scout[col].astype(str) == lvl).astype(float)
            aug_heldout[dummy] = (heldout[col].astype(str) == lvl).astype(float)
            dummy_pool.append(dummy)
            dummy_source[dummy] = col
    return aug_scout, aug_heldout, dummy_pool, dummy_source


def detect_multivariable_derived(
    scout: pd.DataFrame,
    heldout: pd.DataFrame,
    target_columns: list[str],
    predictor_pool: list[str],
    *,
    categorical_columns: list[str] | None = None,
    max_categorical_levels: int = 8,
    recon_r2_min: float = 0.998,
    residual_ratio_max: float = 0.005,
    margin_min: float = 0.02,
    min_predictors: int = 2,
    max_predictors: int = 4,
    refit_stability_min: float = 0.99,
) -> dict[str, dict[str, Any]]:
    """Return ``{target -> derived-record}`` for near-deterministically reconstructed targets.

    Uses scout-only feature selection + fitting and untouched held-out validation.
    Low-cardinality categorical/binary predictors are one-hot encoded (bounded) so
    a constructed index that depends on a categorical term (e.g. a gate/flag) can
    be reconstructed; high-cardinality categoricals are skipped. Thresholds are
    near-determinism conventions (extremely high held-out R², near-zero residual,
    real margin over the best single predictor), NOT tuned to any dataset.
    """
    out: dict[str, dict[str, Any]] = {}
    if len(scout) < 10 or len(heldout) < 5:
        return out
    dummy_source: dict[str, str] = {}
    if categorical_columns:
        scout, heldout, dummy_pool, dummy_source = _encode_categoricals(
            scout, heldout, categorical_columns, max_categorical_levels
        )
        # Categorical dummies expand the predictor pool only (never targets).
        predictor_pool = list(predictor_pool) + [d for d in dummy_pool if d not in predictor_pool]

    def _to_source_vars(feats: list[str]) -> list[str]:
        """Map dummy features back to their source categorical column (de-duplicated)."""
        out_vars: list[str] = []
        for f in feats:
            src = dummy_source.get(f, f)
            if src not in out_vars:
                out_vars.append(src)
        return out_vars
    for target in target_columns:
        pool = [c for c in predictor_pool if c != target]
        if len(pool) < min_predictors:
            continue
        # Best single ordinary predictor (held-out) — the "ordinary candidate".
        best_single = -np.inf
        for f in pool:
            res = _fit_eval(scout, heldout, target, [f])
            if res is not None and np.isfinite(res[0]):
                best_single = max(best_single, res[0])
        feats = _forward_select(scout, target, pool, max_predictors)
        if len(feats) < min_predictors:
            continue
        # Held-out backward pruning: drop any predictor whose removal barely
        # changes the held-out reconstruction. This removes spurious "riders"
        # (e.g. a planted-null column that only helped by overfitting scout), so
        # the reported source set is minimal and does not falsely name unrelated
        # columns as part of the construction.
        def _ho_r2(fs: list[str]) -> float:
            r = _fit_eval(scout, heldout, target, fs, multiplicative=False)
            return r[0] if (r is not None and np.isfinite(r[0])) else -np.inf

        pruning = True
        while len(feats) > min_predictors and pruning:
            base = _ho_r2(feats)
            pruning = False
            for f in list(feats):
                reduced = [x for x in feats if x != f]
                if len(reduced) < min_predictors:
                    continue
                if _ho_r2(reduced) >= base - 0.0005:
                    feats = reduced
                    pruning = True
                    break
        if len(feats) < min_predictors:
            continue
        affine = _fit_eval(scout, heldout, target, feats, multiplicative=False)
        multip = _fit_eval(scout, heldout, target, feats, multiplicative=True) if len(feats) <= 3 else None
        best = affine
        construction = "affine"
        if multip is not None and affine is not None and multip[0] > affine[0] + 0.001:
            best, construction = multip, "low_order_multiplicative"
        if best is None or not np.isfinite(best[0]):
            continue
        heldout_r2 = float(best[0])
        residual_ratio = max(0.0, 1.0 - heldout_r2)
        margin = heldout_r2 - (best_single if np.isfinite(best_single) else 0.0)
        if not (heldout_r2 >= recon_r2_min and residual_ratio <= residual_ratio_max and margin >= margin_min):
            continue
        # Repeated-refit stability: refit on resamples of scout, held-out R² spread.
        rng = np.random.default_rng(9173)
        refit_attempts = 8
        refit_r2 = []
        for _ in range(refit_attempts):
            idx = rng.integers(0, len(scout), len(scout))
            res = _fit_eval(scout.iloc[idx], heldout, target, feats, multiplicative=(construction != "affine"))
            if res is not None and np.isfinite(res[0]):
                refit_r2.append(res[0])
        refit_median = float(np.median(refit_r2)) if refit_r2 else heldout_r2
        if refit_median < refit_stability_min:
            continue
        coefs = {f: round(float(c), 6) for f, c in zip(feats, best[1][1:1 + len(feats)])}
        source_vars = _to_source_vars(feats)
        # A cluster needs >= 2 DISTINCT source variables — a target reconstructed
        # from the dummies of a single categorical is just a lookup/group effect,
        # not a multivariable dependency cluster.
        if len({target, *source_vars}) < min_predictors + 1:
            continue
        out[target] = {
            "target": target,
            "source_variables": source_vars,
            "encoded_features": feats,
            "construction": construction,
            "coefficients": coefs,
            "intercept": round(float(best[1][0]), 6),
            "held_out_r2": round(heldout_r2, 6),
            "residual_variance_ratio": round(residual_ratio, 6),
            "best_single_predictor_r2": round(float(best_single), 6) if np.isfinite(best_single) else None,
            "margin_over_best_single": round(float(margin), 6),
            "refit_median_r2": round(refit_median, 6),
            "refit_r2_spread": round(float(max(refit_r2) - min(refit_r2)), 6) if len(refit_r2) > 1 else 0.0,
            # Diagnostic count of valid repeated-refits (already computed above);
            # surfaced for display only — does not affect detection/classification.
            "valid_refit_count": len(refit_r2),
            "refit_attempts": refit_attempts,
            "n_predictors": len(feats),
        }
    return out
