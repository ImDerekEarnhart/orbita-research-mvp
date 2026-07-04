"""Structural / transform artifact detection for uploaded tables.

Ordinary pairwise candidate generation treats every column pair as a potential
scientific hypothesis. But many pairs are *tautological*: a column versus its
own log transform, a duplicated column, a unit conversion (kg vs g), or a field
that is an exact algebraic function of others. These pass falsification trivially
(or fail it as if they were real hypotheses) and pollute the belief graph.

This module classifies such pairs up front so the pipeline can label them
``artifact``/``structural_relation`` instead of mining them as science.

Detection is numeric, not name-based: a column literally named ``log_mass`` is
only flagged when its values actually match ``log10(mass)`` (or ln/log2), so a
coincidental name never suppresses a genuine finding.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

# Threshold for matching a column to a *transform* of another (e.g. log). The
# stored transform column is usually rounded (3-4 decimals), so this is high but
# not machine-exact.
_PERFECT_R2 = 0.99995
# Threshold for declaring a *duplicate / unit conversion / derived field*. These
# are exact constructions (same quantity, ×constant, a+b, a/b) and fit to floating
# point. A genuine tight physical law (e.g. Kepler's log-period vs log-radius at
# R²≈0.999999) has real residual scatter and stays *below* this ceiling, so it is
# never mistaken for an artifact.
_EXACT_R2 = 1.0 - 1e-9
# Near-copy / target-leakage band: a relationship this tight (but not machine
# exact) means one column carries essentially ALL the information in the other —
# a noisy duplicate, an affine copy, or a near-deterministic transform. A
# genuine strong scientific relationship keeps a real residual (R² well under
# this, e.g. ~0.92 with slope ≠ 1); a leaked/derived column does not. Both a very
# high affine R² AND a very high |correlation| are required so a coincidentally
# tight but structurally different relationship is not mislabelled.
_NEAR_COPY_R2 = 0.999
_NEAR_COPY_CORR = 0.9995
_MIN_POINTS = 5


def _numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[mask], y[mask]
    if len(xs) < _MIN_POINTS or np.ptp(xs) <= 1e-15 or np.ptp(ys) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[mask], pred[mask]
    if len(y) < _MIN_POINTS:
        return float("nan")
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-15:
        return float("nan")
    return 1.0 - float(np.sum((y - pred) ** 2)) / denom


def _affine_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Best affine fit y ≈ a + b·x; return (r2, slope, intercept)."""
    mask = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[mask], y[mask]
    if len(xs) < _MIN_POINTS or np.ptp(xs) <= 1e-15:
        return float("nan"), 0.0, 0.0
    A = np.column_stack([np.ones(len(xs)), xs])
    beta, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = beta[0] + beta[1] * xs
    return _r2(ys, pred), float(beta[1]), float(beta[0])


def _is_log_of(values: np.ndarray, base_values: np.ndarray) -> str | None:
    """Return the log base name if `values` ≈ log(base_values), else None."""
    mask = np.isfinite(values) & np.isfinite(base_values) & (base_values > 0)
    if mask.sum() < _MIN_POINTS:
        return None
    v, b = values[mask], base_values[mask]
    for name, fn in (("log10", np.log10), ("ln", np.log), ("log2", np.log2)):
        with np.errstate(all="ignore"):
            transformed = fn(b)
        r2 = _r2(v, transformed)
        if math.isfinite(r2) and r2 >= _PERFECT_R2:
            # Direct identity (slope≈1, intercept≈0) or a scaled log.
            return name
    return None


def _classify_pair(
    x: np.ndarray,
    y: np.ndarray,
    xname: str,
    yname: str,
    *,
    near_copy_r2: float = _NEAR_COPY_R2,
    near_copy_corr: float = _NEAR_COPY_CORR,
) -> dict[str, Any] | None:
    """Classify a numeric column pair as a structural relation, or None."""
    # 1. Log transform in either direction.
    log_base = _is_log_of(y, x)
    if log_base:
        return {"kind": "log_transform", "detail": f"{yname} ≈ {log_base}({xname})"}
    log_base = _is_log_of(x, y)
    if log_base:
        return {"kind": "log_transform", "detail": f"{xname} ≈ {log_base}({yname})"}

    # 2. Affine-exact (duplicate / unit conversion / mirrored column). Uses the
    # machine-exact ceiling so a tight-but-real law is not flagged as an artifact.
    r2, slope, intercept = _affine_r2(x, y)
    if math.isfinite(r2) and r2 >= _EXACT_R2:
        if abs(intercept) <= 1e-9 and abs(abs(slope) - 1.0) <= 1e-9:
            return {"kind": "mirrored_duplicate", "detail": f"{yname} ≈ {xname}"}
        if abs(intercept) <= 1e-6 * (1.0 + abs(np.nanmean(y))):
            return {"kind": "unit_conversion", "detail": f"{yname} ≈ {slope:.6g}·{xname}"}
        return {"kind": "affine_dependence", "detail": f"{yname} ≈ {slope:.6g}·{xname} + {intercept:.6g}"}

    # 3. Near-copy / target-leakage band: extremely tight AND near-identity.
    # One column is a noisy DUPLICATE of the other — same quantity (slope ≈ 1,
    # intercept ≈ 0) with residual variance near zero. A genuine tight law
    # relates DIFFERENT quantities (slope ≠ 1, meaningful intercept) and is
    # deliberately NOT flagged, however high its R² — near-identity, not merely
    # high correlation, is what separates a leaked copy from real science.
    corr = _pearson(x, y)
    if math.isfinite(r2) and r2 >= near_copy_r2 and math.isfinite(corr) and abs(corr) >= near_copy_corr:
        finite_y = y[np.isfinite(y)]
        std_y = float(np.std(finite_y)) if len(finite_y) else 0.0
        near_identity = (
            abs(abs(slope) - 1.0) <= 0.05
            and (std_y == 0.0 or abs(intercept) <= 0.05 * std_y)
        )
        if near_identity:
            residual_ratio = max(0.0, 1.0 - float(r2))
            return {
                "kind": "near_duplicate_copy",
                "detail": (
                    f"{yname} ≈ {slope:.6g}·{xname} + {intercept:.6g} "
                    f"(R²={r2:.6f}, |r|={abs(corr):.6f}, residual variance ratio={residual_ratio:.2e})"
                ),
                "leakage_risk": "high",
                "similarity_metric": "affine_r2",
                "similarity": round(float(r2), 6),
                "correlation": round(float(corr), 6),
                "residual_variance_ratio": round(residual_ratio, 8),
                "slope": round(float(slope), 6),
                "intercept": round(float(intercept), 6),
                # Direction of derivation is not identifiable from values alone;
                # both columns are reported and the pair is downgraded, never mined.
                "suspected_source_column": sorted([xname, yname])[0],
                "derived_column_candidate": sorted([xname, yname])[1],
                "disposition": "downgraded_to_artifact",
            }
    return None


def _classify_derived(
    df: pd.DataFrame,
    target: str,
    others: list[str],
    *,
    near_derived_r2: float = _NEAR_COPY_R2,
) -> dict[str, Any] | None:
    """Detect ``target ≈ f(a, b)`` for a (near-)deterministic two-column combination.

    Two bands, mirroring the near-copy logic:

    * **Exact** (``R² ≥ _EXACT_R2``): a machine-exact algebraic construction
      (e.g. ``total_mass = m1 + m2``) → ``derived_field``.
    * **Near-exact** (``near_derived_r2 ≤ R² < _EXACT_R2``): a noisy accounting
      identity — the target is an algebraic combination of two columns plus small
      residual noise or output rounding (e.g. ``energy_after ≈ energy_before −
      energy_cost``) → ``near_derived_field``. A genuine empirical law almost
      never reconstructs to an *exact algebraic op of two specific columns* at
      R² ≥ 0.999 with real scatter, so this is a strong, general derived-field
      signal — not tuned to any dataset.

    Returns the tightest match across all column pairs and ops.
    """
    y = _numeric(df, target)
    finite_y = y[np.isfinite(y)]
    std_y = float(np.std(finite_y)) if len(finite_y) else 0.0
    ops = (
        ("sum", lambda a, b: a + b),
        ("difference", lambda a, b: a - b),
        ("product", lambda a, b: a * b),
        ("ratio", lambda a, b: np.divide(a, b, out=np.full_like(a, np.nan), where=np.abs(b) > 1e-12)),
    )
    # Track the best EXACT reconstruction (any coefficients) and, separately, the
    # best near-exact reconstruction that is a UNIT-coefficient identity.
    best_exact: tuple[float, str, str, str] | None = None
    best_identity: tuple[float, str, str, str] | None = None
    for a_name, b_name in combinations(others, 2):
        a, b = _numeric(df, a_name), _numeric(df, b_name)
        for op_name, fn in ops:
            with np.errstate(all="ignore"):
                combo = fn(a, b)
            r2, slope, intercept = _affine_r2(combo, y)
            if not math.isfinite(r2):
                continue
            if r2 >= _EXACT_R2 and (best_exact is None or r2 > best_exact[0]):
                best_exact = (r2, op_name, a_name, b_name)
            # Near-exact band requires a UNIT-COEFFICIENT identity (slope ≈ ±1,
            # intercept ≈ 0): the target literally equals a ± b (an accounting
            # identity) plus small residual noise — NOT a weighted regression like
            # y = 5·x2 + 5·x3, which is a genuine composite to be discovered.
            near_identity = (
                abs(abs(slope) - 1.0) <= 0.05
                and (std_y == 0.0 or abs(intercept) <= 0.05 * std_y)
            )
            if r2 >= near_derived_r2 and near_identity and (best_identity is None or r2 > best_identity[0]):
                best_identity = (r2, op_name, a_name, b_name)

    if best_exact is not None:
        r2, op_name, a_name, b_name = best_exact
        return {
            "kind": "derived_field",
            "detail": f"{target} ≈ {op_name}({a_name}, {b_name})",
            "inputs": [a_name, b_name],
        }
    if best_identity is not None:
        r2, op_name, a_name, b_name = best_identity
        residual_ratio = max(0.0, 1.0 - float(r2))
        return {
            "kind": "near_derived_field",
            "detail": (
                f"{target} ≈ {op_name}({a_name}, {b_name}) "
                f"(near-exact accounting identity, R²={r2:.6f}, residual variance ratio={residual_ratio:.2e})"
            ),
            "inputs": [a_name, b_name],
            "leakage_risk": "high",
            "similarity_metric": "affine_r2",
            "similarity": round(float(r2), 6),
            "residual_variance_ratio": round(residual_ratio, 8),
            "op": op_name,
            "disposition": "downgraded_to_artifact",
        }
    return None


def detect_structural_relations(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    identifier_columns: list[str] | None = None,
    *,
    near_copy_r2: float = _NEAR_COPY_R2,
    near_copy_corr: float = _NEAR_COPY_CORR,
    near_derived_r2: float = _NEAR_COPY_R2,
) -> dict[str, dict[str, Any]]:
    """Return a map from "colA||colB" (sorted) to a structural classification.

    The key uses the sorted column pair so candidate generation can look up any
    ordering. Identifier/helper columns are reported under a single-column key.
    """
    if numeric_columns is None:
        numeric_columns = [
            str(c) for c in df.columns
            if float(pd.to_numeric(df[c], errors="coerce").notna().mean()) >= 0.85
            and int(df[c].nunique(dropna=True)) >= 3
        ]
    identifier_columns = identifier_columns or []
    relations: dict[str, dict[str, Any]] = {}

    cols = {c: _numeric(df, c) for c in numeric_columns}
    for xname, yname in combinations(numeric_columns, 2):
        classification = _classify_pair(
            cols[xname], cols[yname], xname, yname,
            near_copy_r2=near_copy_r2, near_copy_corr=near_copy_corr,
        )
        if classification:
            relations[_pair_key(xname, yname)] = {**classification, "columns": [xname, yname]}

    # Derived fields: a column that is a deterministic function of two others.
    for target in numeric_columns:
        if _single_key(target) in relations:
            continue
        others = [c for c in numeric_columns if c != target]
        derived = _classify_derived(df, target, others, near_derived_r2=near_derived_r2)
        if derived:
            inputs = derived.get("inputs", [])
            for src in inputs:
                # setdefault: never overwrite a more specific pair classification
                # already found by _classify_pair (e.g. a near-copy), which is the
                # correct label for that exact pair.
                relations.setdefault(_pair_key(target, src), {**derived, "columns": [target, src]})

    for ident in identifier_columns:
        relations[_single_key(str(ident))] = {
            "kind": "identifier",
            "detail": f"{ident} is an identifier/helper column",
            "columns": [str(ident)],
        }
    return relations


def _pair_key(a: str, b: str) -> str:
    return "||".join(sorted([a, b]))


def _single_key(a: str) -> str:
    return a


def structural_for(relations: dict[str, dict[str, Any]], a: str, b: str) -> dict[str, Any] | None:
    return relations.get(_pair_key(a, b)) or relations.get(_single_key(a)) or relations.get(_single_key(b))
