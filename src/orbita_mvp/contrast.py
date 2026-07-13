"""Deterministic binary/predeclared contrast analysis.

The routines here describe contrasts inside the supplied finite dataset. They
do not infer causality or attach a physical interpretation to a group label.
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd

from orbita_discovery.core import Candidate, Falsification, Verdict


PREDICTOR_INTERPRETATIONS = {
    "auto",
    "numeric",
    "categorical",
    "binary_indicator",
    "predeclared_contrast",
}
CONTRAST_DIRECTIONS = {
    "two_sided",
    "positive_greater_than_reference",
    "positive_less_than_reference",
}
PRIMARY_EFFECTS = {
    "mean_difference",
    "ratio",
    "percentage_change",
    "standardized_effect",
}
CONTRAST_VALIDATION_METHODS = {
    "automatic_conservative",
    "blocked_holdout",
    "paired_permutation_exact",
    "bootstrap_by_block",
}


def validate_predictor_interpretation(value: str | None) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized not in PREDICTOR_INTERPRETATIONS:
        raise ValueError(
            f"predictor_interpretation must be one of {sorted(PREDICTOR_INTERPRETATIONS)}"
        )
    return normalized


def validate_contrast_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not config:
        return None
    result = dict(config)
    for field in ("outcome_column", "contrast_column", "positive_level", "reference_level"):
        if result.get(field) is None or str(result.get(field)).strip() == "":
            raise ValueError(f"predeclared contrast requires {field}")
    result["outcome_column"] = str(result["outcome_column"])
    result["contrast_column"] = str(result["contrast_column"])
    result["positive_level"] = str(result["positive_level"])
    result["reference_level"] = str(result["reference_level"])
    if result["positive_level"] == result["reference_level"]:
        raise ValueError("positive_level and reference_level must differ")
    block = result.get("block_column")
    result["block_column"] = str(block) if block not in (None, "") else None
    direction = str(result.get("direction") or "two_sided")
    if direction not in CONTRAST_DIRECTIONS:
        raise ValueError(f"contrast direction must be one of {sorted(CONTRAST_DIRECTIONS)}")
    result["direction"] = direction
    primary = str(result.get("primary_effect") or "mean_difference")
    if primary not in PRIMARY_EFFECTS:
        raise ValueError(f"primary_effect must be one of {sorted(PRIMARY_EFFECTS)}")
    result["primary_effect"] = primary
    method = str(result.get("validation_method") or "automatic_conservative")
    if method not in CONTRAST_VALIDATION_METHODS:
        raise ValueError(
            f"contrast validation_method must be one of {sorted(CONTRAST_VALIDATION_METHODS)}"
        )
    result["validation_method"] = method
    return result


def _level_masks(series: pd.Series, positive_level: str, reference_level: str):
    labels = series.astype(str)
    positive = labels == str(positive_level)
    if reference_level == "__rest__":
        reference = ~positive & series.notna()
    else:
        reference = labels == str(reference_level)
    return positive, reference


def encode_contrast(
    dataframe: pd.DataFrame,
    *,
    outcome_column: str,
    contrast_column: str,
    positive_level: str,
    reference_level: str,
    block_column: str | None = None,
) -> pd.DataFrame:
    required = [outcome_column, contrast_column]
    if block_column:
        required.append(block_column)
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"contrast columns not found: {', '.join(missing)}")
    positive, reference = _level_masks(
        dataframe[contrast_column], positive_level, reference_level
    )
    selected = dataframe.loc[positive | reference, required].copy()
    selected["__y"] = pd.to_numeric(selected[outcome_column], errors="coerce")
    selected["__x"] = positive.loc[selected.index].astype(int)
    selected["__level"] = np.where(selected["__x"] == 1, "positive", "reference")
    selected["__source_index"] = selected.index
    if block_column:
        selected["__block"] = selected[block_column].astype(str)
    selected = selected.dropna(subset=["__y"])
    return selected.reset_index(drop=True)


def _paired_rows(encoded: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    if "__block" not in encoded.columns:
        return pd.DataFrame(), 0, 0
    all_blocks = int(encoded["__block"].nunique())
    rows: list[dict[str, Any]] = []
    for block, group in encoded.groupby("__block", sort=True):
        positive = group.loc[group["__x"] == 1, "__y"]
        reference = group.loc[group["__x"] == 0, "__y"]
        if positive.empty or reference.empty:
            continue
        pos_mean = float(positive.mean())
        ref_mean = float(reference.mean())
        rows.append(
            {
                "block": str(block),
                "positive": pos_mean,
                "reference": ref_mean,
                "delta": pos_mean - ref_mean,
            }
        )
    pairs = pd.DataFrame(rows)
    return pairs, all_blocks, all_blocks - len(pairs)


def _bootstrap_ci(values: np.ndarray, *, seed: int, draws: int = 600) -> list[float] | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = [float(np.mean(values[rng.integers(0, len(values), len(values))])) for _ in range(draws)]
    return [round(float(np.percentile(means, 2.5)), 6), round(float(np.percentile(means, 97.5)), 6)]


def _paired_sign_permutation_p(deltas: np.ndarray, *, seed: int) -> float | None:
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    n = len(deltas)
    if n == 0:
        return None
    observed = abs(float(np.mean(deltas)))
    if n <= 18:
        extreme = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            total += 1
            if abs(float(np.mean(deltas * np.asarray(signs)))) >= observed - 1e-12:
                extreme += 1
        return round(extreme / total, 8)
    rng = np.random.default_rng(seed)
    draws = 20_000
    extreme = 0
    for _ in range(draws):
        signs = rng.choice((-1.0, 1.0), size=n)
        extreme += abs(float(np.mean(deltas * signs))) >= observed - 1e-12
    return round((extreme + 1) / (draws + 1), 8)


def _direction_consistency(deltas: np.ndarray, direction: str) -> tuple[int, int, str]:
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    if not len(deltas):
        return 0, 0, "unavailable"
    if direction == "positive_greater_than_reference":
        return int(np.sum(deltas > 0)), len(deltas), "positive > reference"
    if direction == "positive_less_than_reference":
        return int(np.sum(deltas < 0)), len(deltas), "positive < reference"
    expected_sign = 1 if float(np.mean(deltas)) >= 0 else -1
    count = int(np.sum(deltas > 0)) if expected_sign > 0 else int(np.sum(deltas < 0))
    label = "observed positive direction" if expected_sign > 0 else "observed negative direction"
    return count, len(deltas), label


def analyze_predeclared_contrast(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    *,
    seed: int = 92617,
) -> dict[str, Any]:
    cfg = validate_contrast_config(config)
    assert cfg is not None
    encoded = encode_contrast(
        dataframe,
        outcome_column=cfg["outcome_column"],
        contrast_column=cfg["contrast_column"],
        positive_level=cfg["positive_level"],
        reference_level=cfg["reference_level"],
        block_column=cfg.get("block_column"),
    )
    positive = encoded.loc[encoded["__x"] == 1, "__y"].to_numpy(float)
    reference = encoded.loc[encoded["__x"] == 0, "__y"].to_numpy(float)
    if not len(positive) or not len(reference):
        return {
            "analysis_kind": "predeclared_contrast",
            "validation_status": "not_evaluable",
            "reason": "Both the positive and reference levels require numeric outcomes.",
            "provenance": cfg,
        }

    positive_mean = float(np.mean(positive))
    reference_mean = float(np.mean(reference))
    difference = positive_mean - reference_mean
    ratio = positive_mean / reference_mean if abs(reference_mean) > 1e-12 else None
    percentage = 100.0 * difference / reference_mean if abs(reference_mean) > 1e-12 else None
    pooled_denominator = max(len(positive) + len(reference) - 2, 1)
    pooled_variance = (
        max(len(positive) - 1, 0) * float(np.var(positive, ddof=1) if len(positive) > 1 else 0.0)
        + max(len(reference) - 1, 0) * float(np.var(reference, ddof=1) if len(reference) > 1 else 0.0)
    ) / pooled_denominator
    standardized = difference / math.sqrt(pooled_variance) if pooled_variance > 1e-15 else None

    X = np.column_stack([np.ones(len(encoded)), encoded["__x"].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, encoded["__y"].to_numpy(float), rcond=None)
    predicted = X @ beta
    total_ss = float(np.sum((encoded["__y"] - encoded["__y"].mean()) ** 2))
    indicator_r2 = (
        1.0 - float(np.sum((encoded["__y"].to_numpy(float) - predicted) ** 2)) / total_ss
        if total_ss > 1e-15
        else None
    )

    pairs, total_blocks, dropped_blocks = _paired_rows(encoded)
    matched = bool(cfg.get("block_column"))
    if matched:
        deltas = pairs["delta"].to_numpy(float) if len(pairs) else np.asarray([], dtype=float)
        ci = _bootstrap_ci(deltas, seed=seed)
        paired_sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
        matched_effect = float(np.mean(deltas)) / paired_sd if paired_sd > 1e-15 else None
        consistency, consistency_total, consistency_label = _direction_consistency(
            deltas, cfg["direction"]
        )
        permutation_p = _paired_sign_permutation_p(deltas, seed=seed)
    else:
        rng = np.random.default_rng(seed)
        boot_differences = []
        for _ in range(600):
            p = positive[rng.integers(0, len(positive), len(positive))]
            r = reference[rng.integers(0, len(reference), len(reference))]
            boot_differences.append(float(np.mean(p) - np.mean(r)))
        ci = [
            round(float(np.percentile(boot_differences, 2.5)), 6),
            round(float(np.percentile(boot_differences, 97.5)), 6),
        ]
        matched_effect = None
        consistency, consistency_total, consistency_label = 0, 0, "not matched"
        permutation_p = None

    ci_excludes_zero = bool(ci and (ci[1] < 0 or ci[0] > 0))
    observed_direction_matches = True
    if cfg["direction"] == "positive_greater_than_reference":
        observed_direction_matches = difference > 0
    elif cfg["direction"] == "positive_less_than_reference":
        observed_direction_matches = difference < 0
    if matched and len(pairs) < 2:
        validation_status = "not_evaluable"
    elif not observed_direction_matches:
        validation_status = "direction_contradicted"
    elif matched and permutation_p is not None and permutation_p <= 0.05 and ci_excludes_zero:
        validation_status = "validated_in_dataset"
    elif not matched and ci_excludes_zero:
        validation_status = "validated_in_dataset"
    else:
        validation_status = "not_supported"

    if matched:
        validation_score = {
            "metric": "paired_sign_permutation_p_value",
            "value": permutation_p,
        }
    else:
        bootstrap_direction_stability = float(np.mean(
            [(value > 0) == (difference > 0) for value in boot_differences]
        )) if boot_differences else None
        validation_score = {
            "metric": "bootstrap_direction_stability",
            "value": round(bootstrap_direction_stability, 6)
            if bootstrap_direction_stability is not None else None,
        }

    cautions: list[str] = []
    if min(len(positive), len(reference)) / max(len(positive), len(reference)) < 0.2:
        cautions.append(
            "The contrast groups are severely imbalanced; effect and interval estimates may be unstable."
        )
    reachable_column = "reachable_state_count"
    if reachable_column in dataframe.columns:
        positive_mask, reference_mask = _level_masks(
            dataframe[cfg["contrast_column"]], cfg["positive_level"], cfg["reference_level"]
        )
        pos_reachable = pd.to_numeric(
            dataframe.loc[positive_mask, reachable_column], errors="coerce"
        ).dropna()
        ref_reachable = pd.to_numeric(
            dataframe.loc[reference_mask, reachable_column], errors="coerce"
        ).dropna()
        if len(pos_reachable) and len(ref_reachable) and not math.isclose(
            float(pos_reachable.mean()), float(ref_reachable.mean()), rel_tol=1e-9, abs_tol=1e-12
        ):
            cautions.append(
                "Reachable-state count differs between groups. Compare dimension-normalized outcomes "
                "or size-matched controls before interpreting the contrast as structural."
            )

    return {
        "analysis_kind": "predeclared_contrast",
        "simulation_finding": "A contrast exists in the supplied generated dataset.",
        "validation_status": validation_status,
        "validation_score": validation_score,
        "observed_direction_matches_hypothesis": observed_direction_matches,
        "outcome_column": cfg["outcome_column"],
        "contrast_column": cfg["contrast_column"],
        "positive_level": cfg["positive_level"],
        "reference_level": cfg["reference_level"],
        "group_counts": {
            "positive": int(len(positive)),
            "reference": int(len(reference)),
        },
        "group_means": {
            "positive": round(positive_mean, 6),
            "reference": round(reference_mean, 6),
        },
        "mean_difference": round(difference, 6),
        "ratio": round(ratio, 6) if ratio is not None and math.isfinite(ratio) else None,
        "percentage_change": (
            round(percentage, 6) if percentage is not None and math.isfinite(percentage) else None
        ),
        "standardized_effect": round(standardized, 6) if standardized is not None else None,
        "matched_standardized_effect": round(matched_effect, 6) if matched_effect is not None else None,
        "confidence_interval_95": ci,
        "indicator_regression": {
            "intercept": round(float(beta[0]), 6),
            "coefficient": round(float(beta[1]), 6),
            "coefficient_equals_mean_difference": math.isclose(float(beta[1]), difference, abs_tol=1e-9),
            "r2": round(float(indicator_r2), 6) if indicator_r2 is not None else None,
        },
        "matched_pairs": {
            "block_column": cfg.get("block_column"),
            "total_blocks": total_blocks,
            "complete_pairs": int(len(pairs)),
            "dropped_incomplete_pairs": int(dropped_blocks),
            "direction_consistency_count": consistency,
            "direction_consistency_total": consistency_total,
            "direction": consistency_label,
            "mean_delta": round(float(pairs["delta"].mean()), 6) if len(pairs) else None,
            "paired_permutation_p_value": permutation_p,
        },
        "primary_effect": cfg["primary_effect"],
        "validation_method": cfg["validation_method"],
        "interpretation_cautions": cautions,
        "interpretation_scope": (
            "Simulation contrast only. This result does not establish physical causality, novelty, "
            "or exceptional dynamics relative to size-matched systems."
        ),
        "provenance": {
            **cfg,
            "rows_used": int(len(encoded)),
            "bootstrap_seed": seed,
            "bootstrap_draws": 600,
            "matched_pairs_only": matched,
        },
    }


def contrast_config_from_candidate(candidate: Candidate) -> dict[str, Any]:
    payload = candidate.payload
    return {
        "outcome_column": payload.get("outcome"),
        "contrast_column": payload.get("contrast_column") or payload.get("group"),
        "positive_level": payload.get("positive_level"),
        "reference_level": payload.get("reference_level"),
        "block_column": payload.get("block_column"),
        "direction": payload.get("contrast_direction") or "two_sided",
        "primary_effect": payload.get("primary_effect") or "mean_difference",
        "validation_method": payload.get("validation_method") or "automatic_conservative",
    }


class PredeclaredContrastJudge:
    """Review-only judge for a single predeclared finite-dataset contrast."""

    name = "predeclared_contrast"

    def judge(self, candidate: Candidate, evidence: Any, domain: Any) -> Verdict:
        summary = analyze_predeclared_contrast(
            domain.df,
            contrast_config_from_candidate(candidate),
        )
        status = "provisional" if summary.get("validation_status") == "validated_in_dataset" else "unknown"
        score = (summary.get("indicator_regression") or {}).get("r2")
        return Verdict(
            status,
            float(score) if isinstance(score, (int, float)) and math.isfinite(float(score)) else 0.0,
            {
                "analysis_kind": "predeclared_contrast",
                "validation_status": summary.get("validation_status"),
                "validation_score": summary.get("validation_score"),
                "review_required": True,
                "uses_full_predeclared_dataset": True,
            },
        )


class PredeclaredContrastValidationFalsifier:
    """Records the declared exact/bootstrap result without a predictive R2 gate."""

    name = "predeclared_contrast_validation"

    def attempt(self, candidate: Candidate, evidence: Any, domain: Any) -> Falsification:
        summary = analyze_predeclared_contrast(
            domain.df,
            contrast_config_from_candidate(candidate),
        )
        validation = summary.get("validation_score") or {}
        value = validation.get("value")
        metric = float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else 0.0
        return Falsification(
            self.name,
            killed=summary.get("validation_status") == "direction_contradicted",
            metric=metric,
            detail={
                "validation_status": summary.get("validation_status"),
                "validation_score": validation,
                "confidence_interval_95": summary.get("confidence_interval_95"),
                "matched_pairs": summary.get("matched_pairs"),
                "review_required": True,
                "n": sum((summary.get("group_counts") or {}).values()),
            },
        )
