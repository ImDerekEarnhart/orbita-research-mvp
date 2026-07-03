"""Generic falsifiers for any fittable domain."""
from __future__ import annotations

import math
from statistics import median, pstdev, quantiles
from typing import Any, Protocol, runtime_checkable

from .core import Candidate, Domain, Falsification

# Canonical name of the fixed-model test-resampling check. Historically this
# check was named ``cross_seed`` (and still is on older ledgers/plans/APIs); it
# was misleading, because it does NOT refit the model per seed — it re-scores a
# single scout-fitted model against bootstrap resamples of the validation rows.
# The honest name is ``validation_resample``; ``cross_seed`` is retained as a
# recognised alias everywhere that reads the check (see
# ``orbita_mvp.semantics.RESAMPLE_CHECK_NAMES``).
VALIDATION_RESAMPLE_NAME = "validation_resample"
CROSS_SEED_ALIAS = "cross_seed"


@runtime_checkable
class FittableDomain(Protocol):
    def splits(self, evidence: Any, seed: int) -> tuple[Any, Any]: ...
    def refit(self, c: Candidate, train: Any) -> Any: ...
    def score(self, c: Candidate, model: Any, test: Any) -> float: ...
    def baseline_score(self, test: Any) -> float: ...


class BaselineFalsifier:
    name = "baseline"

    def __init__(self, margin: float = 0.05):
        self.margin = margin

    def attempt(self, c: Candidate, evidence: Any, domain: Domain) -> Falsification:
        if not isinstance(domain, FittableDomain):
            return Falsification(self.name, False, detail={"skipped": "domain not fittable"})
        train, test = domain.splits(evidence, seed=0)
        model = domain.refit(c, train)
        score = domain.score(c, model, test)
        baseline = domain.baseline_score(test)
        delta = score - baseline
        return Falsification(
            self.name,
            killed=delta < self.margin,
            metric=delta,
            detail={"score": round(score, 6), "baseline": round(baseline, 6), "margin": self.margin, "n": len(test)},
        )


class HeldOutFalsifier:
    name = "held_out"

    def __init__(self, min_score: float = 0.3):
        self.min_score = min_score

    def attempt(self, c: Candidate, evidence: Any, domain: Domain) -> Falsification:
        if not isinstance(domain, FittableDomain):
            return Falsification(self.name, False, detail={"skipped": "domain not fittable"})
        train, test = domain.splits(evidence, seed=1)
        model = domain.refit(c, train)
        score = domain.score(c, model, test)
        return Falsification(
            self.name,
            killed=score < self.min_score,
            metric=score,
            detail={"score": round(score, 6), "minimum": self.min_score, "n": len(test)},
        )


class CrossSeedFalsifier:
    """Fixed-model sensitivity to the validation sample (``validation_resample``).

    IMPORTANT: this check does NOT refit the model per seed. ``domain.splits``
    returns the *same* training partition (the locked scout rows) for every
    seed and only bootstrap-resamples the validation rows, so this measures how
    stable one fitted model's score is under resampling of the test set — not
    reproducibility of the fit itself. For genuine per-seed refitting use
    :class:`RepeatedRefitValidator`. Emitted under the honest name
    ``validation_resample``; ``cross_seed`` remains a recognised alias.
    """

    name = VALIDATION_RESAMPLE_NAME
    aliases = (CROSS_SEED_ALIAS,)

    def __init__(self, seeds: int = 7, min_median: float = 0.3, max_spread: float | None = None):
        self.seeds = seeds
        self.min_median = min_median
        self.max_spread = max_spread

    def attempt(self, c: Candidate, evidence: Any, domain: Domain) -> Falsification:
        if not isinstance(domain, FittableDomain):
            return Falsification(self.name, False, detail={"skipped": "domain not fittable"})
        scores: list[float] = []
        test_n = 0
        for seed in range(2, 2 + self.seeds):
            train, test = domain.splits(evidence, seed=seed)
            model = domain.refit(c, train)
            scores.append(domain.score(c, model, test))
            test_n = len(test)
        med = median(scores)
        spread = max(scores) - min(scores)
        killed = med < self.min_median
        if self.max_spread is not None:
            killed = killed or spread > self.max_spread
        return Falsification(
            self.name,
            killed=killed,
            metric=med,
            detail={
                "check_kind": "fixed_model_validation_resample",
                "median": round(med, 6),
                "spread": round(spread, 6),
                "seeds": self.seeds,
                "min_median": self.min_median,
                "max_spread": self.max_spread,
                "n": test_n,
            },
        )


def _model_coefficients(model: Any) -> dict[str, float]:
    """Best-effort extraction of a fitted model's coefficients for stability stats."""
    if not isinstance(model, dict):
        return {}
    if "slope" in model and isinstance(model["slope"], (int, float)):
        return {"slope": float(model["slope"])}
    coeffs = model.get("coefficients")
    if isinstance(coeffs, dict):
        return {str(k): float(v) for k, v in coeffs.items() if isinstance(v, (int, float))}
    means = model.get("means")
    if isinstance(means, dict):
        return {str(k): float(v) for k, v in means.items() if isinstance(v, (int, float))}
    return {}


class RepeatedRefitValidator:
    """Genuine repeated independent refitting (``repeated_refit``).

    Unlike ``validation_resample`` (which holds one scout-fitted model fixed),
    this validator, for every seed, draws a *fresh* train/validation partition
    from the modelling pool, **refits** the candidate model on the new training
    rows, and evaluates it on the untouched validation rows. It reports the
    distribution of scores, coefficients, and direction across independent fits
    — the correct evidence for a claim about model *reproducibility*.

    Diagnostic-only by default: ``min_valid_median`` is ``None`` so the check
    never kills a candidate and therefore never changes existing survivor /
    refuted classification. Persisted independently of ``validation_resample``.
    """

    name = "repeated_refit"

    def __init__(self, seeds: int = 12, min_valid_median: float | None = None):
        self.seeds = seeds
        self.min_valid_median = min_valid_median

    def attempt(self, c: Candidate, evidence: Any, domain: Domain) -> Falsification:
        if not isinstance(domain, FittableDomain):
            return Falsification(self.name, False, detail={"skipped": "domain not fittable"})
        split = getattr(domain, "repeated_refit_split", None)
        if split is None:
            return Falsification(self.name, False, detail={"skipped": "domain has no repeated_refit_split"})

        scores: list[float] = []
        coeff_fits: list[dict[str, float]] = []
        train_ns: list[int] = []
        val_ns: list[int] = []
        fit_failures = 0
        for seed in range(self.seeds):
            train, val = split(seed)
            model = domain.refit(c, train)
            if not (isinstance(model, dict) and model.get("valid")):
                fit_failures += 1
                continue
            score = domain.score(c, model, val)
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                fit_failures += 1
                continue
            scores.append(float(score))
            coeff_fits.append(_model_coefficients(model))
            train_ns.append(len(train))
            val_ns.append(len(val))

        valid = len(scores)
        detail: dict[str, Any] = {
            "check_kind": "repeated_independent_refit",
            "seeds": self.seeds,
            "valid_fits": valid,
            "fit_failures": fit_failures,
            "valid_fit_fraction": round(valid / self.seeds, 4) if self.seeds else 0.0,
        }
        med = 0.0
        if valid:
            med = float(median(scores))
            lower_q = min(scores)
            if valid >= 4:
                # 10th-percentile-ish lower tail via deciles.
                lower_q = float(quantiles(scores, n=10)[0])
            detail.update({
                "median": round(med, 6),
                "lower_quantile": round(float(lower_q), 6),
                "score_min": round(min(scores), 6),
                "score_max": round(max(scores), 6),
                "score_variance": round(pstdev(scores) ** 2, 6) if valid > 1 else 0.0,
                "train_n_median": int(median(train_ns)),
                "val_n_median": int(median(val_ns)),
            })
            detail["coefficient_stability"] = _coefficient_stability(coeff_fits)
            detail["direction_stability"] = _overall_direction_stability(detail["coefficient_stability"])

        killed = False
        if self.min_valid_median is not None and valid and med < self.min_valid_median:
            killed = True
        return Falsification(self.name, killed=killed, metric=round(med, 6), detail=detail)


def _coefficient_stability(coeff_fits: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Per-coefficient mean, variance, and sign stability across refits."""
    keys: set[str] = set()
    for fit in coeff_fits:
        keys.update(fit.keys())
    out: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        vals = [fit[key] for fit in coeff_fits if key in fit]
        if not vals:
            continue
        med = float(median(vals))
        med_sign = 1.0 if med > 0 else -1.0 if med < 0 else 0.0
        if med_sign == 0.0:
            sign_stability = 1.0
        else:
            agree = sum(1 for v in vals if (v > 0) == (med_sign > 0) and v != 0)
            sign_stability = round(agree / len(vals), 4)
        out[key] = {
            "mean": round(float(sum(vals) / len(vals)), 6),
            "median": round(med, 6),
            "variance": round(pstdev(vals) ** 2, 6) if len(vals) > 1 else 0.0,
            "sign_stability": sign_stability,
            "n_fits": len(vals),
        }
    return out


def _overall_direction_stability(coefficient_stability: dict[str, dict[str, float]]) -> float | None:
    """Weakest per-coefficient sign stability = the claim's overall direction stability."""
    stats = [c.get("sign_stability") for c in coefficient_stability.values() if c.get("sign_stability") is not None]
    return round(min(stats), 4) if stats else None


DEFAULT_FALSIFIERS = [BaselineFalsifier(), HeldOutFalsifier(), CrossSeedFalsifier()]
