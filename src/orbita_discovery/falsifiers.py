"""Generic falsifiers for any fittable domain."""
from __future__ import annotations

from statistics import median
from typing import Any, Protocol, runtime_checkable

from .core import Candidate, Domain, Falsification


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
            detail={"score": round(score, 6), "baseline": round(baseline, 6), "margin": self.margin},
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
            detail={"score": round(score, 6), "minimum": self.min_score},
        )


class CrossSeedFalsifier:
    name = "cross_seed"

    def __init__(self, seeds: int = 7, min_median: float = 0.3, max_spread: float | None = None):
        self.seeds = seeds
        self.min_median = min_median
        self.max_spread = max_spread

    def attempt(self, c: Candidate, evidence: Any, domain: Domain) -> Falsification:
        if not isinstance(domain, FittableDomain):
            return Falsification(self.name, False, detail={"skipped": "domain not fittable"})
        scores: list[float] = []
        for seed in range(2, 2 + self.seeds):
            train, test = domain.splits(evidence, seed=seed)
            model = domain.refit(c, train)
            scores.append(domain.score(c, model, test))
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
                "median": round(med, 6),
                "spread": round(spread, 6),
                "seeds": self.seeds,
                "min_median": self.min_median,
                "max_spread": self.max_spread,
            },
        )


DEFAULT_FALSIFIERS = [BaselineFalsifier(), HeldOutFalsifier(), CrossSeedFalsifier()]
