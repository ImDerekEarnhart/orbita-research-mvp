"""Extra falsifiers specific to the orbita_mvp pipeline."""
from __future__ import annotations

from orbita_discovery.core import Candidate, Falsification
from orbita_discovery.falsifiers import FittableDomain


class AblationFalsifier:
    """Kill a composite_linear candidate if any predictor contributes < min_contribution R².

    For non-composite candidates, this falsifier always passes (no ablation needed).
    For composite candidates, the model's ``ablation_contributions`` dict (computed in
    ``UploadedTableDomain.refit``) is used directly, avoiding a second OLS pass.
    """

    name = "ablation"

    def __init__(self, min_contribution: float = 0.01):
        self.min_contribution = min_contribution

    def attempt(self, c: Candidate, evidence: object, domain: object) -> Falsification:
        if c.payload.get("kind") != "composite_linear":
            return Falsification(self.name, killed=False, metric=1.0,
                                 detail={"skipped": "not a composite candidate"})
        if not isinstance(domain, FittableDomain):
            return Falsification(self.name, killed=False,
                                 detail={"skipped": "domain not fittable"})
        train, _ = domain.splits(evidence, seed=1)
        model = domain.refit(c, train)
        if not model.get("valid"):
            return Falsification(self.name, killed=True, metric=0.0,
                                 detail={"error": "composite refit produced invalid model"})

        contributions = model.get("ablation_contributions", {})
        useless = {p: v for p, v in contributions.items() if v < self.min_contribution}
        min_contrib = min(contributions.values()) if contributions else 0.0
        killed = bool(useless)
        return Falsification(
            self.name,
            killed=killed,
            metric=min_contrib,
            detail={
                "contributions": contributions,
                "min_contribution_threshold": self.min_contribution,
                "useless_predictors": useless,
            },
        )


class ImprovementFalsifier:
    """Kill a composite if it doesn't improve on the best individual survivor score.

    ``best_individual_score`` is read from the candidate's ``scout_metric`` dict,
    which is set by ``build_composite_candidates`` at composition time.
    """

    name = "improvement"

    def __init__(self, min_improvement: float = 0.01):
        self.min_improvement = min_improvement

    def attempt(self, c: Candidate, evidence: object, domain: object) -> Falsification:
        if c.payload.get("kind") != "composite_linear":
            return Falsification(self.name, killed=False, metric=1.0,
                                 detail={"skipped": "not a composite candidate"})
        if not isinstance(domain, FittableDomain):
            return Falsification(self.name, killed=False,
                                 detail={"skipped": "domain not fittable"})
        train, test = domain.splits(evidence, seed=1)
        model = domain.refit(c, train)
        if not model.get("valid"):
            return Falsification(self.name, killed=True, metric=0.0,
                                 detail={"error": "refit failed"})
        composite_score = domain.score(c, model, test)
        best_individual = c.payload.get("scout_metric", {}).get("best_individual_score", 0.0)
        improvement = composite_score - best_individual
        killed = improvement < self.min_improvement
        return Falsification(
            self.name,
            killed=killed,
            metric=improvement,
            detail={
                "composite_score": round(composite_score, 6),
                "best_individual_score": round(best_individual, 6),
                "improvement": round(improvement, 6),
                "min_improvement": self.min_improvement,
            },
        )
