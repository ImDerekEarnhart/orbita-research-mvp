"""Extra falsifiers specific to the orbita_mvp pipeline."""
from __future__ import annotations

from orbita_discovery.core import Candidate, Falsification
from orbita_discovery.falsifiers import FittableDomain

from .metrics import NULL_SCORE, higher_is_better, is_improvement


class AblationFalsifier:
    """Kill a composite_linear candidate if any predictor contributes < min_contribution R².

    For non-composite candidates, this falsifier always passes (no ablation needed).
    For composite candidates, the model's ``ablation_contributions`` dict (computed in
    ``UploadedTableDomain.refit``) is used directly, avoiding a second OLS pass.

    Ablation contributions are always in R² units (marginal R² each predictor adds),
    regardless of ``evaluation_metric``.  A contribution of < 0.01 means the predictor
    explains less than 1 % of residual variance — it is not earning its place.
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

    Metric-direction aware: for R² (higher-is-better) the composite must exceed
    the best individual score by at least ``min_improvement``.  For error metrics
    (rmsle, rmse, mae — lower-is-better) the composite must be *lower* by at least
    ``min_improvement``.

    Score sources
    -------------
    * ``best_individual_metric_score`` in the candidate's ``scout_metric`` dict —
      set by the service layer using ``domain.score_metric()`` after phase 1.
      Preferred because it is in the same units as the composite's metric score.
    * ``best_individual_score`` — fallback (R² verdict score from phase 1).
    * ``evaluation_metric`` attribute on the domain — used to call
      ``domain.score_metric()`` on the composite and to determine direction.
      Falls back to ``"r2"`` when the domain does not expose this attribute.
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

        metric: str = getattr(domain, "evaluation_metric", "r2")
        train, test = domain.splits(evidence, seed=1)
        model = domain.refit(c, train)
        if not model.get("valid"):
            return Falsification(self.name, killed=True, metric=0.0,
                                 detail={"error": "refit failed"})

        # Score the composite under the configured metric
        score_fn = getattr(domain, "score_metric", None)
        if score_fn is not None:
            composite_score = score_fn(c, model, test)
        else:
            composite_score = domain.score(c, model, test)

        # Prefer metric-specific individual score; fall back to R² verdict
        scout_metric = c.payload.get("scout_metric", {})
        best_individual = float(
            scout_metric.get("best_individual_metric_score")
            if scout_metric.get("best_individual_metric_score") is not None
            else scout_metric.get("best_individual_score", NULL_SCORE.get(metric, 0.0))
        )

        improved = is_improvement(metric, composite_score, best_individual, self.min_improvement)
        killed = not improved

        # Express the gap in a direction-neutral way: positive = improvement
        if higher_is_better(metric):
            gap = composite_score - best_individual
        else:
            gap = best_individual - composite_score

        return Falsification(
            self.name,
            killed=killed,
            metric=gap,
            detail={
                "evaluation_metric": metric,
                "higher_is_better": higher_is_better(metric),
                "composite_score": composite_score,
                "best_individual_score": best_individual,
                "improvement_gap": gap,
                "min_improvement": self.min_improvement,
            },
        )
