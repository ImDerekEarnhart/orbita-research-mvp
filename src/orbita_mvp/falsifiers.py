"""Extra falsifiers specific to the orbita_mvp pipeline."""
from __future__ import annotations

from orbita_discovery.core import Candidate, Falsification
from orbita_discovery.falsifiers import FittableDomain

from .metrics import NULL_SCORE, higher_is_better, is_improvement


class AblationFalsifier:
    """Kill a composite_linear candidate if any predictor contributes < min_contribution.

    For non-composite candidates, this falsifier always passes (no ablation needed).

    Partition
    ---------
    Uses ``evidence["confirmation"]`` (the selection partition) — never the scout
    partition.  This prevents data-partition violations and matches the partition
    used by ImprovementFalsifier and HeldOutFalsifier.

    Metric
    ------
    Contributions are computed in ``evaluation_metric`` units (metric-aware):

    * Higher-is-better metrics (R²):
        contribution = full_score − reduced_score  (positive when predictor is useful)
    * Lower-is-better metrics (RMSLE, RMSE, MAE):
        contribution = reduced_score − full_score  (positive when predictor is useful;
        removing a useful predictor *increases* the error)

    A predictor with contribution < min_contribution does not earn its place.
    The R² ablation_contributions in the model dict (fitted during refit()) are
    retained as a report-only diagnostic but are NOT used here.
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

        # Use selection partition — not scout (fixes data-partition violation)
        selection = evidence["confirmation"]

        metric: str = getattr(domain, "evaluation_metric", "r2")
        hib = higher_is_better(metric)
        score_fn = getattr(domain, "score_metric", None) or domain.score

        full_model = domain.refit(c, selection)
        if not full_model.get("valid"):
            return Falsification(self.name, killed=True, metric=0.0,
                                 detail={"error": "composite refit produced invalid model",
                                         "partition": "selection"})

        full_score = score_fn(c, full_model, selection)
        predictors = list(c.payload.get("predictors", []))

        contributions: dict[str, float] = {}
        per_predictor: list[dict] = []
        for p in predictors:
            reduced_preds = [x for x in predictors if x != p]
            if not reduced_preds:
                contributions[p] = float("inf")
                per_predictor.append({"predictor": p, "contribution": float("inf"),
                                      "full_score": full_score, "reduced_score": None})
                continue
            reduced_payload = {**c.payload, "predictors": reduced_preds,
                               "id": c.id + f"_drop_{p}"}
            reduced_c = Candidate(id=reduced_payload["id"], statement=c.statement,
                                  payload=reduced_payload)
            reduced_model = domain.refit(reduced_c, selection)
            if not reduced_model.get("valid"):
                contributions[p] = 0.0
                per_predictor.append({"predictor": p, "contribution": 0.0,
                                      "full_score": full_score, "reduced_score": None,
                                      "error": "reduced refit invalid"})
                continue
            reduced_score = score_fn(reduced_c, reduced_model, selection)
            if hib:
                contrib = round(full_score - reduced_score, 6)
            else:
                contrib = round(reduced_score - full_score, 6)
            contributions[p] = contrib
            per_predictor.append({"predictor": p, "contribution": contrib,
                                   "full_score": full_score, "reduced_score": reduced_score,
                                   "pass": contrib >= self.min_contribution})

        useless = {p: v for p, v in contributions.items()
                   if v < self.min_contribution and v != float("inf")}
        min_contrib = min((v for v in contributions.values() if v != float("inf")),
                          default=0.0)
        killed = bool(useless)
        return Falsification(
            self.name,
            killed=killed,
            metric=min_contrib,
            detail={
                "evaluation_metric": metric,
                "higher_is_better": hib,
                "full_score": full_score,
                "contributions": contributions,
                "per_predictor": per_predictor,
                "min_contribution_threshold": self.min_contribution,
                "useless_predictors": useless,
                "partition": "selection",
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
