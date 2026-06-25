"""Post-hoc composite candidate builder — composition_v1.

Composition strategy v1 (composition_v1)
-----------------------------------------
After individual pairwise survivors are known, this module proposes composite
linear models that combine multiple predictors for the same outcome.

**Known limitation (composition_v1)**: only predictors that passed univariate
pairwise screening (|Pearson r| ≥ 0.2 on the scout partition) are eligible.
A predictor with weak marginal correlation but meaningful *conditional* value
(e.g. a suppressor variable) is never proposed because it fails the pairwise
threshold and therefore does not appear in the survivor set passed here.

Planned extension (composition_v2):
  1. Fit the best individual/composite on the selection partition.
  2. Compute residuals on the selection folds.
  3. Screen all remaining (non-survivor) predictors against those residuals.
  4. Propose additions where the incremental R² on residuals exceeds a threshold.
  5. Test each addition with ImprovementFalsifier + AblationFalsifier.
  Categorical predictors and interaction terms should be eligible through the
  same falsification process.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]


def build_composite_candidates(
    survivors: list[dict[str, Any]],
    *,
    min_predictors: int = 2,
    max_predictors: int = 10,
    metric_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Return composite_linear candidates for any outcome with ≥ min_predictors survivors.

    Each composite groups all ``linear_association`` survivors sharing the same
    outcome column.  The composite id is deterministic from its sorted predictor
    list and outcome.  Parent candidate ids are recorded for provenance.

    Parameters
    ----------
    metric_scores:
        Optional mapping of ``candidate_id → metric_score`` computed by the
        service layer using ``domain.score_metric()`` after phase 1.  When
        provided, ``scout_metric["best_individual_metric_score"]`` is set so
        ``ImprovementFalsifier`` can compare in the correct metric units.
        If absent, the fallback ``best_individual_score`` (R² verdict) is used.
    """
    by_outcome: dict[str, list[dict[str, Any]]] = {}
    for f in survivors:
        pay = f["candidate"]["payload"]
        if pay.get("kind") != "linear_association":
            continue
        outcome = pay.get("outcome")
        predictor = pay.get("predictor")
        if not outcome or not predictor:
            continue
        by_outcome.setdefault(outcome, []).append(f)

    composites: list[dict[str, Any]] = []
    for outcome, group in by_outcome.items():
        if len(group) < min_predictors:
            continue
        predictors = sorted({f["candidate"]["payload"]["predictor"] for f in group})
        if len(predictors) > max_predictors:
            predictors = predictors[:max_predictors]
        raw = f"composite_linear|{'|'.join(predictors)}|{outcome}"
        cid = f"composite:{_slug(outcome)}:{hashlib.sha256(raw.encode()).hexdigest()[:8]}"
        predictor_str = " + ".join(predictors)

        parent_scores = {
            f["candidate"]["payload"]["predictor"]: round(f["verdict"]["score"], 6)
            for f in group
        }
        best_r2 = round(max(f["verdict"]["score"] for f in group), 6)

        scout_metric: dict[str, Any] = {
            "parent_scores": parent_scores,
            "best_individual_score": best_r2,
        }
        if metric_scores is not None:
            parent_metric = {
                f["candidate"]["payload"]["predictor"]: metric_scores.get(f["candidate"]["id"])
                for f in group
                if f["candidate"]["id"] in metric_scores
            }
            valid_scores = [v for v in parent_metric.values() if v is not None]
            if valid_scores:
                scout_metric["parent_metric_scores"] = parent_metric
                scout_metric["best_individual_metric_score"] = round(
                    min(valid_scores) if True else max(valid_scores),
                    6,
                )
                # Overwrite with correct direction in service.py (passed as argument)
                # The service knows the metric direction; we store the raw values here.

        composites.append({
            "id": cid,
            "statement": (
                f"{outcome} can be predicted by a composite of [{predictor_str}] "
                f"(linear model, {len(predictors)} predictors)."
            ),
            "kind": "composite_linear",
            "composition_strategy": "composition_v1",
            "predictors": predictors,
            "outcome": outcome,
            "parent_candidate_ids": [f["candidate"]["id"] for f in group],
            "scout_metric": scout_metric,
            "parents": [f["candidate"]["id"] for f in group],
        })
    return composites
