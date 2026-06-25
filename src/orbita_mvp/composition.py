"""Post-hoc composite candidate builder — composition_v1 and composition_v1_1.

Composition strategy v1 (composition_v1)
-----------------------------------------
After individual pairwise survivors are known, this module proposes composite
linear models that combine multiple predictors for the same outcome.

**Known limitation (composition_v1)**: only predictors that passed univariate
pairwise screening (|Pearson r| ≥ 0.2 on the scout partition) are eligible.
A predictor with weak marginal correlation but meaningful *conditional* value
(e.g. a suppressor variable) is never proposed because it fails the pairwise
threshold and therefore does not appear in the survivor set passed here.

Composition strategy v1.1 — backward elimination (composition_v1_1_backward_elimination)
------------------------------------------------------------------------------------------
Extends v1 by iteratively removing the weakest-contributing predictor (measured
on the selection partition under the configured evaluation metric) until all
remaining predictors meet the ablation threshold or only one predictor remains.
The reduced composite is proposed as an *additional* candidate alongside the
original full composite; both appear in the ledger with derivation edges.

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


def build_backward_eliminated_composites(
    full_composites: list[dict[str, Any]],
    domain: Any,
    *,
    min_contribution: float = 0.01,
    min_predictors: int = 2,
) -> list[dict[str, Any]]:
    """Produce reduced composite candidates via metric-aware backward elimination.

    For each full composite in *full_composites*, iteratively removes the
    predictor with the lowest ablation contribution (computed on the selection
    partition under ``domain.evaluation_metric``) until all remaining predictors
    pass the threshold or only ``min_predictors`` remain.

    Returns only candidates where at least one predictor was eliminated AND the
    reduced set has ≥ min_predictors predictors.  The original full composite is
    already in *full_composites* and need not be duplicated here.

    Parameters
    ----------
    full_composites:
        Specs produced by ``build_composite_candidates`` (composition_v1).
    domain:
        An ``UploadedTableDomain`` instance with ``selection``, ``refit``,
        ``score_metric``, and ``evaluation_metric`` attributes.
    min_contribution:
        Minimum per-predictor contribution; predictors below this are removed.
    min_predictors:
        Minimum number of predictors required for a composite to be proposed.
    """
    from orbita_discovery.core import Candidate
    from .metrics import higher_is_better

    metric: str = getattr(domain, "evaluation_metric", "r2")
    hib = higher_is_better(metric)
    score_fn = getattr(domain, "score_metric", None) or domain.score
    selection = domain.selection

    reduced: list[dict[str, Any]] = []

    for full_spec in full_composites:
        predictors = list(full_spec["predictors"])
        outcome = full_spec["outcome"]
        elimination_ledger: list[dict[str, Any]] = []

        while len(predictors) >= min_predictors + 1:
            trial_payload = {**full_spec, "predictors": predictors}
            trial_c = Candidate(id=full_spec["id"] + "_trial", statement=full_spec["statement"],
                                payload=trial_payload)
            full_model = domain.refit(trial_c, selection)
            if not full_model.get("valid"):
                break
            full_score = score_fn(trial_c, full_model, selection)

            contributions: dict[str, float] = {}
            for p in predictors:
                reduced_preds = [x for x in predictors if x != p]
                rc_payload = {**full_spec, "predictors": reduced_preds,
                              "id": full_spec["id"] + f"_drop_{p}"}
                rc = Candidate(id=rc_payload["id"], statement=full_spec["statement"],
                               payload=rc_payload)
                rm = domain.refit(rc, selection)
                if not rm.get("valid"):
                    contributions[p] = 0.0
                    continue
                rs = score_fn(rc, rm, selection)
                contributions[p] = round(rs - full_score if not hib else full_score - rs, 6)

            failing = {p: v for p, v in contributions.items() if v < min_contribution}
            if not failing:
                break  # all pass — no further elimination needed

            worst = min(failing, key=failing.get)
            elimination_ledger.append({
                "step": len(elimination_ledger) + 1,
                "eliminated": worst,
                "contribution": contributions[worst],
                "full_score": full_score,
                "all_contributions": dict(contributions),
                "metric": metric,
                "partition": "selection",
            })
            predictors.remove(worst)

        if not elimination_ledger or len(predictors) < min_predictors:
            continue  # no elimination happened or result too small

        sorted_preds = sorted(predictors)
        raw = f"composite_linear_v1_1|{'|'.join(sorted_preds)}|{outcome}"
        cid = f"composite:{_slug(outcome)}:{hashlib.sha256(raw.encode()).hexdigest()[:8]}"
        pred_str = " + ".join(sorted_preds)
        eliminated = [e["eliminated"] for e in elimination_ledger]

        parent_scores = {p: v for p, v in
                         full_spec.get("scout_metric", {}).get("parent_scores", {}).items()
                         if p in sorted_preds}
        scout_metric: dict[str, Any] = {
            **full_spec.get("scout_metric", {}),
            "parent_scores": parent_scores,
        }
        if "best_individual_score" in full_spec.get("scout_metric", {}):
            valid_scores = [v for k, v in parent_scores.items() if v is not None]
            if valid_scores:
                scout_metric["best_individual_score"] = round(max(valid_scores), 6)
        if "parent_metric_scores" in full_spec.get("scout_metric", {}):
            pm = {p: v for p, v in
                  full_spec["scout_metric"]["parent_metric_scores"].items()
                  if p in sorted_preds}
            scout_metric["parent_metric_scores"] = pm
            vals = [v for v in pm.values() if v is not None]
            if vals:
                scout_metric["best_individual_metric_score"] = round(
                    max(vals) if hib else min(vals), 6
                )

        reduced.append({
            "id": cid,
            "statement": (
                f"{outcome} can be predicted by a backward-eliminated composite of "
                f"[{pred_str}] (linear model, {len(sorted_preds)} predictors; "
                f"{len(eliminated)} predictor(s) eliminated: {eliminated})."
            ),
            "kind": "composite_linear",
            "composition_strategy": "composition_v1_1_backward_elimination",
            "predictors": sorted_preds,
            "outcome": outcome,
            "parent_candidate_ids": full_spec["parent_candidate_ids"],
            "scout_metric": scout_metric,
            "parents": [full_spec["id"]] + list(full_spec.get("parents", [])),
            "eliminated_predictors": eliminated,
            "elimination_ledger": elimination_ledger,
            "original_composite_id": full_spec["id"],
        })

    return reduced
