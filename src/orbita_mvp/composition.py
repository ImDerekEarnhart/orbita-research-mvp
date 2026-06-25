"""Post-hoc composite candidate builder.

After individual pairwise survivors are known, this module proposes composite
linear models that combine multiple predictors for the same outcome.  The
composite is domain-agnostic: it knows nothing about column semantics.
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
) -> list[dict[str, Any]]:
    """Return composite_linear candidates for any outcome that has >=min_predictors survivors.

    Each composite groups all pairwise-linear survivors that share the same
    outcome column.  The composite id is deterministic from its sorted predictor
    list and outcome.  Parent candidate ids are recorded for provenance.
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
        composites.append({
            "id": cid,
            "statement": (
                f"{outcome} can be predicted by a composite of [{predictor_str}] "
                f"(linear model, {len(predictors)} predictors)."
            ),
            "kind": "composite_linear",
            "predictors": predictors,
            "outcome": outcome,
            "parent_candidate_ids": [f["candidate"]["id"] for f in group],
            "scout_metric": {
                "parent_scores": {
                    f["candidate"]["payload"]["predictor"]: round(f["verdict"]["score"], 6)
                    for f in group
                },
                "best_individual_score": round(
                    max(f["verdict"]["score"] for f in group), 6
                ),
            },
            "parents": [f["candidate"]["id"] for f in group],
        })
    return composites
