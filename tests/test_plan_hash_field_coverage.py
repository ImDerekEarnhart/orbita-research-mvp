"""Exhaustive field-by-field plan hash coverage for orbita-research-plan/0.3.

Every field in the v0.3 canonical payload must independently affect the hash.
Historical v0.2 hashes must remain verifiable without alteration.

Canonical v0.3 payload fields (from _IMMUTABLE_FIELDS_V03):
  top-level: target_transform, outcome_domain, evaluation_metric,
             ablation_metric, composition_strategy, thresholds, candidate_generation

  thresholds subfields: commit_at, baseline_margin, held_out_min, cross_seed_count,
                        cross_seed_min, cross_seed_max_spread, composite_min_predictors,
                        composite_max_predictors, composite_min_improvement,
                        ablation_min_contribution, ablation_min_absolute_improvement,
                        ablation_min_relative_improvement

  candidate_generation subfields: seed, scout_fraction, confirmation_fraction,
                                  final_validation_fraction
"""
from __future__ import annotations

import pytest

from orbita_mvp.compiler import (
    PLAN_SCHEMA_V02,
    PLAN_SCHEMA_V03,
    compute_plan_hash,
)


# ---------------------------------------------------------------------------
# Canonical v0.3 baseline plan — all fields present with stable defaults
# ---------------------------------------------------------------------------

_BASELINE_V03 = {
    "schema_version": PLAN_SCHEMA_V03,
    "target_transform": "log1p",
    "outcome_domain": "nonneg",
    "evaluation_metric": "rmsle",
    "ablation_metric": "rmsle",
    "composition_strategy": "composition_v1_1_backward_elimination",
    "thresholds": {
        "commit_at": 0.25,
        "baseline_margin": 0.05,
        "held_out_min": 0.15,
        "cross_seed_count": 9,
        "cross_seed_min": 0.15,
        "cross_seed_max_spread": 0.65,
        "composite_min_predictors": 2,
        "composite_max_predictors": 10,
        "composite_min_improvement": 0.01,
        "ablation_min_contribution": 0.01,
        "ablation_min_absolute_improvement": 0.01,
        "ablation_min_relative_improvement": None,
    },
    "candidate_generation": {
        "seed": 20260625,
        "scout_fraction": 0.60,
        "confirmation_fraction": 0.25,
        "final_validation_fraction": 0.15,
    },
}


def _mutate(plan: dict, *path, value) -> dict:
    """Return a shallow-copy of plan with one nested value replaced."""
    import copy
    result = copy.deepcopy(plan)
    d = result
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value
    return result


_baseline_hash = compute_plan_hash(_BASELINE_V03)


# ---------------------------------------------------------------------------
# Parametrized: every top-level and nested field must change the hash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,new_value", [
    # Top-level immutable fields
    (("target_transform",), None),
    (("target_transform",), "sqrt"),
    (("outcome_domain",), None),
    (("outcome_domain",), "real"),
    (("evaluation_metric",), "r2"),
    (("evaluation_metric",), "rmse"),
    (("evaluation_metric",), "mae"),
    (("ablation_metric",), "r2"),
    (("ablation_metric",), None),
    (("composition_strategy",), "composition_v1"),
    (("composition_strategy",), None),
    # thresholds subfields
    (("thresholds", "commit_at"), 0.30),
    (("thresholds", "baseline_margin"), 0.10),
    (("thresholds", "held_out_min"), 0.20),
    (("thresholds", "cross_seed_count"), 5),
    (("thresholds", "cross_seed_min"), 0.10),
    (("thresholds", "cross_seed_max_spread"), 0.50),
    (("thresholds", "composite_min_predictors"), 3),
    (("thresholds", "composite_max_predictors"), 5),
    (("thresholds", "composite_min_improvement"), 0.05),
    (("thresholds", "ablation_min_contribution"), 0.05),
    (("thresholds", "ablation_min_absolute_improvement"), 0.05),
    (("thresholds", "ablation_min_relative_improvement"), 0.10),
    # candidate_generation subfields
    (("candidate_generation", "seed"), 99999),
    (("candidate_generation", "scout_fraction"), 0.70),
    (("candidate_generation", "confirmation_fraction"), 0.20),
    (("candidate_generation", "final_validation_fraction"), 0.10),
])
def test_v03_hash_changes_when_field_mutated(path, new_value):
    """Mutating any single field in the v0.3 canonical payload must change the hash."""
    mutated = _mutate(_BASELINE_V03, *path, value=new_value)
    mutated_hash = compute_plan_hash(mutated)
    assert mutated_hash != _baseline_hash, (
        f"Hash did not change when {'.'.join(str(p) for p in path)!r} "
        f"was changed to {new_value!r}. "
        "This field is NOT protected by the plan hash."
    )


# ---------------------------------------------------------------------------
# Verify canonical payload is serialized in a stable, deterministic order
# ---------------------------------------------------------------------------

def test_v03_hash_is_stable_across_dict_insertion_orders():
    """Plan hash must not depend on the insertion order of keys in the plan dict."""
    import copy

    plan_a = copy.deepcopy(_BASELINE_V03)
    # Build plan_b with keys inserted in reverse order
    plan_b = {}
    for k in reversed(list(plan_a.keys())):
        plan_b[k] = copy.deepcopy(plan_a[k])

    assert compute_plan_hash(plan_a) == compute_plan_hash(plan_b), (
        "v0.3 hash must be deterministic regardless of dict key insertion order"
    )


# ---------------------------------------------------------------------------
# Verify the exact canonical payload string (change detector)
# ---------------------------------------------------------------------------

def test_v03_canonical_payload_is_documented():
    """Print and verify the exact canonical JSON used for hashing.

    This is a documentation test — it fails if the canonical payload changes,
    alerting developers that existing plan hashes would be invalidated.
    """
    import json
    import hashlib

    fields = (
        "target_transform", "outcome_domain", "evaluation_metric",
        "ablation_metric", "composition_strategy", "thresholds", "candidate_generation",
    )
    subset = {k: _BASELINE_V03.get(k) for k in fields}
    canonical = json.dumps(subset, sort_keys=True, separators=(",", ":"), default=str)
    computed = hashlib.sha256(canonical.encode()).hexdigest()

    # Recomputing via the engine must match
    assert compute_plan_hash(_BASELINE_V03) == computed, (
        "compute_plan_hash must produce the same result as direct SHA-256 of "
        "the sorted canonical JSON of immutable fields"
    )


# ---------------------------------------------------------------------------
# v0.2 fields NOT in v0.3 immutable set: schema_version, mode, goal, etc.
# mutating these must NOT change the v0.3 hash
# ---------------------------------------------------------------------------

def test_v03_hash_stable_for_non_immutable_fields():
    """Fields outside the v0.3 immutable set must not affect the hash."""
    import copy

    plan_a = copy.deepcopy(_BASELINE_V03)
    plan_b = copy.deepcopy(_BASELINE_V03)
    plan_b["mode"] = "guided"
    plan_b["goal"] = "find the best predictor"
    plan_b["status"] = "approved"
    plan_b["some_future_field"] = {"deeply": {"nested": True}}

    assert compute_plan_hash(plan_a) == compute_plan_hash(plan_b), (
        "Non-immutable fields (mode, goal, status, etc.) must not affect the v0.3 hash"
    )


# ---------------------------------------------------------------------------
# Historical v0.2 hashes must remain stable (cross-schema non-interference)
# ---------------------------------------------------------------------------

def test_v02_hash_unaffected_by_v03_only_fields():
    """Adding v0.3-only fields to a v0.2 plan must not change the v0.2 hash."""
    plan_clean = {
        "schema_version": PLAN_SCHEMA_V02,
        "target_transform": "log1p",
        "outcome_domain": "nonneg",
        "evaluation_metric": "rmsle",
        "thresholds": {"commit_at": 0.25},
        "candidate_generation": {"seed": 20260623},
    }
    plan_with_v03_fields = {
        **plan_clean,
        "ablation_metric": "rmsle",
        "composition_strategy": "composition_v1_1_backward_elimination",
        "new_future_field": "irrelevant",
    }
    assert compute_plan_hash(plan_clean) == compute_plan_hash(plan_with_v03_fields), (
        "v0.3-only fields must not corrupt historical v0.2 hashes"
    )


# ---------------------------------------------------------------------------
# v0.3 and v0.2 hashes differ even for identical shared fields
# ---------------------------------------------------------------------------

def test_v02_and_v03_hashes_differ_for_same_shared_fields():
    """Same shared values produce different hashes under v0.2 vs v0.3 (different field sets)."""
    shared_core = {
        "target_transform": "log1p",
        "outcome_domain": "nonneg",
        "evaluation_metric": "rmsle",
        "thresholds": {"commit_at": 0.25},
        "candidate_generation": {"seed": 20260623},
    }
    plan_v02 = {**shared_core, "schema_version": PLAN_SCHEMA_V02}
    plan_v03 = {
        **shared_core,
        "schema_version": PLAN_SCHEMA_V03,
        "ablation_metric": "rmsle",
        "composition_strategy": "composition_v1_1_backward_elimination",
    }
    assert compute_plan_hash(plan_v02) != compute_plan_hash(plan_v03), (
        "v0.2 and v0.3 plans with identical shared fields must produce different hashes "
        "because v0.3 includes additional fields in the canonical payload"
    )


# ---------------------------------------------------------------------------
# Partition fractions sum to ≤ 1.0 (structural sanity, not a hash test)
# ---------------------------------------------------------------------------

def test_v03_baseline_partition_fractions_are_valid():
    """Confirm baseline partition fractions are internally consistent."""
    cg = _BASELINE_V03["candidate_generation"]
    total = cg["scout_fraction"] + cg["confirmation_fraction"] + cg["final_validation_fraction"]
    assert abs(total - 1.0) < 1e-9, f"Partition fractions must sum to 1.0; got {total}"
