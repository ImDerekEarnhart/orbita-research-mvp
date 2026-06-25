"""Schema migration tests for plan hashing.

Invariants:
  - v0.2 historical plan hashes remain verifiable (Run 001, Run 002A)
  - v0.3 new fields (ablation_metric, composition_strategy) are included in hash
  - Mutating ablation_metric changes v0.3 hash but not v0.2 hash
  - Executing a v0.2 plan raises a clear error; historical integrity verification succeeds
  - Migrated fields don't corrupt existing hashes
"""
from __future__ import annotations

import pytest

from orbita_mvp.compiler import (
    PLAN_SCHEMA_V02,
    PLAN_SCHEMA_V03,
    compute_plan_hash,
    verify_plan_schema_executable,
)


# ---------------------------------------------------------------------------
# Minimal plan factory helpers
# ---------------------------------------------------------------------------

def _v02_plan(**overrides) -> dict:
    """Minimal v0.2 plan matching the canonical field set."""
    plan = {
        "schema_version": PLAN_SCHEMA_V02,
        "target_transform": None,
        "outcome_domain": None,
        "evaluation_metric": "r2",
        "thresholds": {"commit_at": 0.25},
        "candidate_generation": {"seed": 20260623},
    }
    plan.update(overrides)
    return plan


def _v03_plan(**overrides) -> dict:
    """Minimal v0.3 plan with all required immutable fields."""
    plan = {
        "schema_version": PLAN_SCHEMA_V03,
        "target_transform": None,
        "outcome_domain": None,
        "evaluation_metric": "r2",
        "ablation_metric": "r2",
        "composition_strategy": "composition_v1_1_backward_elimination",
        "thresholds": {"commit_at": 0.25},
        "candidate_generation": {"seed": 20260623},
    }
    plan.update(overrides)
    return plan


# ---------------------------------------------------------------------------
# Test 1: v0.2 plan hash does not include ablation_metric
# ---------------------------------------------------------------------------

def test_v02_hash_excludes_ablation_metric():
    """v0.2 hash is computed on 5 original fields only; ablation_metric is ignored."""
    plan_a = _v02_plan()
    plan_b = _v02_plan(ablation_metric="rmsle")  # extra field not in v0.2 set

    hash_a = compute_plan_hash(plan_a)
    hash_b = compute_plan_hash(plan_b)

    assert hash_a == hash_b, (
        "v0.2 hashes must be identical regardless of ablation_metric presence"
    )


# ---------------------------------------------------------------------------
# Test 2: v0.2 plan hash does not include composition_strategy
# ---------------------------------------------------------------------------

def test_v02_hash_excludes_composition_strategy():
    """v0.2 hash is stable even when composition_strategy differs."""
    plan_a = _v02_plan()
    plan_b = _v02_plan(composition_strategy="composition_v1_1_backward_elimination")

    assert compute_plan_hash(plan_a) == compute_plan_hash(plan_b)


# ---------------------------------------------------------------------------
# Test 3: v0.3 hash changes when ablation_metric changes
# ---------------------------------------------------------------------------

def test_v03_hash_changes_with_ablation_metric():
    """Mutating ablation_metric must change the v0.3 hash."""
    plan_r2 = _v03_plan(ablation_metric="r2")
    plan_rmsle = _v03_plan(ablation_metric="rmsle")

    assert compute_plan_hash(plan_r2) != compute_plan_hash(plan_rmsle), (
        "v0.3 hash must differ when ablation_metric changes"
    )


# ---------------------------------------------------------------------------
# Test 4: v0.3 hash changes when composition_strategy changes
# ---------------------------------------------------------------------------

def test_v03_hash_changes_with_composition_strategy():
    """Mutating composition_strategy must change the v0.3 hash."""
    plan_v1 = _v03_plan(composition_strategy="composition_v1")
    plan_v11 = _v03_plan(composition_strategy="composition_v1_1_backward_elimination")

    assert compute_plan_hash(plan_v1) != compute_plan_hash(plan_v11), (
        "v0.3 hash must differ when composition_strategy changes"
    )


# ---------------------------------------------------------------------------
# Test 5: Schema-aware v0.2 hash stability (production Run 002A integrity invariant)
# ---------------------------------------------------------------------------

def test_v02_plan_hash_is_stable_across_new_field_additions():
    """A v0.2 plan hash must remain byte-for-byte identical even after new v0.3 fields
    are added to the plan dict — verifying the schema-aware routing invariant.

    NOTE ON PRODUCTION RUN 002A:
    The actual Run 002A plan hash is f699bdc51e94a4a73dcf2535fe400e1d442f4c3ae6c62e87d27793ee7f61ed01
    (audit manifest: C:\\Users\\Dereks\\run_002a_audit_manifest.json, plan_d4569047c29146d2).
    That plan's candidate_generation dict includes the full output of generate_table_candidates()
    on the 750k-row training CSV and is only accessible from the Railway production DB.
    This test verifies the invariant (new fields cannot corrupt historical hashes) using a
    synthetic v0.2 plan with a pre-validated reference hash.

    The pre-validated hash below was computed by the CURRENT engine — it serves as a
    regression guard: if the v0.2 hash routing changes, this test fails.
    """
    # Synthetic v0.2 plan with known canonical content.
    # Hash pre-computed by current engine on 2026-06-25 after schema-aware routing was added.
    # This is the reference hash for schema-integrity regression testing.
    SYNTHETIC_V02_HASH = "e8c65d02beab73e8b5e3110e662bcd283a2fb27f51d895844c33f14a79479202"

    plan_v02 = {
        "schema_version": PLAN_SCHEMA_V02,
        "target_transform": "log1p",
        "outcome_domain": "nonneg",
        "evaluation_metric": "rmsle",
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
        },
        "candidate_generation": {
            "seed": 20260623,
            "scout_fraction": 0.6,
            "confirmation_fraction": 0.25,
            "final_validation_fraction": 0.15,
        },
    }

    # Regression guard: hash must be stable under the current engine
    assert compute_plan_hash(plan_v02) == SYNTHETIC_V02_HASH, (
        "v0.2 hash routing changed — historical plan hashes would be corrupted."
    )

    # Core invariant: adding v0.3 fields MUST NOT change the v0.2 hash
    plan_v02_with_new_fields = {
        **plan_v02,
        "ablation_metric": "rmsle",
        "composition_strategy": "composition_v1_1_backward_elimination",
        "some_future_field": True,
    }
    assert compute_plan_hash(plan_v02_with_new_fields) == SYNTHETIC_V02_HASH, (
        "Adding new fields to a v0.2 plan must not change its hash. "
        "This invariant protects Run 001, Run 002A, and all historical plans."
    )


# ---------------------------------------------------------------------------
# Test 6: v0.2 plan execution raises a clear error
# ---------------------------------------------------------------------------

def test_v02_plan_execution_raises_clear_error():
    """verify_plan_schema_executable must raise for v0.2 plans with a useful message."""
    plan = _v02_plan()
    with pytest.raises(ValueError) as exc_info:
        verify_plan_schema_executable(plan)
    msg = str(exc_info.value)
    assert "orbita-research-plan/0.2" in msg, "Error must identify the failing schema"
    assert "orbita-research-plan/0.3" in msg, "Error must name the required schema"


# ---------------------------------------------------------------------------
# Test 7: v0.3 plan execution does NOT raise
# ---------------------------------------------------------------------------

def test_v03_plan_is_executable():
    """verify_plan_schema_executable must not raise for v0.3 plans."""
    plan = _v03_plan()
    verify_plan_schema_executable(plan)  # must not raise


# ---------------------------------------------------------------------------
# Test 8: Missing schema_version defaults to v0.2 behavior
# ---------------------------------------------------------------------------

def test_missing_schema_defaults_to_v02():
    """Plans without schema_version are treated as v0.2; their hash excludes new fields."""
    plan_no_schema = {
        "target_transform": None,
        "outcome_domain": None,
        "evaluation_metric": "r2",
        "ablation_metric": "rmsle",
        "thresholds": {"commit_at": 0.25},
        "candidate_generation": {"seed": 20260623},
    }
    plan_explicit_v02 = {
        "schema_version": PLAN_SCHEMA_V02,
        "target_transform": None,
        "outcome_domain": None,
        "evaluation_metric": "r2",
        "ablation_metric": "rmsle",
        "thresholds": {"commit_at": 0.25},
        "candidate_generation": {"seed": 20260623},
    }
    assert compute_plan_hash(plan_no_schema) == compute_plan_hash(plan_explicit_v02), (
        "Plans without schema_version must hash identically to explicit v0.2 plans"
    )


# ---------------------------------------------------------------------------
# Test 9: Adding new v0.3 fields to a v0.2 plan does not corrupt the v0.2 hash
# ---------------------------------------------------------------------------

def test_adding_v03_fields_to_v02_plan_does_not_corrupt_hash():
    """If a v0.2 plan gets extra keys (e.g. for migration tooling), the hash stays stable."""
    plan_clean = _v02_plan()
    plan_extra_keys = _v02_plan(
        ablation_metric="rmsle",
        composition_strategy="composition_v1_1_backward_elimination",
        extra_field="ignored",
    )
    assert compute_plan_hash(plan_clean) == compute_plan_hash(plan_extra_keys), (
        "Extra fields not in the v0.2 immutable set must not affect the v0.2 hash"
    )


# ---------------------------------------------------------------------------
# Test 10: v0.2 and v0.3 hashes for identical shared fields differ (schema affects hash)
# ---------------------------------------------------------------------------

def test_v02_and_v03_hashes_differ_for_same_shared_fields():
    """Same shared-field values produce different hashes under v0.2 vs v0.3 schemas."""
    shared = {
        "target_transform": None,
        "outcome_domain": None,
        "evaluation_metric": "r2",
        "thresholds": {"commit_at": 0.25},
        "candidate_generation": {"seed": 20260623},
    }
    plan_v02 = {**shared, "schema_version": PLAN_SCHEMA_V02}
    plan_v03 = {
        **shared,
        "schema_version": PLAN_SCHEMA_V03,
        "ablation_metric": "r2",
        "composition_strategy": "composition_v1_1_backward_elimination",
    }
    # v0.3 includes extra fields → its canonical JSON differs → hash differs
    assert compute_plan_hash(plan_v02) != compute_plan_hash(plan_v03)
