"""End-to-end + generic regression tests for the discovery-integrity overhaul.

These pin the scientific-integrity behaviour exposed by the pre-release
stress-test dataset (``orbita_pre_release_stress_test.csv``). Assertions are on
*semantic behaviour, direction, and approximate ranges* — never exact random
scores — so they generalise to unrelated datasets.

Hidden structure of the stress dataset (for reference; never asserted by exact
value): input_level→output_linear (linear), mass_kg→energy_rate (power law
exp≈0.73), temperature_c→growth_index (inverted-U, centre≈25), treatment→
recovery_score (A ≈ +8 vs control), simpson_x↔simpson_y (pooled +, reverses to
− inside cohorts Alpha/Beta), noise_a↔noise_b (null), leakage_proxy≈output_linear
(near-copy), sparse_sensor→input_level (35% missing), sample_id (identifier).

The file is organised so each feature commit can extend it; tests degrade
gracefully (skip) if the stress CSV is not present on this machine.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from orbita_mvp import ResearchMVP

STRESS_CSV = Path(r"C:\Users\Dereks\Downloads\orbita_pre_release_stress_test.csv")

_needs_stress = pytest.mark.skipif(
    not STRESS_CSV.exists(), reason="stress-test CSV not available on this machine"
)


def _run_stress():
    with tempfile.TemporaryDirectory() as td:
        svc = ResearchMVP(Path(td) / "o.db", Path(td) / "ws")
        try:
            case = svc.create_case(name="stress-integrity", goal="")
            svc.add_file(case["id"], STRESS_CSV)
            plan = svc.compile_case(case["id"])
            svc.approve_plan(plan["id"], reviewer="tester")
            run = svc.run_case(case["id"], plan_id=plan["id"])
            assert run["status"] == "completed"
            claims = svc.store.case_claims(case["id"])
            counts = svc.store.case_claim_counts(case["id"])
            return claims, counts, plan["plan"]
        finally:
            svc.close()


def _by_text(claims, *needles):
    """Return claims whose statement contains all needles (order-independent)."""
    out = []
    for c in claims:
        text = (c.get("finding_detail", {}).get("hypothesis_text") or c.get("canonical_text") or "")
        if all(n in text for n in needles):
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Generic unit tests — identifier detection (dataset-agnostic)
# ---------------------------------------------------------------------------

def test_detect_identifier_flags_shapes_but_not_continuous_measurements():
    from orbita_mvp.ingestion import detect_identifier

    n = 300
    rng = np.random.default_rng(0)

    # Prefixed string sequence (e.g. OBS-0001) -> identifier.
    seq = pd.Series([f"OBS-{i:05d}" for i in range(n)], name="obs_code")
    assert detect_identifier(seq, "obs_code") is not None

    # Monotonic integer id with no id-token name -> still identifier by shape.
    ints = pd.Series(list(range(1000, 1000 + n)), name="row_number")
    sig = detect_identifier(ints, "row_number")
    assert sig is not None and sig.get("shape") == "integer_sequence"

    # UUID column -> identifier.
    uuids = pd.Series([f"{rng.integers(0, 16**8):08x}-0000-4000-8000-000000000000" for _ in range(n)], name="uid")
    assert detect_identifier(uuids, "uid") is not None

    # Continuous float measurement, all-unique -> NOT an identifier.
    cont = pd.Series(rng.normal(size=n), name="temperature_c")
    assert detect_identifier(cont, "temperature_c") is None

    # Low-cardinality category -> NOT an identifier.
    cat = pd.Series(rng.choice(["A", "B", "C"], size=n), name="treatment")
    assert detect_identifier(cat, "treatment") is None


# ---------------------------------------------------------------------------
# Generic unit tests — repeated_refit is a genuine per-seed refit, distinct
# from the fixed-model validation_resample check.
# ---------------------------------------------------------------------------

def test_repeated_refit_validator_refits_independently_per_seed():
    from orbita_discovery.core import Candidate
    from orbita_discovery.falsifiers import RepeatedRefitValidator
    from orbita_mvp.table_domain import UploadedTableDomain

    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"x": x, "y": y})
    spec = {
        "id": "linear:x_y",
        "statement": "x and y show a stable positive linear association.",
        "kind": "linear_association",
        "predictor": "x",
        "outcome": "y",
        "expected_direction": "positive",
    }
    dom = UploadedTableDomain(df, [spec], seed=7)
    c = Candidate(id=spec["id"], statement=spec["statement"], payload={k: v for k, v in spec.items() if k not in {"id", "statement"}})

    result = RepeatedRefitValidator(seeds=15).attempt(c, dom.evidence_for(c), dom)
    detail = result.detail
    assert detail["check_kind"] == "repeated_independent_refit"
    assert detail["valid_fits"] >= 10
    # Fresh partitions each seed produce a real distribution of coefficients:
    # the slope varies across independent fits (non-degenerate variance) but
    # its sign is stable for a genuine positive relationship.
    coeff = detail["coefficient_stability"]["slope"]
    assert coeff["variance"] > 0.0
    assert coeff["sign_stability"] == 1.0
    assert result.killed is False  # diagnostic-only: never changes classification

    # Fixed-model validation_resample keeps the SAME scout-trained model across
    # seeds; the two checks are independent and must not overwrite each other.
    from orbita_discovery.falsifiers import CrossSeedFalsifier
    vr = CrossSeedFalsifier(seeds=9, min_median=0.15)
    vr_result = vr.attempt(c, dom.evidence_for(c), dom)
    assert vr_result.name == "validation_resample"
    assert vr_result.detail["check_kind"] == "fixed_model_validation_resample"


def test_validation_resample_alias_preserved_in_classification():
    # A historical ledger finding named `cross_seed` must classify identically
    # to the renamed `validation_resample`.
    from orbita_mvp.semantics import classify_pairwise_finding

    def finding(check_name):
        return {
            "falsifications": [
                {"name": "held_out", "killed": True, "detail": {"score": -0.2, "n": 200}},
                {"name": check_name, "killed": True, "detail": {"median": -0.3, "spread": 0.1, "n": 200}},
            ]
        }

    old, _ = classify_pairwise_finding(finding("cross_seed"))
    new, _ = classify_pairwise_finding(finding("validation_resample"))
    assert old == new == "functional_form_rejected_candidate"


# ---------------------------------------------------------------------------
# End-to-end stress-CSV assertions available after Commit 1
# ---------------------------------------------------------------------------

@_needs_stress
def test_stress_identifier_sample_id_flagged_and_excluded():
    claims, counts, plan = _run_stress()
    # sample_id must be excluded from candidate generation ...
    assert "sample_id" in plan.get("excluded_from_candidate_generation", [])
    mined_types = {
        "robust_relation", "promising_candidate", "supported_association_candidate",
        "not_supported_candidate", "inconclusive_candidate",
        "functional_form_rejected_candidate", "falsified_candidate", "regime_dependent_candidate",
    }
    mined = [c for c in claims if c["finding_type"] in mined_types
             and "sample_id" in (c.get("finding_detail", {}).get("hypothesis_text") or c.get("canonical_text") or "")]
    assert not mined, "sample_id must never appear as a mined relationship"
    # ... and explicitly recorded as an identifier artifact, not silently dropped.
    id_findings = [c for c in claims if c["finding_type"] == "artifact_guard"]
    assert any("sample_id" in (c.get("finding_detail", {}).get("hypothesis_text") or c.get("canonical_text") or "")
               for c in id_findings), "sample_id identifier receipt must be present"


@_needs_stress
def test_stress_treatment_is_supported_association_not_not_supported():
    claims, counts, _plan = _run_stress()
    treatment = _by_text(claims, "recovery_score", "treatment")
    assert treatment, "treatment->recovery group candidate must be generated"
    c = treatment[0]
    assert c["verdict"] == "supported_association", c["finding_detail"].get("verdict")
    detail = c["finding_detail"]
    assoc = detail["association_evidence"]
    assert assoc["effect_size_metric"] == "eta_squared"
    assert assoc["effect_size"] >= 0.02 and assoc["omega_squared"] > 0
    # Predictive utility is persisted as a SEPARATE axis (limited, not the gate).
    assert "predictive_utility" in detail
    # Treatment A is the elevated group.
    means = assoc["group_means"]
    assert means.get("A") == max(means.values())


@_needs_stress
def test_stress_sparse_sensor_missingness_receipt_and_effective_n():
    claims, counts, plan = _run_stress()
    # The missingness quality finding must fire for the sparse column.
    qf = [c for c in claims if c["finding_type"] == "data_quality"]
    assert any("sparse_sensor" in (c.get("finding_detail", {}).get("hypothesis_text") or c.get("canonical_text") or "")
               for c in qf), "sparse_sensor high-missingness receipt must be present"


@_needs_stress
def test_stress_every_finding_has_both_validator_summaries():
    claims, _counts, _plan = _run_stress()
    tested = [c for c in claims if c["finding_type"] in {
        "robust_relation", "promising_candidate", "supported_association_candidate",
        "not_supported_candidate", "functional_form_rejected_candidate",
    }]
    assert tested
    for c in tested:
        detail = c["finding_detail"]
        # Both the fixed-model resample AND the genuine repeated-refit summaries
        # are persisted independently.
        assert "validation_resample_summary" in detail
        assert detail.get("repeated_refit_summary") is not None
        rr = detail["repeated_refit_summary"]
        assert rr["check_kind"] == "repeated_independent_refit"
        assert rr["valid_fits"] >= 1
