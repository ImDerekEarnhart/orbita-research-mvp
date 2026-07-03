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


# ---------------------------------------------------------------------------
# Generic unit test — nonlinear forms are scored in original units and recover
# the true power-law exponent.
# ---------------------------------------------------------------------------

def test_fit_form_recovers_power_law_exponent_in_original_space():
    from orbita_mvp.table_domain import _fit_form

    rng = np.random.default_rng(3)
    x = rng.uniform(1, 50, size=400)
    y = 4.5 * x ** 0.73 * np.exp(rng.normal(scale=0.03, size=400))  # power law + small noise

    ll = _fit_form(x, y, "log_log")
    assert ll is not None
    assert abs(ll["exponent"] - 0.73) < 0.05
    # Original-space R² must be high and comparable to a linear fit's R².
    assert ll["r2"] > 0.9

    quad = _fit_form(x, y, "quadratic")
    assert quad is not None and quad["r2"] > 0.9

    # Log of non-positive values is inapplicable, not a crash.
    yneg = y - y.mean()
    assert _fit_form(x, yneg, "log_y") is None


# ---------------------------------------------------------------------------
# End-to-end stress-CSV assertions available after Commit 2 (nonlinear families)
# ---------------------------------------------------------------------------

@_needs_stress
def test_stress_power_law_family_grouped_and_preferred_exponent():
    claims, _counts, _plan = _run_stress()
    fam = _by_text(claims, "energy_rate", "mass_kg")
    forms = {c["finding_detail"].get("model_family", {}).get("form") for c in fam
             if c["finding_detail"].get("model_family")}
    # Raw linear and the log-log power law are members of ONE family.
    assert {"linear", "log_log"} <= forms, forms
    # The power-law form is preferred and its exponent is close to the true 0.73.
    loglog = [c for c in fam if c["finding_detail"].get("model_family", {}).get("form") == "log_log"]
    assert loglog
    mf = loglog[0]["finding_detail"]["model_family"]
    assert mf["preferred_form"] == "log_log", mf
    assert mf["is_preferred"] is True
    exp = mf.get("power_law_exponent") or mf.get("preferred_power_law_exponent")
    assert exp is not None and 0.6 <= exp <= 0.85, exp
    # Both forms survived; the raw linear is supported but not preferred.
    linear = [c for c in fam if c["finding_detail"].get("model_family", {}).get("form") == "linear"]
    assert linear and linear[0]["verdict"] == "committed"
    assert linear[0]["finding_detail"]["model_family"]["is_preferred"] is False


@_needs_stress
def test_stress_temperature_quadratic_preferred_linear_is_functional_form_rejected():
    claims, _counts, _plan = _run_stress()
    fam = _by_text(claims, "temperature_c", "growth_index")
    assert fam, "temperature/growth family must be generated (nonlinear screen)"

    quad = [c for c in fam if "quadratic" in c["canonical_text"]]
    assert quad, "a quadratic form must be generated for the inverted-U relationship"
    assert quad[0]["verdict"] == "committed"
    assert quad[0]["finding_detail"]["model_family"]["is_preferred"] is True

    linear = [c for c in fam if c["finding_detail"].get("model_family", {}).get("form") == "linear"]
    assert linear, "the raw linear form must be present as a family member"
    lc = linear[0]
    # The failed linear fit is a wrong functional form, NOT a refutation of the
    # underlying relationship.
    assert lc["verdict"] == "functional_form_rejected"
    assert lc["verdict"] != "rejected"
    assert lc["finding_detail"].get("alternative_candidate_id"), "must point at the surviving curved sibling"


# ---------------------------------------------------------------------------
# Generic unit test — subgroup reversal detector (Simpson vs. non-reversing).
# ---------------------------------------------------------------------------

def test_detect_subgroup_reversal_flags_simpson_but_not_consistent_relationship():
    from orbita_mvp.subgroup import detect_subgroup_reversal

    rng = np.random.default_rng(5)
    rows = []
    # Two cohorts, each with a strong NEGATIVE within-group slope, offset so the
    # pooled slope is POSITIVE (classic Simpson).
    for cohort, x0, y0 in [("A", 0.0, 0.0), ("B", 10.0, 10.0)]:
        for _ in range(150):
            x = x0 + rng.uniform(0, 5)
            y = y0 - 1.5 * (x - x0) + rng.normal(scale=0.5)
            rows.append({"x": x, "y": y, "cohort": cohort, "flat": rng.choice(["p", "q"])})
    df = pd.DataFrame(rows)

    report = detect_subgroup_reversal(df, "x", "y", ["cohort", "flat"], min_group_n=25)
    assert report is not None
    assert report["conditioning_variable"] == "cohort"
    assert report["pooled_direction"] == "positive"
    assert all(g["direction"] == "negative" for g in report["groups"])
    assert {s["group_value"] for s in report["scoped_claims"]} == {"A", "B"}

    # A relationship that is consistently positive in every subgroup must NOT be
    # flagged as a reversal.
    rows2 = []
    for cohort in ["A", "B"]:
        for _ in range(150):
            x = rng.uniform(0, 10)
            rows2.append({"x": x, "y": 2.0 * x + rng.normal(scale=0.5), "cohort": cohort})
    df2 = pd.DataFrame(rows2)
    assert detect_subgroup_reversal(df2, "x", "y", ["cohort"], min_group_n=25) is None


# ---------------------------------------------------------------------------
# End-to-end stress-CSV assertions available after Commit 3 (subgroup reversal)
# ---------------------------------------------------------------------------

@_needs_stress
def test_stress_simpson_pooled_is_regime_dependent_not_committed():
    claims, counts, _plan = _run_stress()
    linear = [c for c in claims
              if "simpson_x and simpson_y" in c["canonical_text"] and "linear" in c["canonical_text"]]
    assert linear, "the pooled linear simpson_x->simpson_y claim must be generated"
    c = linear[0]
    # Item 5/6: not committed as a universal positive relationship; classified
    # as a subgroup reversal.
    assert c["verdict"] == "regime_dependent", c["verdict"]
    assert c["verdict"] != "committed"
    sw = c["finding_detail"]["subgroup_warning"]
    assert sw["conditioning_variable"] == "cohort"
    assert sw["pooled_direction"] == "positive"
    # Item 7: both cohorts negative within-group.
    dirs = {g["group"]: g["direction"] for g in sw["groups"]}
    assert dirs.get("Alpha") == "negative" and dirs.get("Beta") == "negative"
    assert counts["regime_dependent_count"] >= 1


@_needs_stress
def test_stress_simpson_scoped_within_group_claims_recorded():
    claims, _counts, _plan = _run_stress()
    scoped = [c for c in claims if c["finding_type"] == "scoped_association"
              and "simpson" in c["canonical_text"].lower()]
    by_group = {}
    for c in scoped:
        scope = c["finding_detail"].get("scope", {})
        by_group[scope.get("group_value")] = c
    assert "Alpha" in by_group and "Beta" in by_group
    for g in ("Alpha", "Beta"):
        c = by_group[g]
        assert c["verdict"] == "supported_association"
        assert c["finding_detail"]["association_evidence"]["direction"] == "negative"
        assert "negative association" in c["canonical_text"]


# ---------------------------------------------------------------------------
# Generic unit tests — near-copy leakage detection and composite failure modes.
# ---------------------------------------------------------------------------

def test_near_copy_flags_noisy_duplicate_but_not_scaled_real_law():
    from orbita_mvp.artifacts import detect_structural_relations

    rng = np.random.default_rng(9)
    base = rng.uniform(50, 200, size=400)
    df = pd.DataFrame({
        "source": base,
        "leaky_copy": base + rng.normal(scale=0.2, size=400),   # near-identity noisy copy
        "real_scaled": 2.6 * base + 15 + rng.normal(scale=8, size=400),  # real, slope != 1
    })
    rel = detect_structural_relations(df)
    kinds = {tuple(sorted(v["columns"])): v["kind"] for v in rel.values()}
    assert kinds.get(("leaky_copy", "source")) == "near_duplicate_copy"
    # A strong but genuinely different-quantity relationship (slope 2.6) is NOT
    # flagged as a leak, however tight.
    assert ("real_scaled", "source") not in kinds


def test_composite_no_incremental_value_is_not_refutation(tmp_path):
    rng = np.random.default_rng(11)
    n = 400
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=0.3, size=n)  # collinear with x1 (corr < near-copy)
    y = 2.0 * x1 + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"row_id": range(n), "x1": x1, "x2": x2, "y": y})
    p = tmp_path / "redundant.csv"
    df.to_csv(p, index=False)

    svc = ResearchMVP(tmp_path / "t.db", tmp_path / "ws")
    try:
        c = svc.create_case(name="redundant", goal="")
        svc.add_file(c["id"], p)
        svc.compile_case(c["id"])
        svc.run_case(c["id"], auto_approve=True)
        claims = svc.store.case_claims(c["id"])
    finally:
        svc.close()

    composites = [cl for cl in claims if "composite" in cl["canonical_text"].lower()]
    assert composites, "a composite should be proposed from two collinear predictors"
    killed = [cl for cl in composites if cl["finding_type"] == "no_incremental_value_candidate"]
    assert killed, "a redundant composite must be classified no_incremental_value, not refuted"
    for cl in killed:
        assert cl["verdict"] == "not_supported"
        assert cl["verdict"] != "rejected"
        diag = cl["finding_detail"].get("rejection_diagnostics", {})
        modes = set(diag.get("composite_failure_mode", []))
        assert modes and modes <= {"improvement", "ablation"}, modes
        assert "incremental value" in (cl["finding_detail"].get("rejection_reason") or "").lower()


# ---------------------------------------------------------------------------
# End-to-end stress-CSV assertions available after Commit 4 (leakage detection)
# ---------------------------------------------------------------------------

@_needs_stress
def test_stress_leakage_proxy_flagged_as_near_copy_artifact_not_committed():
    claims, _counts, plan = _run_stress()

    near_copies = [c for c in claims
                   if (c.get("finding_detail", {}) or {}).get("artifact_kind") == "near_duplicate_copy"]
    assert near_copies, "output_linear/leakage_proxy near-copy must be detected"
    leak = near_copies[0]
    assert leak["verdict"] == "artifact"
    cols = set(leak["finding_detail"].get("artifact_warning", {}).get("columns", []))
    assert cols == {"output_linear", "leakage_proxy"}
    warn = leak["finding_detail"]["artifact_warning"]
    assert warn["type"] == "target_leakage_near_copy"
    assert warn["leakage_risk"] == "high"
    assert warn["correlation"] > 0.999 and warn["residual_variance_ratio"] < 0.01

    # It must NOT also be mined and committed as an ordinary linear discovery.
    mined = [c for c in claims
             if c["finding_type"] == "robust_relation"
             and "output_linear" in c["canonical_text"] and "leakage_proxy" in c["canonical_text"]
             and "linear association" in c["canonical_text"]]
    assert not mined, "the near-copy pair must not be committed as a linear discovery"


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
