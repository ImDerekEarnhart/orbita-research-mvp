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

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from orbita_mvp import ResearchMVP

STRESS_CSV = Path(r"C:\Users\Dereks\Downloads\orbita_pre_release_stress_test.csv")
TCELL_CSV = Path(r"C:\Users\Dereks\Downloads\tcell_orbita_blind_challenge.csv")
CRUCIBLE_D_CSV = Path(r"C:\Users\Dereks\Downloads\orbita_crucible_v1_discovery_files\crucible_task_d_discovery.csv")

_needs_stress = pytest.mark.skipif(
    not STRESS_CSV.exists(), reason="stress-test CSV not available on this machine"
)


def _run_csv(csv_path: Path):
    with tempfile.TemporaryDirectory() as td:
        svc = ResearchMVP(Path(td) / "o.db", Path(td) / "ws")
        try:
            case = svc.create_case(name=csv_path.stem, goal="")
            svc.add_file(case["id"], csv_path)
            plan = svc.compile_case(case["id"])
            svc.approve_plan(plan["id"], reviewer="tester")
            svc.run_case(case["id"], plan_id=plan["id"])
            return svc.store.case_claims(case["id"]), plan["plan"]
        finally:
            svc.close()


def _derived_clusters(claims):
    return [c for c in claims if (c.get("finding_detail", {}) or {}).get("artifact_kind") == "likely_derived_variable"]


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


# ===========================================================================
# Commit 6 — general fixes (A multivariable derived, B budgeting, C temporal,
# D artifact propagation).
# ===========================================================================

# --- A: multivariable noisy-derived-variable detection ---------------------

def test_multivariable_derived_flags_constructed_index_not_genuine_relationship():
    from orbita_mvp.derived import detect_multivariable_derived

    rng = np.random.default_rng(4)
    n = 400
    x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    # A near-deterministic 3-variable constructed index.
    idx = x1 + 2.0 * x2 - 0.5 * x3 + rng.normal(scale=0.01, size=n)
    # A GENUINE strong multivariable scientific relationship with real residual.
    y_real = x1 + x2 + rng.normal(scale=2.0, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "idx": idx, "y_real": y_real})
    scout, heldout = df.iloc[:250], df.iloc[250:]
    cols = ["x1", "x2", "x3", "idx", "y_real"]

    res = detect_multivariable_derived(scout, heldout, cols, cols)
    # The constructed index is flagged as near-deterministically reconstructed.
    assert "idx" in res
    assert res["idx"]["held_out_r2"] >= 0.99
    assert res["idx"]["n_predictors"] >= 2
    # The genuine noisy multivariable relationship is NOT auto-flagged.
    assert "y_real" not in res


@pytest.mark.skipif(not TCELL_CSV.exists(), reason="T-cell CSV not available")
def test_tcell_adenosine_pressure_is_artifact_qualified():
    claims, _plan = _run_csv(TCELL_CSV)
    clusters = _derived_clusters(claims)
    named = set()
    for c in clusters:
        named.update(c["finding_detail"]["artifact_warning"]["member_columns"])
    assert "adenosine_pressure" in named, "adenosine_pressure must be artifact-qualified"
    # A derived cluster verdict is an artifact, with direction stated as undetermined.
    for c in clusters:
        assert c["verdict"] == "artifact"
        assert c["finding_detail"]["artifact_warning"]["derivation_direction"] == "undetermined"


@pytest.mark.skipif(not CRUCIBLE_D_CSV.exists(), reason="Crucible D CSV not available")
def test_crucible_d_derived_index_is_artifact_qualified():
    claims, _plan = _run_csv(CRUCIBLE_D_CSV)
    named = set()
    for c in _derived_clusters(claims):
        named.update(c["finding_detail"]["artifact_warning"]["member_columns"])
    assert "derived_index" in named, "derived_index must be artifact-qualified"
    # The planted null must never be pulled into a derived cluster or committed.
    assert "null_chemistry" not in named
    assert not [c for c in claims if "null_chemistry" in c["canonical_text"].lower() and c["verdict"] == "committed"]


# --- D1/D2 display-only: cluster public label + drawer fields --------------

def test_derived_cluster_public_label_and_drawer_fields_display_only():
    from orbita_mvp.graph_ui import build_graph_data, DERIVED_CLUSTER_LABEL

    rng = np.random.default_rng(4)
    n = 400
    x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    idx = x1 + 2.0 * x2 - 0.5 * x3 + rng.normal(scale=0.01, size=n)  # constructed index
    y = 1.5 * x1 + rng.normal(scale=2.0, size=n)                     # genuine, noisy
    df = pd.DataFrame({"row_id": range(n), "x1": x1, "x2": x2, "x3": x3, "idx": idx, "y": y})

    with tempfile.TemporaryDirectory() as td:
        svc = ResearchMVP(Path(td) / "o.db", Path(td) / "ws")
        try:
            p = Path(td) / "d.csv"; df.to_csv(p, index=False)
            case = svc.create_case(name="cluster", goal="")
            svc.add_file(case["id"], p); svc.compile_case(case["id"]); svc.run_case(case["id"], auto_approve=True)
            claims = svc.store.case_claims(case["id"])
            clusters = [c for c in claims if (c.get("finding_detail", {}) or {}).get("artifact_kind") == "likely_derived_variable"]
            assert clusters, "expected a near-deterministic dependency cluster"
            cl = clusters[0]; fd = cl["finding_detail"]; aw = fd["artifact_warning"]

            # Display-only: stored verdict + finding_type are UNCHANGED.
            assert cl["verdict"] == "artifact"
            assert cl["finding_type"] == "artifact"

            # Drawer fields (D2) all persisted.
            assert aw["type"] == "likely_derived_variable"
            assert aw["derivation_direction"] == "undetermined"
            assert "idx" in aw["member_columns"]
            br = aw["best_reconstruction"]
            assert br["reconstruction_metric"] == "held_out_r2"
            assert br["held_out_r2"] is not None
            assert br["residual_variance_ratio"] is not None
            assert br["valid_refit_count"] is not None and br["refit_attempts"] is not None

            # Public label (D1) applied to the graph node; stored statement (full_text) unchanged.
            g = build_graph_data(case["id"], svc.ledger.db.conn)
            node = next(nd for nd in g["nodes"] if nd.get("id") == cl["claim_id"])
            assert node["label"] == DERIVED_CLUSTER_LABEL
            assert node["display_label"] == DERIVED_CLUSTER_LABEL
            assert "near-deterministic" in node["full_text"].lower()  # underlying statement intact
            assert node["artifact_warning"]["derivation_direction"] == "undetermined"

            # Non-cluster claim labels are NOT relabelled.
            others = [nd for nd in g["nodes"] if nd.get("type") == "claim" and nd.get("id") != cl["claim_id"]]
            assert all(nd["label"] != DERIVED_CLUSTER_LABEL for nd in others)
        finally:
            svc.close()


# --- Near-exact accounting identity (derived-field) detection --------------

def test_near_exact_accounting_identity_flagged_but_weighted_composite_is_not():
    from orbita_mvp.artifacts import detect_structural_relations

    rng = np.random.default_rng(0)
    n = 300
    a = rng.uniform(10, 100, n)
    b = rng.uniform(1, 20, n)
    # A near-exact UNIT-coefficient accounting identity (a - b + tiny noise).
    identity = a - b + rng.normal(0, 0.01, n)
    # A genuine WEIGHTED composite (fitted coefficients) -- must NOT be flagged.
    composite = 5.0 * a + 5.0 * b + rng.normal(0, 0.02, n)
    df = pd.DataFrame({"a": a, "b": b, "identity": identity, "composite": composite})
    rel = detect_structural_relations(df, numeric_columns=["a", "b", "identity", "composite"])
    kinds = {v.get("kind") for v in rel.values()}
    cols_flagged = set()
    for v in rel.values():
        cols_flagged.update(v.get("inputs", []) or [])
        cols_flagged.update(c for c in (v.get("columns") or []))
    # The unit identity is caught as a (near-)derived field...
    derived = [v for v in rel.values() if v.get("kind") in ("derived_field", "near_derived_field")
               and "identity" in (v.get("columns") or [])]
    assert derived, "near-exact unit accounting identity should be flagged as derived"
    # ...but the weighted composite is left mineable, not flagged as derived.
    comp_derived = [v for v in rel.values() if v.get("kind") in ("derived_field", "near_derived_field")
                    and "composite" in (v.get("columns") or [])]
    assert not comp_derived, "a weighted composite (slope != 1) must not be flagged as a derived field"


# --- B: candidate-family budgeting -----------------------------------------

def test_nonlinear_budget_does_not_crowd_out_linear_pairs():
    from orbita_mvp.table_domain import generate_table_candidates

    rng = np.random.default_rng(7)
    n = 300
    base = rng.normal(size=n)
    data = {f"v{i}": base * (0.5 + 0.15 * i) + rng.normal(scale=1.0, size=n) for i in range(8)}
    df = pd.DataFrame(data)

    # Linear-only baseline (nonlinear budget 0) under a small saturated cap.
    lin_only, _ = generate_table_candidates(df, max_candidates=10, nonlinear_budget=0)
    linear_ids_only = {c["id"] for c in lin_only if c["kind"] == "linear_association"}

    # Same cap, but with nonlinear search enabled.
    full, _ = generate_table_candidates(df, max_candidates=10, nonlinear_budget=20)
    linear_ids_full = {c["id"] for c in full if c["kind"] == "linear_association"}

    # No legitimate linear pair is dropped solely because nonlinear search was added.
    assert linear_ids_only <= linear_ids_full
    # Nonlinear candidates come from a SEPARATE budget (added on top).
    assert any(c["kind"] == "nonlinear_association" for c in full)
    assert len(full) > len(lin_only)


# --- Categorical-aware multivariable dependency clusters (HW-05) ------------

def test_dependency_cluster_with_numeric_and_categorical_term():
    from orbita_mvp.derived import detect_multivariable_derived

    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    cat = rng.choice(["A", "B"], n)
    # A constructed index that depends on numeric terms AND a binary/categorical term.
    idx = 2.0 * x1 + 3.0 * x2 + 5.0 * (cat == "B") + rng.normal(0, 0.01, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "idx": idx})
    scout, heldout = df.iloc[:250], df.iloc[250:]

    res = detect_multivariable_derived(
        scout, heldout, ["x1", "x2", "idx"], ["x1", "x2", "idx"], categorical_columns=["cat"]
    )
    assert "idx" in res, "a constructed index depending on a categorical term must be detected"
    assert "cat" in res["idx"]["source_variables"], "the categorical member must be named (mapped from its dummy)"
    assert res["idx"]["held_out_r2"] >= 0.99


def test_ordinary_group_difference_not_flagged_as_dependency_cluster():
    from orbita_mvp.derived import detect_multivariable_derived

    rng = np.random.default_rng(1)
    n = 400
    g = rng.choice(["A", "B", "C"], n)
    base = {"A": 1.0, "B": 5.0, "C": 9.0}
    y = np.array([base[gg] for gg in g]) + rng.normal(0, 2.0, n)   # noisy group effect
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    df = pd.DataFrame({"g": g, "y": y, "x1": x1, "x2": x2})
    scout, heldout = df.iloc[:250], df.iloc[250:]

    res = detect_multivariable_derived(
        scout, heldout, ["y", "x1", "x2"], ["y", "x1", "x2"], categorical_columns=["g"]
    )
    assert "y" not in res, "an ordinary (noisy) group difference must NOT be flagged as a dependency cluster"


# --- Informative missingness (MNAR) diagnostic ------------------------------

def test_informative_missingness_detects_mnar_not_mcar():
    from orbita_mvp.missingness import detect_informative_missingness

    rng = np.random.default_rng(0)
    n = 500
    z = rng.uniform(0, 10, n)
    x = 2.0 * z + rng.normal(0, 1, n)
    x[z > 7] = np.nan                      # MNAR: x missing when z is high
    y = rng.normal(size=n)
    y[rng.random(n) < 0.2] = np.nan        # MCAR: missing at random
    df = pd.DataFrame({"z": z, "x": x, "y": y, "w": rng.normal(size=n)})

    findings = detect_informative_missingness(df)
    flagged = {f["column"] for f in findings}
    assert "x" in flagged, "MNAR missingness (depends on z) must be flagged"
    assert "y" not in flagged, "MCAR (random) missingness must NOT be flagged"

    xf = next(f for f in findings if f["column"] == "x")["informative_missingness"]
    assert "z" in [p["predictor"] for p in xf["strongest_predictors"]]
    assert xf["effect_size"] > 0.2 and xf["missingness_rate"] > 0.05
    assert xf["n_present"] > 0 and xf["n_missing"] > 0
    assert xf["validation"]["folds_stable"] is True


def test_informative_missingness_e2e_downgrades_committed_to_provisional():
    # A moderately-missing numeric column whose missingness is informative should
    # (a) get an informative-missingness data-quality finding, and (b) have any
    # relationship on it reported provisional (with a warning), not committed.
    rng = np.random.default_rng(1)
    n = 500
    z = rng.uniform(0, 10, n)
    driver = rng.uniform(0, 10, n)
    meas = 3.0 * z + rng.normal(0, 0.5, n)   # a real strong relationship z -> meas
    meas[driver > 9] = np.nan                # ~10% MNAR: stays mineable (numeric_fraction ~0.9)
    df = pd.DataFrame({"row_id": range(n), "z": z, "driver": driver, "meas": meas})
    with tempfile.TemporaryDirectory() as td:
        svc = ResearchMVP(Path(td) / "o.db", Path(td) / "ws")
        try:
            p = Path(td) / "d.csv"; df.to_csv(p, index=False)
            c = svc.create_case(name="mnar", goal="")
            svc.add_file(c["id"], p); svc.compile_case(c["id"]); svc.run_case(c["id"], auto_approve=True)
            claims = svc.store.case_claims(c["id"])
        finally:
            svc.close()
    # (a) informative-missingness finding present for meas.
    im = [cl for cl in claims if (cl.get("finding_detail", {}) or {}).get("informative_missingness")]
    assert any(cl["finding_detail"]["informative_missingness"]["column"] == "meas" for cl in im)
    # (b) mined relationships on meas are provisional (not committed), with the warning.
    mined_types = {"robust_relation", "promising_candidate", "supported_association_candidate"}
    meas_rel = [cl for cl in claims if "meas" in cl["canonical_text"] and cl["finding_type"] in mined_types]
    assert meas_rel, "expected a relationship on meas to be mined"
    for cl in meas_rel:
        assert cl["verdict"] != "committed", "relationship on an MNAR column must not be committed"
        assert (cl["finding_detail"] or {}).get("informative_missingness_warning")


# --- Repeated-entity (grouping-key) role ------------------------------------

def test_repeated_entity_role_distinct_from_identifier_and_category():
    from orbita_mvp.ingestion import detect_repeated_entity, profile_dataframe

    n = 300
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "transition_id": range(n),                       # unique -> identifier
        "episode_id": np.repeat(np.arange(n // 6), 6),   # 50 entities x 6 rows -> repeated_entity
        "world_mode": rng.choice(["A", "B", "C"], n),    # low-cardinality category, NOT entity
        "value": rng.normal(size=n),
    })
    # Direct detector: episode_id is a repeated entity; world_mode is not.
    assert detect_repeated_entity(df["episode_id"], "episode_id") is not None
    assert detect_repeated_entity(df["world_mode"], "world_mode") is None
    # A non-unique column WITHOUT an entity-style name is not swept in.
    assert detect_repeated_entity(pd.Series(np.repeat(np.arange(50), 6), name="score_band"), "score_band") is None

    roles = {c["name"]: c["inferred_role"] for c in profile_dataframe(df)["column_profiles"]}
    assert roles["transition_id"] == "identifier"
    assert roles["episode_id"] == "repeated_entity"
    assert roles["world_mode"] == "group_or_category"


# --- C: temporal-axis detection --------------------------------------------

def test_temporal_axis_distinguished_from_identifier():
    from orbita_mvp.ingestion import detect_identifier, detect_temporal, profile_dataframe

    n = 300
    year = pd.Series(range(1700, 1700 + n), name="year")
    ids = detect_identifier(year, "year")
    assert detect_temporal(year, "year", ids) is not None  # temporal, not identifier

    sid = pd.Series([f"OBS-{i:04d}" for i in range(n)], name="sample_id")
    sid_ids = detect_identifier(sid, "sample_id")
    assert sid_ids is not None and detect_temporal(sid, "sample_id", sid_ids) is None  # stays identifier

    rown = pd.Series(range(n), name="row_number")
    rown_ids = detect_identifier(rown, "row_number")
    assert rown_ids is not None and detect_temporal(rown, "row_number", rown_ids) is None  # stays identifier

    df = pd.DataFrame({
        "year": range(1700, 1700 + n),
        "sample_id": [f"S{i:04d}" for i in range(n)],
        "val": np.random.default_rng(0).normal(size=n),
    })
    roles = {c["name"]: c["inferred_role"] for c in profile_dataframe(df)["column_profiles"]}
    assert roles["year"] == "temporal_index"
    assert roles["sample_id"] == "identifier"


def test_chronological_validation_orders_partitions_by_time():
    from orbita_mvp.table_domain import UploadedTableDomain

    n = 120
    rng = np.random.default_rng(1)
    x = np.arange(n) + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"year": range(2000, 2000 + n), "x": x, "y": 2.0 * x + rng.normal(scale=1.0, size=n)})
    spec = {
        "id": "linear:x_y", "statement": "x and y show a stable positive linear association.",
        "kind": "linear_association", "predictor": "x", "outcome": "y", "expected_direction": "positive",
    }
    dom = UploadedTableDomain(df, [spec], time_column="year")
    # Earliest rows train, latest rows are the held-out final validation.
    assert dom.scout["year"].max() < dom.selection["year"].min()
    assert dom.selection["year"].max() < dom.final_validation["year"].min()
    # Repeated-refit windows are also time-ordered (train precedes validation).
    train, val = dom.repeated_refit_split(0)
    assert train["year"].max() <= val["year"].min()


# --- D: artifact-column propagation (direction stated, source not contaminated) ---

@_needs_stress
def test_near_copy_direction_undetermined_and_source_not_contaminated():
    claims, _counts, _plan = _run_stress()
    near_copies = [c for c in claims
                   if (c.get("finding_detail", {}) or {}).get("artifact_kind") == "near_duplicate_copy"]
    assert near_copies
    assert near_copies[0]["finding_detail"]["artifact_warning"]["derivation_direction"] == "undetermined"
    # Because direction is undetermined, no ordinary finding is auto-marked
    # artifact_contaminated (the legitimate source column keeps its findings).
    contaminated = [c for c in claims
                    if (c.get("finding_detail", {}) or {}).get("artifact_warning", {}).get("type") == "artifact_contaminated"]
    assert not contaminated


# ---------------------------------------------------------------------------
# End-to-end cross-surface consistency (Commit 5): claims API, belief graph,
# report HTML, report JSON, and claim counts must agree on every verdict.
# ---------------------------------------------------------------------------

@_needs_stress
def test_stress_all_surfaces_agree_on_verdict():
    from collections import Counter

    from orbita_mvp.graph_ui import build_graph_data

    with tempfile.TemporaryDirectory() as td:
        svc = ResearchMVP(Path(td) / "o.db", Path(td) / "ws")
        try:
            case = svc.create_case(name="surface-consistency", goal="")
            svc.add_file(case["id"], STRESS_CSV)
            plan = svc.compile_case(case["id"])
            svc.approve_plan(plan["id"], reviewer="tester")
            run = svc.run_case(case["id"], plan_id=plan["id"])

            claims = svc.store.case_claims(case["id"])
            counts = svc.store.case_claim_counts(case["id"])
            graph = build_graph_data(case["id"], svc.ledger.db.conn)
            reports = run["result"]["reports"]
            report_json = json.loads(Path(reports["json"]["path"]).read_text(encoding="utf-8"))
            html_text = Path(reports["html"]["path"]).read_text(encoding="utf-8")
        finally:
            svc.close()

    graph_state = {n["id"]: n.get("public_state") for n in graph["nodes"] if n.get("type") == "claim"}
    json_verdict = {c["claim_id"]: c.get("verdict") for c in report_json["claims"]}

    for c in claims:
        cid, verdict = c["claim_id"], c["verdict"]
        # 1. belief graph node public_state == claims-API verdict
        assert graph_state.get(cid) == verdict, (cid, verdict, graph_state.get(cid))
        # 2. report JSON claim verdict == claims-API verdict
        assert json_verdict.get(cid) == verdict, (cid, verdict, json_verdict.get(cid))
        # 3. report HTML provenance table shows this claim with its verdict
        assert cid in html_text

    # 4. claim counts agree with the persisted claim tally
    tally = Counter(c["verdict"] for c in claims)
    assert counts["committed_count"] == tally.get("committed", 0)
    assert counts["supported_association_count"] == tally.get("supported_association", 0)
    assert counts["regime_dependent_count"] == tally.get("regime_dependent", 0)
    assert counts["artifact_count"] == tally.get("artifact", 0)

    # 5. the honest sections exist and there are no genuinely refuted claims here
    assert "Supported associations" in html_text
    assert "Regime-dependent" in html_text
    assert tally.get("rejected", 0) == 0
    # exactly one appearance of "refuted": the executive summary's "0 refuted".
    assert html_text.lower().count("refuted") == 1
    assert "0 refuted" in html_text


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
