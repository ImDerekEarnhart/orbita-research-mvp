"""Regression tests for the killed-candidate classification overhaul.

Root cause (fixed here): ``orbita_discovery.core.resolve_status`` collapsed
*any* killed falsifier into a single "refuted" bucket, with no regard for
sample size, effect direction, or whether an alternative functional form of
the same relationship survived. That conflated three different outcomes:

  1. genuine contradiction (score worse than a trivial baseline)
  2. "didn't clear the bar" (score below threshold but non-negative)
  3. an unreliable verdict from a too-small held-out/cross-seed partition

These tests run the real datasets that exposed the problem end to end and
pin the corrected, honest classification:

* Titanic (891 rows, real file): group-difference candidates that merely
  fall short of the R² threshold (e.g. Pclass x Survived) must be
  "not_supported", never "rejected" — the evidence does not contradict the
  hypothesis, it just isn't strong enough to clear the bar.
* Animal allometry (20 rows, real file): with a confirmation partition of
  only ~5 rows, held-out/cross-seed scores are too noisy to trust. Killed
  candidates must be "inconclusive", never "rejected" — the sample size
  itself invalidates the verdict, independent of what the score shows.
* Every not_supported/inconclusive/rejected claim must carry a metric name,
  a held-out score, the sample size that score was computed on, a full-data
  diagnostic score, and a plain-language rejection reason — so the number on
  screen is never an unlabeled, unexplained "0.000".
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from orbita_mvp import ResearchMVP
from orbita_mvp.semantics import (
    _strip_transform_prefix,
    apply_functional_form_overrides,
    classify_pairwise_finding,
)

TITANIC_CSV = Path(r"C:\Users\Dereks\Desktop\TitanicData\train.csv")
ALLOMETRY_CSV = Path(r"C:\Users\Dereks\Downloads\animal_allometry.csv")


def _run_case(csv_path: Path, name: str):
    with tempfile.TemporaryDirectory() as td:
        svc = ResearchMVP(Path(td) / "o.db", Path(td) / "ws")
        try:
            case = svc.create_case(name=name, goal="")
            svc.add_file(case["id"], csv_path)
            plan = svc.compile_case(case["id"])
            svc.approve_plan(plan["id"], reviewer="tester")
            run = svc.run_case(case["id"], plan_id=plan["id"])
            assert run["status"] == "completed"
            claims = svc.store.case_claims(case["id"])
            counts = svc.store.case_claim_counts(case["id"])
            return claims, counts
        finally:
            svc.close()


def _run_case_with_report(csv_path: Path, name: str):
    """Like _run_case, but also returns the generated markdown report text."""
    with tempfile.TemporaryDirectory() as td:
        svc = ResearchMVP(Path(td) / "o.db", Path(td) / "ws")
        try:
            case = svc.create_case(name=name, goal="")
            svc.add_file(case["id"], csv_path)
            plan = svc.compile_case(case["id"])
            svc.approve_plan(plan["id"], reviewer="tester")
            run = svc.run_case(case["id"], plan_id=plan["id"])
            assert run["status"] == "completed"
            claims = svc.store.case_claims(case["id"])
            report_path = Path(run["result"]["reports"]["markdown"]["path"])
            report_text = report_path.read_text(encoding="utf-8")
            return claims, report_text
        finally:
            svc.close()


# ---------------------------------------------------------------------------
# 1. Unit tests for the classifier itself (no dataset dependency)
# ---------------------------------------------------------------------------

def _finding(killed_checks: list[dict]) -> dict:
    return {
        "candidate": {"id": "linear:x:y", "payload": {"kind": "linear_association", "predictor": "x", "outcome": "y"}},
        "falsifications": [
            {"name": c["name"], "killed": True, "detail": c["detail"]}
            for c in killed_checks
        ],
    }


def test_small_partition_is_inconclusive_regardless_of_score():
    # A held-out score that would otherwise pass (0.9) must still be
    # reclassified as inconclusive if the test partition is too small to
    # trust — sample size invalidates the verdict before the score is even
    # considered.
    finding = _finding([{"name": "held_out", "detail": {"score": 0.9, "n": 3}}])
    ftype, diag = classify_pairwise_finding(finding, min_reliable_partition_n=8)
    assert ftype == "inconclusive_candidate"
    assert diag["smallest_test_partition_n"] == 3


def test_lone_negative_held_out_score_is_not_supported_not_refuted():
    # A negative score from a SINGLE fixed split (held_out always uses
    # seed=1) must not, by itself, be treated as evidence the hypothesis is
    # wrong — one influential point landing in that one split is noise.
    # Without cross_seed corroborating the negative result, this must fall
    # to not_supported, never refuted.
    finding = _finding([{"name": "held_out", "detail": {"score": -0.4, "n": 200}}])
    ftype, diag = classify_pairwise_finding(finding, min_reliable_partition_n=8)
    assert ftype == "not_supported_candidate"
    assert "not a persistent negative signal" in diag["reason"]


def test_lone_negative_baseline_score_is_not_supported_not_refuted():
    finding = _finding([{"name": "baseline", "detail": {"score": -0.3, "baseline": 0.0, "n": 200}}])
    ftype, diag = classify_pairwise_finding(finding, min_reliable_partition_n=8)
    assert ftype == "not_supported_candidate"


def test_persistent_negative_cross_seed_on_generic_claim_is_functional_form_rejected():
    # cross_seed aggregates across 9 reseeds/resamples — a negative MEDIAN
    # here reflects a pattern that survives repeated resampling: the tested
    # MODEL FORM performed below baseline. For a generic, non-directional,
    # non-predictive-performance claim (the default), that must NOT
    # automatically refute the underlying relationship — it means this
    # functional form was the wrong shape, which is functional_form_rejected.
    finding = _finding([{"name": "cross_seed", "detail": {"median": -0.25, "spread": 0.1, "n": 200}}])
    ftype, diag = classify_pairwise_finding(finding, min_reliable_partition_n=8)
    assert ftype == "functional_form_rejected_candidate"
    assert diag["cross_seed_median"] == -0.25


def test_composite_killed_by_non_pairwise_check_on_tiny_sample_is_inconclusive():
    # Composite candidates are killed by ImprovementFalsifier/AblationFalsifier
    # — names outside baseline/held_out/cross_seed — but those checks score on
    # the exact same tiny confirmation partition. A composite killed only by
    # "improvement" with n below the reliability floor must still be
    # inconclusive, not blindly refuted just because its own falsifier name
    # isn't one of the three core pairwise checks.
    finding = {
        "candidate": {"id": "composite:y:abc", "payload": {"kind": "composite_linear", "predictors": ["a", "b"], "outcome": "y"}},
        "falsifications": [
            {"name": "improvement", "killed": True, "detail": {"composite_score": -0.2, "n": 5}},
        ],
    }
    ftype, diag = classify_pairwise_finding(
        finding, min_reliable_partition_n=8, is_explicit_predictive_claim=True
    )
    assert ftype == "inconclusive_candidate"
    assert diag["smallest_test_partition_n"] == 5


def test_composite_killed_by_non_pairwise_check_on_large_sample_is_refuted():
    # Same shape, but the sample is large enough to trust: an explicit
    # predictive claim (composite) killed by improvement/ablation on a
    # reliable sample IS refuted — those falsifiers directly test "did this
    # predict better than baseline," which is exactly what the claim asserts.
    finding = {
        "candidate": {"id": "composite:y:abc", "payload": {"kind": "composite_linear", "predictors": ["a", "b"], "outcome": "y"}},
        "falsifications": [
            {"name": "improvement", "killed": True, "detail": {"composite_score": -0.2, "n": 200}},
        ],
    }
    ftype, diag = classify_pairwise_finding(
        finding, min_reliable_partition_n=8, is_explicit_predictive_claim=True
    )
    assert ftype == "falsified_candidate"


def test_stable_opposite_direction_against_directional_claim_is_refuted():
    # The candidate asserts a specific direction (e.g. "a stable positive
    # linear association"). If the fitted relationship is stably in the
    # OPPOSITE direction, that is a failed directional prediction — direct
    # contradiction of the literal claim, not just an unhelpful model shape.
    finding = _finding([{"name": "cross_seed", "detail": {"median": -0.3, "spread": 0.1, "n": 200}}])
    ftype, diag = classify_pairwise_finding(
        finding, min_reliable_partition_n=8, direction_conflict=True
    )
    assert ftype == "falsified_candidate"
    assert "opposite direction" in diag["reason"]


def test_explicit_predictive_claim_below_baseline_is_refuted():
    # A composite candidate's own statement asserts "Y can be predicted by
    # a composite of [...]" — an explicit claim of predictive performance
    # above baseline. Persistently scoring below baseline directly
    # contradicts that literal claim, so this is refuted, not merely a
    # rejected functional form.
    finding = _finding([{"name": "cross_seed", "detail": {"median": -0.15, "spread": 0.05, "n": 200}}])
    ftype, diag = classify_pairwise_finding(
        finding, min_reliable_partition_n=8, is_explicit_predictive_claim=True
    )
    assert ftype == "falsified_candidate"
    assert "asserts predictive performance" in diag["reason"]
    assert diag["is_predictive_claim"] is True


def test_functional_form_rejection_links_to_surviving_log_log_sibling():
    # End-to-end of the exact scenario the fix targets: a raw-linear
    # candidate with persistent negative cross-seed performance (generic,
    # non-directional, non-predictive claim) becomes functional_form_rejected
    # on its own; when a transformed sibling of the same variables also
    # survived in the same run, the override pass links the two together.
    raw_finding_dict = {
        "candidate": {
            "id": "linear:body_mass:brain_mass",
            "statement": "body_mass and brain_mass show a stable positive linear association.",
            "payload": {"kind": "linear_association", "predictor": "body_mass", "outcome": "brain_mass"},
        },
        "falsifications": [
            {"name": "cross_seed", "killed": True, "detail": {"median": -0.4, "spread": 0.2, "n": 50}},
        ],
    }
    raw_ftype, _diag = classify_pairwise_finding(raw_finding_dict, min_reliable_partition_n=8)
    assert raw_ftype == "functional_form_rejected_candidate"

    log_finding_dict = {
        "candidate": {
            "id": "linear:log_body_mass:log_brain_mass",
            "payload": {"kind": "linear_association", "predictor": "log_body_mass", "outcome": "log_brain_mass"},
        },
    }
    overrides = apply_functional_form_overrides([
        (raw_finding_dict, raw_ftype),
        (log_finding_dict, "robust_relation"),
    ])
    assert "linear:body_mass:brain_mass" in overrides
    new_type, extra = overrides["linear:body_mass:brain_mass"]
    assert new_type == "functional_form_rejected_candidate"
    assert extra["alternative_candidate_id"] == "linear:log_body_mass:log_brain_mass"


def test_negative_held_out_plus_positive_cross_seed_stays_not_supported():
    # held_out (one split) shows negative, but cross_seed (9 resamples)
    # shows the median is actually fine — the single split was the outlier,
    # not the relationship. Must not be refuted.
    finding = _finding([
        {"name": "held_out", "detail": {"score": -0.4, "n": 200}},
        {"name": "cross_seed", "detail": {"median": 0.08, "spread": 0.3, "n": 200}},
    ])
    ftype, diag = classify_pairwise_finding(finding, min_reliable_partition_n=8)
    assert ftype == "not_supported_candidate"


def test_below_threshold_but_nonnegative_is_not_supported():
    finding = _finding([{"name": "held_out", "detail": {"score": 0.05, "n": 200}}])
    ftype, diag = classify_pairwise_finding(finding, min_reliable_partition_n=8)
    assert ftype == "not_supported_candidate"


def test_strip_transform_prefix():
    assert _strip_transform_prefix("log_body_mass") == "body_mass"
    assert _strip_transform_prefix("Log_Body_Mass") == "Body_Mass"
    assert _strip_transform_prefix("log10_x") == "x"
    assert _strip_transform_prefix("body_mass") == "body_mass"


def test_functional_form_override_links_raw_to_surviving_log_sibling():
    raw_finding = {
        "candidate": {
            "id": "linear:body_mass:brain_mass",
            "payload": {"kind": "linear_association", "predictor": "body_mass", "outcome": "brain_mass"},
        }
    }
    log_finding = {
        "candidate": {
            "id": "linear:log_body_mass:log_brain_mass",
            "payload": {"kind": "linear_association", "predictor": "log_body_mass", "outcome": "log_brain_mass"},
        }
    }
    overrides = apply_functional_form_overrides([
        (raw_finding, "not_supported_candidate"),
        (log_finding, "robust_relation"),
    ])
    assert "linear:body_mass:brain_mass" in overrides
    new_type, extra = overrides["linear:body_mass:brain_mass"]
    assert new_type == "functional_form_rejected_candidate"
    assert extra["alternative_candidate_id"] == "linear:log_body_mass:log_brain_mass"
    # The surviving candidate itself must not be touched.
    assert "linear:log_body_mass:log_brain_mass" not in overrides


def test_no_override_when_no_sibling_survived():
    raw_finding = {
        "candidate": {
            "id": "linear:body_mass:brain_mass",
            "payload": {"kind": "linear_association", "predictor": "body_mass", "outcome": "brain_mass"},
        }
    }
    log_finding = {
        "candidate": {
            "id": "linear:log_body_mass:log_brain_mass",
            "payload": {"kind": "linear_association", "predictor": "log_body_mass", "outcome": "log_brain_mass"},
        }
    }
    overrides = apply_functional_form_overrides([
        (raw_finding, "inconclusive_candidate"),
        (log_finding, "inconclusive_candidate"),  # sibling did NOT survive either
    ])
    assert overrides == {}


# ---------------------------------------------------------------------------
# 2. End-to-end: Titanic (891 rows) — group tests must not be "rejected"
# ---------------------------------------------------------------------------

def test_titanic_group_differences_are_not_supported_not_rejected():
    claims, counts = _run_case(TITANIC_CSV, "Titanic classification regression")

    # Nothing in this run should be labeled hard-refuted: every group
    # candidate that missed the bar did so with a small positive (not
    # negative) score, which is "not_supported", not "rejected".
    assert counts["rejected_count"] == 0

    group_claims = [c for c in claims if "differs systematically" in c["canonical_text"]]
    assert group_claims, "expected group-difference candidates to be generated"
    for c in group_claims:
        assert c["finding_type"] == "not_supported_candidate", c["canonical_text"]
        assert c["verdict"] == "not_supported"
        detail = c["finding_detail"]
        assert detail["metric_name"] == "r2"
        assert detail["held_out_score"] is not None
        assert detail["held_out_n"] is not None and detail["held_out_n"] > 100
        assert detail["full_data_score_diagnostic"] is not None
        assert detail["rejection_reason"], "not_supported claim must explain why"

    # The strong real relationship (Pclass x Fare) must still survive.
    assert counts["committed_count"] >= 1
    committed_texts = {c["canonical_text"] for c in claims if c["finding_type"] == "robust_relation"}
    assert any("Pclass" in t and "Fare" in t for t in committed_texts)


# ---------------------------------------------------------------------------
# 3. End-to-end: animal allometry (20 rows) — killed candidates must be
#    "inconclusive", never a confident "rejected", because the held-out
#    partition (~5 rows) is too small to trust.
# ---------------------------------------------------------------------------

def test_allometry_small_sample_candidates_are_inconclusive_not_rejected():
    claims, counts = _run_case(ALLOMETRY_CSV, "Allometry classification regression")

    assert counts["rejected_count"] == 0, (
        "no candidate should be confidently 'refuted' on a 20-row dataset with "
        "a ~5-row held-out partition — that partition size cannot support the claim"
    )
    assert counts["inconclusive_count"] > 0, "small-sample candidates must be flagged inconclusive"

    inconclusive_claims = [c for c in claims if c["finding_type"] == "inconclusive_candidate"]
    for c in inconclusive_claims:
        detail = c["finding_detail"]
        assert detail["rejection_reason"], "inconclusive claim must explain why"
        assert "row" in detail["rejection_reason"].lower() or "partition" in detail["rejection_reason"].lower()
        n = detail.get("held_out_n") or detail.get("baseline_n")
        assert n is not None and n < 8

    # At least the strongest relationship in this dataset must still survive
    # despite the small sample — the gate should not swallow everything.
    assert counts["committed_count"] >= 1


def test_allometry_every_killed_claim_has_full_diagnostic_fields():
    claims, _counts = _run_case(ALLOMETRY_CSV, "Allometry diagnostics regression")
    killed_types = {"falsified_candidate", "not_supported_candidate", "inconclusive_candidate", "functional_form_rejected_candidate"}
    killed = [c for c in claims if c["finding_type"] in killed_types]
    assert killed, "expected at least one killed candidate in this run"
    for c in killed:
        detail = c["finding_detail"]
        assert detail.get("metric_name"), c["canonical_text"]
        assert detail.get("rejection_reason"), c["canonical_text"]
        # held_out_score may be None only if the candidate never reached that
        # check, but at least one score/n pair must be present.
        assert detail.get("held_out_score") is not None or detail.get("baseline_score") is not None


# ---------------------------------------------------------------------------
# 3. The HTML/markdown report must match the claims API, not the raw engine's
#    collapsed final_status. ReportCompiler.build_markdown previously read
#    finding.get("final_status") directly from the raw ledger findings,
#    bypassing the enriched claim_rows entirely — so a not_supported claim
#    (correct everywhere else) still printed "refuted" in the report.
# ---------------------------------------------------------------------------

def test_report_uses_enriched_verdict_not_raw_final_status():
    claims, report_text = _run_case_with_report(TITANIC_CSV, "Titanic report consistency regression")

    not_supported = [c for c in claims if c["finding_type"] == "not_supported_candidate"]
    assert not_supported, "expected at least one not_supported claim in this run"

    for c in not_supported:
        text = c["canonical_text"]
        # Find this candidate's line in the "failed" section of the report
        # and confirm it prints the enriched verdict, not the raw engine
        # status. Every not_supported claim must never appear tagged
        # `refuted` in the report.
        idx = report_text.find(text)
        assert idx != -1, f"candidate statement not found in report: {text}"
        line_end = report_text.find("\n", idx)
        line = report_text[idx:line_end if line_end != -1 else None]
        assert "verdict `not_supported`" in line, (
            f"report line for a not_supported claim must say so, got: {line!r}"
        )
        assert "`refuted`" not in line, f"not_supported claim incorrectly shown as refuted: {line!r}"
