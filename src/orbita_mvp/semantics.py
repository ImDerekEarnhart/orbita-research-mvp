"""Public epistemic semantics for case findings.

The raw `claims.status` column reflects the internal support-engine lifecycle
(every claim is born ``provisional`` and only flips to ``committed`` when a
support chain closes). That value is **not** a faithful public verdict: a
refuted candidate keeps refute-only evidence yet never leaves ``provisional``,
so reading `claims.status` makes a falsified hypothesis look unresolved-but-alive.

This module derives the public verdict from the *finding type* the discovery
pipeline assigned, which already encodes whether the hypothesis survived
falsification, was rejected, or is a structural artifact. It also pulls the
hypothesis text and the individual check scores apart from the affirmative
candidate statement so a rejected finding is never displayed as if it were true.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Finding type -> public epistemic state
# ---------------------------------------------------------------------------

# Public states surfaced to API consumers and the dashboard.
COMMITTED = "committed"
REJECTED = "rejected"
ARTIFACT = "artifact"
PROVISIONAL = "provisional"
UNRESOLVED = "unresolved"
# A falsifier score fell below the support threshold, but did not show
# evidence actively worse than a trivial baseline. Distinct from REJECTED:
# "insufficient support" is not the same claim as "contradicted".
NOT_SUPPORTED = "not_supported"
# The held-out or cross-seed test partition was too small for its score to
# be trustworthy (a handful of rows can flip R² by more than a full point).
# Distinct from UNRESOLVED: the candidate *could* be tested, the result is
# just too noisy to act on with this much data.
INCONCLUSIVE = "inconclusive"
# The tested functional form (e.g. a raw linear fit) failed, but a
# transformed version of the same predictor/outcome pair (e.g. log-log)
# survived. The underlying relationship is not refuted — the specific model
# shape tried here was the wrong one.
FUNCTIONAL_FORM_REJECTED = "functional_form_rejected"
# A real, stable group/variable association (meaningful effect size, stable
# under bootstrap) that nonetheless does NOT clear the standalone predictive
# bar. Distinct from both COMMITTED (which asserts predictive utility) and
# NOT_SUPPORTED (which found no clearing evidence at all): the association is
# supported; its standalone predictive utility is limited.
SUPPORTED_ASSOCIATION = "supported_association"
# The pooled relationship reverses (or materially changes) inside identifiable
# subgroups, so no universal directional claim can be committed. Scoped
# per-group claims are recorded instead.
REGIME_DEPENDENT = "regime_dependent"

PUBLIC_STATES = {
    COMMITTED, REJECTED, ARTIFACT, PROVISIONAL, UNRESOLVED,
    NOT_SUPPORTED, INCONCLUSIVE, FUNCTIONAL_FORM_REJECTED,
    SUPPORTED_ASSOCIATION, REGIME_DEPENDENT,
}

# Canonical mapping required by the spec. Legacy finding-type spellings are
# retained so older rows in the persistent ledger still resolve correctly.
FINDING_TYPE_TO_STATE: dict[str, str] = {
    # Survived all falsifiers and committed.
    "robust_relation": COMMITTED,
    # Killed by at least one falsifier with evidence actively against it
    # (e.g. held-out R² < 0 — worse than predicting the mean).
    "falsified_candidate": REJECTED,
    # Killed by at least one falsifier, but the score simply didn't clear
    # the bar rather than showing evidence against the hypothesis.
    "not_supported_candidate": NOT_SUPPORTED,
    # Killed, but on a test partition too small to trust the verdict.
    "inconclusive_candidate": INCONCLUSIVE,
    # Killed, but a transformed sibling candidate (same columns, different
    # functional form) survived — the relationship exists, this shape of it
    # doesn't.
    "functional_form_rejected_candidate": FUNCTIONAL_FORM_REJECTED,
    # A real association with limited standalone predictive utility.
    "supported_association_candidate": SUPPORTED_ASSOCIATION,
    # A within-subgroup scoped association recorded when the pooled claim is
    # blocked by a subgroup reversal.
    "scoped_association": SUPPORTED_ASSOCIATION,
    # Pooled relationship reverses inside subgroups.
    "regime_dependent_candidate": REGIME_DEPENDENT,
    "subgroup_reversal_candidate": REGIME_DEPENDENT,
    # Structural / transform tautology, not a scientific hypothesis.
    "artifact": ARTIFACT,
    "structural_relation": ARTIFACT,
    # Survived but did not reach the commit bar.
    "promising_candidate": PROVISIONAL,
    "candidate_relation": PROVISIONAL,  # legacy spelling
    # Could not be tested (insufficient data, undefined metric).
    "untestable_candidate": UNRESOLVED,
    "unresolved_candidate": UNRESOLVED,  # legacy spelling
    # Deterministic data-profile findings keep their own bucket as artifacts.
    "data_quality": ARTIFACT,
    "data_error": ARTIFACT,
    "artifact_guard": ARTIFACT,
}

# Colors the dashboard uses per public state.
STATE_COLOR = {
    COMMITTED: "green",
    REJECTED: "red",
    ARTIFACT: "orange",
    PROVISIONAL: "yellow",
    UNRESOLVED: "gray",
    NOT_SUPPORTED: "slate",
    INCONCLUSIVE: "blue-gray",
    FUNCTIONAL_FORM_REJECTED: "amber",
    SUPPORTED_ASSOCIATION: "teal",
    REGIME_DEPENDENT: "purple",
}


def public_state(finding_type: str | None) -> str:
    """Map an internal finding type to its public epistemic state.

    Unknown finding types are conservatively reported as ``unresolved`` rather
    than silently treated as committed.
    """
    if not finding_type:
        return UNRESOLVED
    return FINDING_TYPE_TO_STATE.get(finding_type, UNRESOLVED)


def is_rejected(finding_type: str | None) -> bool:
    return public_state(finding_type) == REJECTED


# ---------------------------------------------------------------------------
# Killed-candidate classification: refuted vs. not_supported vs. inconclusive
# ---------------------------------------------------------------------------
#
# The generic discovery engine (orbita_discovery.core.resolve_status) collapses
# every killed falsifier into one bucket: "refuted". That conflates three very
# different outcomes:
#   1. The evidence actively contradicts the hypothesis (score worse than a
#      trivial baseline — e.g. held-out R² < 0).
#   2. The evidence just didn't clear the support bar (score is non-negative
#      but below the threshold).
#   3. The test partition was too small for the score to mean anything (a
#      handful of held-out rows can swing R² by more than a full point,
#      especially with heavy-tailed data).
#
# This module re-derives a finer verdict from the same falsification detail
# dicts (which now report "n", the test-partition size used) without
# changing the generic engine's contract.

# The fixed-model validation-resample check is emitted under the honest name
# ``validation_resample``; ``cross_seed`` is the historical name kept as a
# recognised alias so old ledgers/plans/APIs still classify identically.
RESAMPLE_CHECK_NAMES = ("validation_resample", "cross_seed")

# Checks whose failure is a *pairwise* fitness signal (not composite predictive
# machinery). ``repeated_refit`` is a diagnostic-only validator and is excluded
# so it never counts as a "predictive check" that could refute a candidate.
_PAIRWISE_CHECK_NAMES = ("baseline", "held_out", "validation_resample", "cross_seed", "repeated_refit")


def _check_detail(falsifications: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for attack in falsifications:
        if attack.get("name") == name:
            return attack.get("detail", {}) or {}
    return {}


def classify_pairwise_finding(
    finding: dict[str, Any],
    *,
    min_reliable_partition_n: int = 8,
    hard_refutation_score_ceiling: float = 0.0,
    is_explicit_predictive_claim: bool = False,
    direction_conflict: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Reclassify a killed finding into refuted / functional_form_rejected /
    not_supported / inconclusive.

    Returns ``(internal_finding_type, diagnostics)``. Only meaningful for
    findings where at least one falsifier reported ``killed=True`` — callers
    should keep their existing robust_relation / promising_candidate logic
    for anything that survived falsification untouched.

    Priority:
      1. Sample size gate, evaluated across EVERY falsification that reports
         an "n" — killed or not, and regardless of check name. Composite
         candidates are killed by ImprovementFalsifier/AblationFalsifier
         (names outside baseline/held_out/cross_seed), but those still score
         on the same tiny confirmation partition, so they are exactly as
         unreliable at small n. Any check run on a partition smaller than
         ``min_reliable_partition_n`` -> "inconclusive_candidate", before
         anything else is considered.
      2. If ``is_explicit_predictive_claim`` (the candidate's own statement
         asserts predictive performance above baseline, e.g. a composite
         "can be predicted by" claim) and it was killed by ANY predictive
         check (improvement, ablation, or a persistent negative cross_seed
         median) -> "refuted". Failing to predict better than baseline on a
         large-enough sample directly contradicts that literal claim.
      3. Elif ``direction_conflict`` (the candidate asserted a specific
         direction — e.g. "a stable positive association" — and the fitted
         relationship is in the opposite direction) and cross_seed shows a
         persistent negative median -> "refuted". A stable opposite-signed
         effect, or a failed preregistered directional prediction, is direct
         contradiction.
      4. Elif cross_seed shows a persistent negative median (aggregated
         across 9 reseeds/resamples, so it reflects a pattern rather than one
         unlucky split) on a generic association/relationship claim with no
         directional or predictive-performance assertion ->
         "functional_form_rejected_candidate". The tested functional form
         (e.g. raw linear) failed, but that does not by itself refute the
         underlying variable relationship — only that this particular model
         shape did not capture it.
      5. A negative score from baseline or held_out ALONE (each computed on
         a single fixed split) never triggers refutation or functional-form
         rejection on its own — one influential point landing in one
         particular split is noise. Falls through to "not_supported".
    """
    falsifications = finding.get("falsifications", []) or []
    killed = [a for a in falsifications if a.get("killed")]

    diagnostics: dict[str, Any] = {
        "killed_checks": [],
        "min_reliable_partition_n": min_reliable_partition_n,
        "hard_refutation_score_ceiling": hard_refutation_score_ceiling,
        "is_predictive_claim": is_explicit_predictive_claim,
    }
    if not killed:
        return "falsified_candidate", diagnostics  # nothing killed it; caller's problem

    # Sample size is a property of the test partition itself, not of which
    # specific falsifier happened to trip. Composite candidates are killed by
    # ImprovementFalsifier/AblationFalsifier — names outside baseline/held_out/
    # cross_seed — but those still score on the same tiny confirmation
    # partition, so they are just as unreliable at small n. Check every
    # falsification that reports an "n" (killed or not), not only the
    # baseline/held_out/cross_seed trio.
    smallest_n: int | None = None
    cross_seed_median: float | None = None
    cross_seed_killed = False
    for attack in falsifications:
        name = attack.get("name")
        detail = attack.get("detail", {}) or {}
        n = detail.get("n")
        if isinstance(n, int):
            smallest_n = n if smallest_n is None else min(smallest_n, n)
        if attack.get("killed"):
            score = detail.get("score") if name in ("baseline", "held_out") else detail.get("median")
            diagnostics["killed_checks"].append({"name": name, "score": score, "n": n})
            if name in RESAMPLE_CHECK_NAMES:
                cross_seed_killed = True
                if isinstance(score, (int, float)):
                    cross_seed_median = score

    diagnostics["smallest_test_partition_n"] = smallest_n
    diagnostics["cross_seed_median"] = cross_seed_median

    if smallest_n is not None and smallest_n < min_reliable_partition_n:
        diagnostics["reason"] = (
            f"Test partition had only {smallest_n} row(s), below the minimum of "
            f"{min_reliable_partition_n} needed for a reliable score."
        )
        return "inconclusive_candidate", diagnostics

    persistent_negative_cross_seed = (
        cross_seed_killed and cross_seed_median is not None and cross_seed_median < hard_refutation_score_ceiling
    )
    killed_by_predictive_check = any(a.get("name") not in _PAIRWISE_CHECK_NAMES for a in killed)

    if is_explicit_predictive_claim and (persistent_negative_cross_seed or killed_by_predictive_check):
        diagnostics["reason"] = (
            "This candidate's own statement asserts predictive performance above baseline; a predictive "
            "check (improvement/ablation/held-out/cross-seed) failed on a large-enough sample, directly "
            "contradicting that claim."
        )
        return "falsified_candidate", diagnostics

    if persistent_negative_cross_seed:
        if direction_conflict:
            diagnostics["reason"] = (
                "The fitted relationship is in the opposite direction from what this candidate asserts — "
                "a stable opposite-signed effect / failed directional prediction."
            )
            return "falsified_candidate", diagnostics
        diagnostics["reason"] = (
            "The tested model form performed persistently below baseline across resamples, but this does "
            "not by itself refute the underlying variable relationship — the functional form tried here "
            "(e.g. a raw linear fit) was likely the wrong shape, not evidence the variables are unrelated."
        )
        return "functional_form_rejected_candidate", diagnostics

    diagnostics["reason"] = "Score(s) below the support threshold, but not a persistent negative signal across resamples."
    return "not_supported_candidate", diagnostics


def _strip_transform_prefix(column: str) -> str:
    """Return the base column name with a leading log-transform prefix removed.

    Used to detect that ``body_mass`` and ``log_body_mass`` (or ``log10_``,
    case-insensitive) refer to the same underlying variable in a different
    functional form, so alternative-form survivors can be linked back to
    their rejected raw-scale counterparts.
    """
    lowered = column.lower()
    for prefix in ("log10_", "log1p_", "log_", "ln_"):
        if lowered.startswith(prefix):
            return column[len(prefix):]
    return column


def _candidate_family_key(payload: dict[str, Any]) -> tuple[str, ...] | None:
    """Group candidates that test the same underlying variables in different forms."""
    kind = payload.get("kind")
    outcome = payload.get("outcome")
    if not outcome:
        return None
    outcome_base = _strip_transform_prefix(str(outcome))
    # Linear and nonlinear (quadratic / log-x / log-y / log-log) forms of the
    # same predictor→outcome pair are members of ONE relationship family, so a
    # killed form can be linked to a surviving sibling form of the same pair.
    if kind in ("linear_association", "nonlinear_association"):
        predictor = payload.get("predictor")
        if not predictor:
            return None
        return ("assoc", _strip_transform_prefix(str(predictor)), outcome_base)
    if kind == "composite_linear":
        predictors = payload.get("predictors") or []
        bases = tuple(sorted(_strip_transform_prefix(str(p)) for p in predictors))
        return (kind, bases, outcome_base)
    return None


def apply_functional_form_overrides(
    findings: list[tuple[dict[str, Any], str]],
) -> dict[str, tuple[str, dict[str, Any]]]:
    """Reclassify killed numeric findings whose transformed sibling survived.

    ``findings`` is a list of ``(finding_dict, internal_finding_type)`` pairs
    already classified by the caller (survivors use robust_relation /
    promising_candidate; killed pairwise candidates use falsified_candidate /
    not_supported_candidate / inconclusive_candidate).

    Returns a dict of ``candidate_id -> (new_finding_type, extra_fields)``.
    Candidates classified ``falsified_candidate`` / ``not_supported_candidate``
    / ``inconclusive_candidate`` whose family sibling survived are upgraded to
    ``functional_form_rejected_candidate``. Candidates already classified
    ``functional_form_rejected_candidate`` by ``classify_pairwise_finding``
    itself (a persistent negative cross-seed on a generic, non-directional,
    non-predictive claim — see that function) keep their type but still get
    ``alternative_candidate_id`` attached here if a surviving sibling exists,
    so the diagnostic is as complete as possible either way. Candidates not
    present in the returned dict keep their original classification.
    """
    survivor_families: dict[tuple[str, ...], str] = {}
    for finding, ftype in findings:
        if ftype not in {"robust_relation", "promising_candidate"}:
            continue
        payload = (finding.get("candidate", {}) or {}).get("payload", {}) or {}
        key = _candidate_family_key(payload)
        if key is not None:
            survivor_families.setdefault(key, finding["candidate"]["id"])

    overrides: dict[str, tuple[str, dict[str, Any]]] = {}
    eligible_types = {
        "falsified_candidate", "not_supported_candidate",
        "inconclusive_candidate", "functional_form_rejected_candidate",
    }
    for finding, ftype in findings:
        if ftype not in eligible_types:
            continue
        payload = (finding.get("candidate", {}) or {}).get("payload", {}) or {}
        key = _candidate_family_key(payload)
        if key is None:
            continue
        alt_id = survivor_families.get(key)
        if not alt_id:
            continue
        cid = finding["candidate"]["id"]
        if alt_id == cid:
            continue
        overrides[cid] = (
            "functional_form_rejected_candidate",
            {"alternative_candidate_id": alt_id, "previous_finding_type": ftype},
        )
    return overrides


# ---------------------------------------------------------------------------
# Hypothesis / verdict separation
# ---------------------------------------------------------------------------

_VERDICT_REASON = {
    COMMITTED: "Survived baseline, held-out, and cross-seed falsification and met the commit threshold.",
    REJECTED: "At least one falsification check found evidence actively against the hypothesis "
               "(a score worse than a trivial baseline), not merely below the support threshold.",
    ARTIFACT: "Structural or transform relationship between columns; not an independent scientific finding.",
    PROVISIONAL: "Survived falsification but did not reach the commit threshold.",
    UNRESOLVED: "Could not be conclusively tested with the available data.",
    NOT_SUPPORTED: "The tested evidence did not clear the support threshold, but did not show evidence "
                   "actively contradicting the hypothesis either.",
    INCONCLUSIVE: "The held-out or cross-seed test partition was too small for its score to be a "
                  "reliable verdict; treat this result as untested rather than refuted.",
    FUNCTIONAL_FORM_REJECTED: "The tested functional form performed persistently below baseline, but this "
                              "does not by itself refute the underlying variable relationship. If a "
                              "transformed version of the same pair survived, it is named in "
                              "alternative_candidate_id.",
    SUPPORTED_ASSOCIATION: "A real, bootstrap-stable association with a meaningful effect size, which "
                           "nonetheless does not reach the standalone predictive-utility bar. The "
                           "association is supported; its usefulness as a lone predictor is limited.",
    REGIME_DEPENDENT: "The pooled relationship reverses or materially changes direction inside "
                      "identifiable subgroups, so no universal directional claim is committed. "
                      "Scoped per-subgroup claims are recorded instead.",
}


def _check_score(falsifications: list[dict[str, Any]], name: str) -> float | None:
    match = RESAMPLE_CHECK_NAMES if name in RESAMPLE_CHECK_NAMES else (name,)
    for attack in falsifications:
        if attack.get("name") in match:
            detail = attack.get("detail", {}) or {}
            # Each falsifier reports its primary metric under a stable key.
            if name == "baseline":
                return detail.get("score")
            if name == "held_out":
                return detail.get("score")
            if name in RESAMPLE_CHECK_NAMES:
                return detail.get("median")
            return attack.get("metric")
    return None


def _cross_seed_summary(falsifications: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Fixed-model validation-resample summary (matches ``validation_resample``
    or the ``cross_seed`` alias)."""
    for attack in falsifications:
        if attack.get("name") in RESAMPLE_CHECK_NAMES:
            detail = attack.get("detail", {}) or {}
            return {
                "check_kind": detail.get("check_kind", "fixed_model_validation_resample"),
                "median": detail.get("median"),
                "spread": detail.get("spread"),
                "seeds": detail.get("seeds"),
                "min_median": detail.get("min_median"),
                "max_spread": detail.get("max_spread"),
                "killed": bool(attack.get("killed")),
            }
    return None


def _repeated_refit_summary(falsifications: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Genuine repeated-independent-refit stability summary."""
    for attack in falsifications:
        if attack.get("name") == "repeated_refit":
            detail = attack.get("detail", {}) or {}
            if detail.get("skipped"):
                return None
            return {
                "check_kind": detail.get("check_kind", "repeated_independent_refit"),
                "median": detail.get("median"),
                "lower_quantile": detail.get("lower_quantile"),
                "score_variance": detail.get("score_variance"),
                "valid_fits": detail.get("valid_fits"),
                "valid_fit_fraction": detail.get("valid_fit_fraction"),
                "fit_failures": detail.get("fit_failures"),
                "direction_stability": detail.get("direction_stability"),
                "coefficient_stability": detail.get("coefficient_stability"),
                "train_n_median": detail.get("train_n_median"),
                "val_n_median": detail.get("val_n_median"),
            }
    return None


def derive_finding_record(
    finding: dict[str, Any],
    finding_type: str,
    *,
    influence_warning: dict[str, Any] | None = None,
    classification_diagnostics: dict[str, Any] | None = None,
    functional_form_override: dict[str, Any] | None = None,
    full_data_score: float | None = None,
    is_predictive_claim: bool = False,
    association_evidence: dict[str, Any] | None = None,
    missingness: dict[str, Any] | None = None,
    model_family: dict[str, Any] | None = None,
    subgroup_warning: dict[str, Any] | None = None,
    artifact_warning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Split an affirmative candidate into hypothesis + verdict + check scores.

    The returned record is what the storage layer persists alongside the claim
    link and what the API returns. ``hypothesis_text`` is the affirmative
    candidate statement; ``verdict`` is the public state; for rejected findings
    the affirmative text must be presented as a *candidate hypothesis*, never as
    an established conclusion.

    ``classification_diagnostics`` (from ``classify_pairwise_finding``) and
    ``functional_form_override`` (from ``apply_functional_form_overrides``)
    are surfaced as ``rejection_reason``/``sample_sizes`` and
    ``alternative_candidate_id`` respectively, so a not_supported/inconclusive/
    functional_form_rejected verdict is always self-explanatory without
    cross-referencing the raw ledger. ``full_data_score`` is a report-only fit
    on the entire dataset (never used for any decision) shown alongside the
    held-out score so users can see both "how well does this fit overall" and
    "did it generalize." ``is_predictive_claim`` is always recorded — even for
    committed/provisional findings — so it is visible on screen whenever the
    candidate's own statement asserts predictive performance (e.g. a composite
    "can be predicted by" claim), not only when that claim gets refuted.
    """
    candidate = finding.get("candidate", {}) or {}
    verdict = finding.get("verdict", {}) or {}
    falsifications = finding.get("falsifications", []) or []
    state = public_state(finding_type)

    passed = [a.get("name") for a in falsifications if not a.get("killed")]
    failed = [a.get("name") for a in falsifications if a.get("killed")]

    record: dict[str, Any] = {
        "hypothesis_text": candidate.get("statement", ""),
        "finding_type": finding_type,
        "verdict": state,
        "verdict_reason": _VERDICT_REASON.get(state, ""),
        "is_candidate_hypothesis": state in {
            REJECTED, ARTIFACT, PROVISIONAL, UNRESOLVED,
            NOT_SUPPORTED, INCONCLUSIVE, FUNCTIONAL_FORM_REJECTED,
            REGIME_DEPENDENT,
        },
        "is_predictive_claim": is_predictive_claim,
        "passed_checks": passed,
        "failed_checks": failed,
        "candidate_score": verdict.get("score"),
        "metric_name": finding.get("selection_metric"),
        "baseline_score": _check_score(falsifications, "baseline"),
        "held_out_score": _check_score(falsifications, "held_out"),
        "held_out_n": _check_detail(falsifications, "held_out").get("n"),
        "baseline_n": _check_detail(falsifications, "baseline").get("n"),
        "full_data_score_diagnostic": full_data_score,
        # Fixed-model validation-resample summary. ``cross_seed_summary`` is kept
        # as a backward-compatible alias; ``validation_resample_summary`` is the
        # honest name. Both point at the same data.
        "cross_seed_summary": _cross_seed_summary(falsifications),
        "validation_resample_summary": _cross_seed_summary(falsifications),
        # Genuine repeated-independent-refit stability (model reproducibility).
        "repeated_refit_summary": _repeated_refit_summary(falsifications),
        "final_status": finding.get("final_status"),
    }

    # ------------------------------------------------------------------
    # Separated evidence axes (association / predictive / functional-form).
    # Each axis is persisted independently so a real association is never
    # collapsed into a single predictive score. Flat legacy fields above are
    # retained for backward compatibility with existing surfaces and tests.
    # ------------------------------------------------------------------
    resample = record["validation_resample_summary"]
    repeated = record["repeated_refit_summary"]
    record["predictive_utility"] = {
        "metric_name": record["metric_name"],
        "held_out_score": record["held_out_score"],
        "held_out_n": record["held_out_n"],
        "baseline_score": record["baseline_score"],
        "beats_baseline": (
            record["held_out_score"] is not None
            and record["baseline_score"] is not None
            and record["held_out_score"] > record["baseline_score"]
        ),
        "full_data_score_diagnostic": full_data_score,
    }
    record["functional_form_stability"] = {
        "form": (model_family or {}).get("form") if model_family else None,
        "preferred_form": (model_family or {}).get("preferred_form") if model_family else None,
        "is_preferred_form": (model_family or {}).get("is_preferred") if model_family else None,
        "validation_resample": resample,
        "repeated_refit": repeated,
        "direction_stability": (repeated or {}).get("direction_stability") if repeated else None,
    }
    if association_evidence:
        record["association_evidence"] = association_evidence
    if model_family:
        record["model_family"] = model_family
    if missingness:
        record["missingness"] = missingness
    if subgroup_warning:
        record["subgroup_warning"] = subgroup_warning
    if artifact_warning:
        record["artifact_warning"] = artifact_warning

    if influence_warning:
        record["influence_warning"] = influence_warning
    if classification_diagnostics:
        record["rejection_reason"] = classification_diagnostics.get("reason")
        record["rejection_diagnostics"] = classification_diagnostics
    if functional_form_override:
        record["alternative_candidate_id"] = functional_form_override.get("alternative_candidate_id")
    return record
