from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _candidate_columns(payload: dict[str, Any]) -> list[str]:
    """Ordered, de-duplicated list of dataset columns a candidate depends on."""
    cols: list[str] = []
    for key in ("predictor", "outcome", "group"):
        v = payload.get(key)
        if v:
            cols.append(str(v))
    for p in payload.get("predictors", []) or []:
        cols.append(str(p))
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _missingness_receipt(df: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Per-candidate complete-case missingness receipt (never imputes)."""
    cols = [c for c in _candidate_columns(payload) if c in df.columns]
    if not cols:
        return None
    total = int(len(df))
    complete = pd.Series(True, index=df.index)
    per_var: dict[str, float] = {}
    group_col = payload.get("group")
    for c in cols:
        col = df[c]
        coerced = pd.to_numeric(col, errors="coerce")
        # Numeric candidate columns: values that cannot be coerced are effectively
        # missing. The categorical grouping column uses raw null semantics.
        if c != group_col and float(coerced.notna().mean()) >= 0.5:
            miss = coerced.isna()
        else:
            miss = col.isna()
        per_var[c] = round(float(miss.mean()), 4)
        complete &= ~miss
    effective_n = int(complete.sum())
    excluded = total - effective_n
    return {
        "total_rows": total,
        "effective_n": effective_n,
        "rows_excluded_missing": excluded,
        "excluded_fraction": round(excluded / total, 4) if total else 0.0,
        "missing_fraction_by_variable": per_var,
        "substantial_missingness": bool(total and excluded / total >= 0.2),
    }


def _association_evidence(df: pd.DataFrame, payload: dict[str, Any], *, seed: int = 92617) -> dict[str, Any] | None:
    """Effect-size + bootstrap-stability evidence, independent of predictive utility."""
    kind = payload.get("kind")
    rng = np.random.default_rng(seed)
    if kind == "linear_association":
        pred, out = payload.get("predictor"), payload.get("outcome")
        if pred not in df.columns or out not in df.columns:
            return None
        x = pd.to_numeric(df[pred], errors="coerce")
        y = pd.to_numeric(df[out], errors="coerce")
        m = x.notna() & y.notna()
        xa, ya = x[m].to_numpy(float), y[m].to_numpy(float)
        if len(xa) < 5:
            return None
        r = float(np.corrcoef(xa, ya)[0, 1])
        boots = []
        for _ in range(300):
            idx = rng.integers(0, len(xa), len(xa))
            rr = np.corrcoef(xa[idx], ya[idx])[0, 1]
            if np.isfinite(rr):
                boots.append(float(rr))
        ci = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)] if boots else None
        sign_stability = round(float(np.mean([(b > 0) == (r > 0) for b in boots])), 4) if boots else None
        return {
            "effect_size_metric": "pearson_r",
            "effect_size": round(r, 4),
            "ci95": ci,
            "bootstrap_sign_stability": sign_stability,
            "n": int(len(xa)),
        }
    if kind == "nonlinear_association":
        from .table_domain import _fit_form
        pred, out = payload.get("predictor"), payload.get("outcome")
        form = payload.get("form")
        if pred not in df.columns or out not in df.columns:
            return None
        x = pd.to_numeric(df[pred], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[out], errors="coerce").to_numpy(float)
        fit = _fit_form(x, y, form)
        if not fit:
            return None
        evidence = {
            "effect_size_metric": f"r2_{form}",
            "effect_size": round(float(fit["r2"]), 4),
            "form": form,
            "n": int(np.isfinite(x).sum()),
        }
        if "exponent" in fit:
            evidence["power_law_exponent"] = round(float(fit["exponent"]), 4)
        return evidence
    if kind == "group_difference":
        group, out = payload.get("group"), payload.get("outcome")
        if group not in df.columns or out not in df.columns:
            return None
        temp = pd.DataFrame({"g": df[group].astype(str), "y": pd.to_numeric(df[out], errors="coerce")}).dropna()
        if len(temp) < 6 or temp["g"].nunique() < 2:
            return None
        overall = float(temp["y"].mean())
        total_ss = float(((temp["y"] - overall) ** 2).sum())
        if total_ss <= 1e-12:
            return None
        grp = list(temp.groupby("g"))
        k, N = len(grp), len(temp)
        between = float(sum(len(s) * (float(s["y"].mean()) - overall) ** 2 for _, s in grp))
        eta2 = between / total_ss
        ms_within = (total_ss - between) / (N - k) if N > k else 0.0
        omega2 = (between - (k - 1) * ms_within) / (total_ss + ms_within) if (total_ss + ms_within) > 0 else None
        means = {str(lab): round(float(s["y"].mean()), 4) for lab, s in grp}
        counts = {str(lab): int(len(s)) for lab, s in grp}
        # Largest-group reference contrast.
        ref = max(counts, key=counts.get)
        contrasts = {lab: round(means[lab] - means[ref], 4) for lab in means if lab != ref}
        return {
            "effect_size_metric": "eta_squared",
            "effect_size": round(float(eta2), 4),
            "omega_squared": round(float(omega2), 4) if omega2 is not None else None,
            "group_means": means,
            "group_counts": counts,
            "reference_group": ref,
            "group_contrasts_vs_reference": contrasts,
            "n": int(N),
            "n_groups": int(k),
        }
    return None


def _qualifies_as_supported_association(assoc: dict[str, Any] | None) -> bool:
    """A generated group effect is a *supported association* when it explains
    variance beyond chance (omega² > 0) with a non-trivial effect size (η² ≥ 0.02,
    the conventional small-effect boundary). Thresholds are effect-size
    conventions, not tuned to any dataset's expected answer."""
    if not assoc or assoc.get("effect_size_metric") != "eta_squared":
        return False
    eta2 = assoc.get("effect_size")
    omega2 = assoc.get("omega_squared")
    return (
        isinstance(eta2, (int, float)) and eta2 >= 0.02
        and isinstance(omega2, (int, float)) and omega2 > 0.005
    )

from orbita import ActorRole, EpistemicLedger, EvidenceKind, Stance
from orbita_discovery.core import Candidate, Engine, Ledger, finding_to_dict, survivors
from orbita_discovery.falsifiers import (
    BaselineFalsifier,
    CrossSeedFalsifier,
    HeldOutFalsifier,
    RepeatedRefitValidator,
)
from orbita_discovery.judges import GatedJudge

from .compiler import ResearchCompiler, compute_plan_hash, verify_plan_schema_executable
from .composition import build_backward_eliminated_composites, build_composite_candidates
from .falsifiers import AblationFalsifier, ImprovementFalsifier
from .model_artifact import (
    model_from_artifact,
    save_model_artifact,
    serialize_deployment_artifact,
    serialize_selection_artifact,
)
from .ingestion import ArtifactIngestor
from .memory import BeliefMemory
from .metrics import higher_is_better, select_best_finding, validate_metric
from .reporting import ReportCompiler
from .storage import CaseStore
from .table_domain import UploadedTableDomain


def _freeze_selected_models(
    findings: list[dict[str, Any]],
    selection_scores: dict[str, float],
    evaluation_metric: str,
    hib: bool,
) -> dict[str, dict[str, Any]]:
    """Deterministically select the winning committed model per outcome column.

    Uses ONLY selection-partition scores.  Must be called BEFORE
    final_validation scores are computed so the holdout partition cannot
    influence model selection.

    Returns a mapping of ``outcome → selection record`` containing:
    ``selected_model_id``, ``selection_metric``, ``selection_metric_score``,
    ``selection_higher_is_better``.
    """
    from .metrics import NULL_SCORE

    survivors_by_outcome: dict[str, list[dict[str, Any]]] = {}
    for f in findings:
        if f["final_status"] == "refuted" or any(a["killed"] for a in f["falsifications"]):
            continue
        outcome = f["candidate"]["payload"].get("outcome")
        if outcome:
            survivors_by_outcome.setdefault(outcome, []).append(f)

    null = NULL_SCORE.get(evaluation_metric, 0.0)
    selected: dict[str, dict[str, Any]] = {}
    for outcome, group in survivors_by_outcome.items():
        def _key(f: dict[str, Any], _null: float = null, _hib: bool = hib) -> tuple:
            sc = selection_scores.get(f["candidate"]["id"])
            if sc is None or not math.isfinite(sc):
                sc = _null
            return (sc if _hib else -sc, f["candidate"]["id"])

        winner = max(group, key=_key)
        cid = winner["candidate"]["id"]
        selected[outcome] = {
            "selected_model_id": cid,
            "selection_metric": evaluation_metric,
            "selection_metric_score": selection_scores.get(cid),
            "selection_higher_is_better": hib,
        }
    return selected


class ResearchMVP:
    """End-to-end local service for intake → plan → discovery → belief graph → report."""

    def __init__(self, db_path: str | Path = "orbita_mvp.db", workspace: str | Path = "orbita_workspace"):
        self.db_path = Path(db_path)
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.ledger = EpistemicLedger(self.db_path)
        self.store = CaseStore(self.ledger, self.workspace)
        self.memory = BeliefMemory(self.ledger)
        self.ingestor = ArtifactIngestor()
        self.compiler = ResearchCompiler()
        self.reporter = ReportCompiler()

    def close(self) -> None:
        self.ledger.close()

    def __enter__(self) -> "ResearchMVP":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Intake and plans
    # ------------------------------------------------------------------
    def create_case(self, *, name: str, goal: str = "", domain_hint: str | None = None) -> dict[str, Any]:
        return self.store.create_case(name=name, goal=goal, domain_hint=domain_hint)

    def add_file(self, case_id: str, file_path: str | Path) -> dict[str, Any]:
        case_dir = self.store.case_dir(case_id) / "uploads"
        record = self.ingestor.ingest(file_path, case_dir)
        return self.store.add_file_record(case_id, record)

    def delete_case(self, case_id: str) -> dict[str, Any]:
        return self.store.delete_case(case_id)

    def compile_case(
        self,
        case_id: str,
        *,
        max_candidates: int = 60,
        target_transform: str | None = None,
        outcome_domain: str | None = None,
        evaluation_metric: str = "r2",
        confirmation_fraction: float = 0.25,
        final_validation_fraction: float = 0.15,
        target_column: str | None = None,
    ) -> dict[str, Any]:
        case = self.store.get_case(case_id)
        plan = self.compiler.compile(
            case,
            max_candidates=max_candidates,
            target_transform=target_transform,
            outcome_domain=outcome_domain,
            evaluation_metric=evaluation_metric,
            confirmation_fraction=confirmation_fraction,
            final_validation_fraction=final_validation_fraction,
            target_column=target_column,
        )
        return self.store.save_plan(case_id, plan, compiler="orbita-heuristic-compiler/0.1")

    def submit_external_plan(self, case_id: str, plan: dict[str, Any], *, compiler: str = "external-ai") -> dict[str, Any]:
        case = self.store.get_case(case_id)
        validated = self.compiler.validate_external_plan(case, plan)
        return self.store.save_plan(case_id, validated, compiler=compiler)

    def revise_plan(self, plan_id: str, plan: dict[str, Any], *, compiler: str = "human-review") -> dict[str, Any]:
        current = self.store.get_plan(plan_id)
        case = self.store.get_case(current["case_id"])
        validated = self.compiler.validate_external_plan(case, plan)
        return self.store.revise_plan(plan_id, validated, compiler=compiler)

    def approve_plan(self, plan_id: str, *, reviewer: str = "local-user") -> dict[str, Any]:
        return self.store.approve_plan(plan_id, reviewer=reviewer)

    # ------------------------------------------------------------------
    # Discovery and import into persistent memory
    # ------------------------------------------------------------------
    def run_case(self, case_id: str, *, plan_id: str | None = None, auto_approve: bool = False, graph_id: str | None = None) -> dict[str, Any]:
        case = self.store.get_case(case_id)
        if plan_id is None:
            if not case["plans"]:
                plan_record = self.compile_case(case_id)
            else:
                plan_record = case["plans"][0]
        else:
            plan_record = self.store.get_plan(plan_id)
        if plan_record["status"] != "approved":
            if not auto_approve:
                raise ValueError("The analysis plan must be approved before execution")
            plan_record = self.store.approve_plan(plan_record["id"], reviewer="auto-approval-requested")
        plan = plan_record["plan"]
        if plan.get("status") in {"needs_data", "no_candidates"}:
            raise ValueError("The plan is not executable: " + "; ".join(plan.get("blocking_questions", [])))

        # Verify plan integrity before execution
        stored_hash = plan.get("plan_hash")
        if stored_hash:
            current_hash = compute_plan_hash(plan)
            if current_hash != stored_hash:
                raise ValueError(
                    f"Plan integrity check failed: stored hash {stored_hash[:12]}… "
                    f"does not match current hash {current_hash[:12]}…. "
                    "The plan may have been modified after compilation."
                )

        # Guard: only v0.3 plans are executable under this engine version.
        # Historical v0.2 plans remain auditable; their hashes still verify.
        verify_plan_schema_executable(plan)

        selected_file = self.store.get_file(plan["selected_dataset"]["file_id"])
        df = pd.read_csv(selected_file["extracted_path"])
        run_record = self.store.create_run(case_id, plan_record["id"])
        run_dir = self.store.case_dir(case_id) / "runs" / run_record["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "discovery_ledger.jsonl"

        try:
            thresholds = plan.get("thresholds", {})
            target_transform = plan.get("target_transform") or None
            evaluation_metric = plan.get("evaluation_metric") or "r2"
            validate_metric(evaluation_metric)

            gen = plan.get("candidate_generation", {})
            scout_fraction = float(gen.get("scout_fraction", 0.6))
            confirmation_fraction = float(gen.get("confirmation_fraction", 0.25))
            final_validation_fraction = float(gen.get("final_validation_fraction", 0.15))
            seed = int(gen.get("seed", 20260623))
            # Chronological validation axis (a clear ordered calendar/sequence
            # column), when the dataset has one — else random validation.
            time_column = plan.get("chronological_axis") or None

            domain = UploadedTableDomain(
                df,
                plan["candidates"],
                scout_fraction=scout_fraction,
                confirmation_fraction=confirmation_fraction,
                final_validation_fraction=final_validation_fraction,
                seed=seed,
                target_transform=target_transform,
                evaluation_metric=evaluation_metric,
                time_column=time_column,
            )
            judge = GatedJudge(
                commit_at=float(thresholds.get("commit_at", 0.25)),
                baseline_margin=float(thresholds.get("baseline_margin", 0.05)),
            )
            pairwise_falsifiers = [
                BaselineFalsifier(margin=float(thresholds.get("baseline_margin", 0.05))),
                HeldOutFalsifier(min_score=float(thresholds.get("held_out_min", 0.15))),
                CrossSeedFalsifier(
                    seeds=int(thresholds.get("cross_seed_count", 9)),
                    min_median=float(thresholds.get("cross_seed_min", 0.15)),
                    max_spread=thresholds.get("cross_seed_max_spread", 0.65),
                ),
                RepeatedRefitValidator(
                    seeds=int(thresholds.get("repeated_refit_count", 12)),
                ),
            ]
            # Phase 1: pairwise candidate falsification
            phase1_ledger = Ledger(ledger_path)
            engine = Engine(judge, pairwise_falsifiers, phase1_ledger)
            engine.run(domain)
            phase1_findings = [finding_to_dict(item) for item in phase1_ledger.entries]
            phase1_survivors = survivors(phase1_ledger)

            # Compute configured-metric scores for phase-1 survivors so that
            # ImprovementFalsifier can compare in the correct metric units.
            survivor_metric_scores: dict[str, float] = {}
            hib = higher_is_better(evaluation_metric)
            for entry in phase1_survivors:
                fd = finding_to_dict(entry)
                cid = fd["candidate"]["id"]
                c_obj = Candidate(
                    id=cid,
                    statement=fd["candidate"]["statement"],
                    payload=fd["candidate"]["payload"],
                )
                ev = domain.evidence_for(c_obj)
                train, test = domain.splits(ev, seed=1)
                model = domain.refit(c_obj, train)
                if model.get("valid"):
                    ms = domain.score_metric(c_obj, model, test)
                    survivor_metric_scores[cid] = ms

            # Phase 2: composite candidate generation and falsification
            composite_min_predictors = int(thresholds.get("composite_min_predictors", 2))
            composite_max_predictors = int(thresholds.get("composite_max_predictors", 10))
            ablation_min_contribution = float(thresholds.get("ablation_min_contribution", 0.01))

            composite_specs = build_composite_candidates(
                [finding_to_dict(s) for s in phase1_survivors],
                min_predictors=composite_min_predictors,
                max_predictors=composite_max_predictors,
                metric_scores=survivor_metric_scores,
            )
            # Fix best_individual_metric_score direction: build_composite_candidates
            # stored min() by default; correct to max/min per actual metric direction.
            for spec in composite_specs:
                sm = spec.get("scout_metric", {})
                pms = sm.get("parent_metric_scores", {})
                if pms:
                    vals = [v for v in pms.values() if v is not None]
                    if vals:
                        sm["best_individual_metric_score"] = (
                            round(max(vals), 6) if hib else round(min(vals), 6)
                        )

            # Add backward-eliminated composites when strategy requests it
            composition_strategy = plan.get("composition_strategy", "composition_v1")
            if composite_specs and "backward_elimination" in composition_strategy:
                # Build a temporary domain for elimination (same partitions as composite_domain)
                # We need the domain before composite_domain is built, so create it now.
                _be_domain = UploadedTableDomain(
                    df,
                    composite_specs,
                    scout_fraction=scout_fraction,
                    confirmation_fraction=confirmation_fraction,
                    final_validation_fraction=final_validation_fraction,
                    seed=seed,
                    target_transform=target_transform,
                    evaluation_metric=evaluation_metric,
                    time_column=time_column,
                )
                reduced_specs = build_backward_eliminated_composites(
                    composite_specs,
                    _be_domain,
                    min_contribution=ablation_min_contribution,
                    min_predictors=composite_min_predictors,
                )
                # Add reduced specs after full specs so full composites are tested first
                composite_specs = composite_specs + reduced_specs

            phase2_findings: list[dict[str, Any]] = []
            if composite_specs:
                composite_domain = UploadedTableDomain(
                    df,
                    composite_specs,
                    scout_fraction=scout_fraction,
                    confirmation_fraction=confirmation_fraction,
                    final_validation_fraction=final_validation_fraction,
                    seed=seed,
                    target_transform=target_transform,
                    evaluation_metric=evaluation_metric,
                    time_column=time_column,
                )
                composite_falsifiers = [
                    ImprovementFalsifier(
                        min_improvement=float(thresholds.get("composite_min_improvement", 0.01))
                    ),
                    HeldOutFalsifier(min_score=float(thresholds.get("held_out_min", 0.15))),
                    CrossSeedFalsifier(
                        seeds=int(thresholds.get("cross_seed_count", 9)),
                        min_median=float(thresholds.get("cross_seed_min", 0.15)),
                        max_spread=thresholds.get("cross_seed_max_spread", 0.65),
                    ),
                    AblationFalsifier(
                        min_contribution=float(thresholds.get("ablation_min_contribution", 0.01))
                    ),
                    RepeatedRefitValidator(
                        seeds=int(thresholds.get("repeated_refit_count", 12)),
                    ),
                ]
                phase2_ledger = Ledger(ledger_path, truncate=False)
                engine2 = Engine(judge, composite_falsifiers, phase2_ledger)
                engine2.run(composite_domain)
                phase2_findings = [finding_to_dict(item) for item in phase2_ledger.entries]

            all_findings = phase1_findings + phase2_findings

            # Runtime leakage guard: if a target_column was frozen in the plan,
            # reject any candidate whose predictor list includes it.
            # This runs on ALL findings (including refuted) to catch any engine bug.
            _target_col = plan.get("target_column") or plan.get("candidate_generation", {}).get("target_column")
            if _target_col:
                for _f in all_findings:
                    _pay = _f["candidate"]["payload"]
                    _pred = _pay.get("predictor")
                    _preds = _pay.get("predictors", [])
                    if _pred == _target_col or _target_col in _preds:
                        raise ValueError(
                            f"Target leakage detected at runtime: target column "
                            f"{_target_col!r} appears as a predictor in candidate "
                            f"{_f['candidate']['id']!r}. Aborting run to prevent "
                            f"invalid results."
                        )

            # Collect selection-partition metric scores for all survivors.
            # Phase-1 scores are in survivor_metric_scores (computed on the
            # selection partition via score_metric with seed=1).
            # Phase-2 composite scores were already computed by
            # ImprovementFalsifier on the same partition; extract from the
            # falsification detail to avoid a redundant refit.
            all_selection_scores: dict[str, float] = dict(survivor_metric_scores)
            for f in phase2_findings:
                cid = f["candidate"]["id"]
                if cid not in all_selection_scores:
                    imp = next(
                        (a for a in f["falsifications"] if a["name"] == "improvement"),
                        None,
                    )
                    if imp and imp.get("detail", {}).get("composite_score") is not None:
                        all_selection_scores[cid] = float(imp["detail"]["composite_score"])

            # Stamp selection_metric_score onto every finding so it is
            # visible in the ledger and available as a legacy fallback.
            for f in all_findings:
                f["selection_metric_score"] = all_selection_scores.get(f["candidate"]["id"])
                f["selection_metric"] = evaluation_metric

            # FREEZE selected model per outcome using ONLY selection-phase
            # evidence — BEFORE final_validation scores are computed so the
            # holdout partition cannot influence model selection.
            selected_models = _freeze_selected_models(
                all_findings, all_selection_scores, evaluation_metric, hib
            )

            # Step A: Selection artifacts — scout-fitted, created BEFORE final-validation
            # exposure.  Created for all survivors so FV scoring never needs to refit.
            # model_artifacts (keyed by outcome_col) is populated here with selection
            # metadata and updated after FV with deployment artifact paths.
            selection_artifacts_by_cid: dict[str, dict[str, Any]] = {}
            model_artifacts: dict[str, dict[str, Any]] = {}
            selected_file_path = selected_file["extracted_path"]
            for finding in all_findings:
                is_surv = (
                    finding["final_status"] != "refuted"
                    and not any(a["killed"] for a in finding["falsifications"])
                )
                if not is_surv:
                    continue
                cid = finding["candidate"]["id"]
                kind_str = finding["candidate"]["payload"].get("kind", "")
                dom_for_sel = (
                    composite_domain if kind_str == "composite_linear" and composite_specs
                    else domain
                )
                try:
                    sel_art = serialize_selection_artifact(
                        run_id=run_record["id"],
                        plan=plan,
                        finding=finding,
                        domain=dom_for_sel,
                    )
                    sel_art_path = save_model_artifact(sel_art, run_dir, kind="selection")
                    selection_artifacts_by_cid[cid] = {
                        "selection_artifact_id": sel_art["selection_artifact_id"],
                        "selection_artifact_path": str(sel_art_path),
                        "selection_artifact_sha256": sel_art["artifact_sha256"],
                        "artifact": sel_art,
                    }
                except Exception as _sel_err:
                    selection_artifacts_by_cid[cid] = {"error": str(_sel_err)}

            for outcome_col, sel_info in selected_models.items():
                sel_id = sel_info["selected_model_id"]
                art_entry = selection_artifacts_by_cid.get(sel_id, {})
                if "error" not in art_entry and art_entry:
                    model_artifacts[outcome_col] = {
                        "selection_artifact_id": art_entry["selection_artifact_id"],
                        "selection_artifact_path": art_entry["selection_artifact_path"],
                        "selection_artifact_sha256": art_entry["selection_artifact_sha256"],
                        "selected_model_id": sel_id,
                    }
                else:
                    model_artifacts[outcome_col] = {
                        "error": art_entry.get("error", "no selection artifact created"),
                        "selected_model_id": sel_id,
                    }

            # ----------------------------------------------------------
            # Step B: Final validation — apply stored selection-artifact
            # coefficients to the held-out final_validation partition.
            # REPORT-ONLY — these scores do NOT alter selected_model_id,
            # model precedence, feature sets, or coefficients.
            # No fitting occurs here; model_from_artifact reconstructs the
            # model dict from stored intercept and coefficients only.
            # ----------------------------------------------------------
            def _score_from_artifact(
                finding: dict[str, Any],
                artifact: dict[str, Any],
                dom: UploadedTableDomain,
            ) -> float | None:
                """Apply stored selection-artifact coefficients to the FV partition."""
                if len(dom.final_validation) < 3:
                    return None
                payload = finding["candidate"]["payload"]
                model = model_from_artifact(artifact, payload)
                if not model.get("valid"):
                    return None
                c_obj = Candidate(
                    id=finding["candidate"]["id"],
                    statement=finding["candidate"]["statement"],
                    payload=payload,
                )
                return dom.score_metric(c_obj, model, dom.final_validation)

            for finding in all_findings:
                is_survivor = (
                    finding["final_status"] != "refuted"
                    and not any(a["killed"] for a in finding["falsifications"])
                )
                if is_survivor:
                    cid = finding["candidate"]["id"]
                    kind_str = finding["candidate"]["payload"].get("kind", "")
                    dom = (
                        composite_domain if kind_str == "composite_linear" and composite_specs
                        else domain
                    )
                    art_entry = selection_artifacts_by_cid.get(cid, {})
                    sel_art = art_entry.get("artifact") if "error" not in art_entry else None
                    try:
                        fvs = _score_from_artifact(finding, sel_art, dom) if sel_art else None
                    except Exception:
                        fvs = None
                    finding["final_validation_metric_score"] = fvs
                    finding["final_validation_metric"] = evaluation_metric
                    finding["final_validation_report_only"] = True
                    finding["evaluation_metric"] = evaluation_metric
                else:
                    finding["final_validation_metric_score"] = None
                    finding["final_validation_metric"] = evaluation_metric
                    finding["final_validation_report_only"] = True
                    finding["evaluation_metric"] = evaluation_metric

            # Step C: Deployment artifacts — refit on full CSV, created AFTER FV
            # scoring has been recorded.  /predict loads deployment artifacts.
            # They are distinct from selection artifacts and reference them by ID.
            for outcome_col, sel_info in selected_models.items():
                sel_id = sel_info["selected_model_id"]
                sel_finding = next(
                    (f for f in all_findings if f["candidate"]["id"] == sel_id), None
                )
                if sel_finding is None:
                    continue
                art_entry = selection_artifacts_by_cid.get(sel_id, {})
                sel_artifact_id = art_entry.get("selection_artifact_id", "")
                fv_score = sel_finding.get("final_validation_metric_score")
                try:
                    dep_art = serialize_deployment_artifact(
                        run_id=run_record["id"],
                        plan=plan,
                        finding=sel_finding,
                        normalized_path=selected_file_path,
                        selection_artifact_id=sel_artifact_id,
                        final_validation_score=fv_score,
                    )
                    dep_art_path = save_model_artifact(dep_art, run_dir, kind="deployment")
                    model_artifacts.setdefault(outcome_col, {}).update({
                        "model_artifact_id": dep_art["model_artifact_id"],
                        "model_artifact_path": str(dep_art_path),
                        "model_artifact_sha256": dep_art["artifact_sha256"],
                    })
                except Exception as _dep_err:
                    model_artifacts.setdefault(outcome_col, {})["deployment_error"] = str(_dep_err)

            all_survivor_ids = [
                f["candidate"]["id"] for f in all_findings
                if f["final_status"] != "refuted"
                and not any(a["killed"] for a in f["falsifications"])
            ]
            engine_result = {
                "run_id": run_record["id"],
                "engine": "orbita-discovery-kit/0.2-compatible",
                "domain": "uploaded_table",
                "evaluation_metric": evaluation_metric,
                "higher_is_better": hib,
                "ledger_path": str(ledger_path.resolve()),
                "candidate_count": len(all_findings),
                "survivor_count": len(all_survivor_ids),
                "survivor_ids": all_survivor_ids,
                "findings": all_findings,
                "composite_candidates_proposed": len(composite_specs),
                "selected_models": selected_models,
                "model_artifacts": model_artifacts,
            }

            import_summary = self._import_result(
                case_id=case_id,
                case_run_id=run_record["id"],
                dataset_file=selected_file,
                plan=plan,
                result=engine_result,
                dataframe=df,
                domain=domain,
                composite_domain=composite_domain if composite_specs else None,
            )

            # Phase 2A: provenance stamp on this run's claims. Reserved origin
            # keys are always present; operators stays [] until the operator
            # registry becomes executable (Phase 2D).
            self.store.stamp_run_claims(
                case_id=case_id,
                run_id=run_record["id"],
                graph_id=graph_id,
                origin={
                    "dataset_ids": [selected_file["id"]],
                    "engine_version": os.getenv("GIT_COMMIT_SHA", os.getenv("RAILWAY_GIT_COMMIT_SHA", "unknown")),
                    "plan_hash": plan.get("plan_hash"),
                    "operators": [],
                },
            )

            # Attach artifact provenance as evidence nodes in the belief graph.
            # Must happen BEFORE capture_graph so these nodes land in the snapshot.
            candidate_to_claim = import_summary.get("candidate_to_claim", {})
            for _oc, _si in selected_models.items():
                _sel_cid = _si["selected_model_id"]
                _claim_id = candidate_to_claim.get(_sel_cid)
                if not _claim_id:
                    continue
                _yart = model_artifacts.get(_oc, {})
                _sel_art_id = _yart.get("selection_artifact_id", "")
                _sel_art_sha = _yart.get("selection_artifact_sha256", "")
                _sel_art_path = _yart.get("selection_artifact_path", "")
                _dep_art_id = _yart.get("model_artifact_id", "")
                _dep_art_sha = _yart.get("model_artifact_sha256", "")
                _dep_art_path = _yart.get("model_artifact_path", "")
                _sel_finding = next((f for f in all_findings if f["candidate"]["id"] == _sel_cid), None)
                _fv = _sel_finding.get("final_validation_metric_score") if _sel_finding else None

                for _evi_uri, _evi_excerpt, _evi_key, _evi_content in [
                    # Selection artifact
                    (
                        f"file://{_sel_art_path}",
                        f"selection-artifact:{_sel_art_id}|sha256:{_sel_art_sha[:16]}",
                        f"sel-artifact:{_sel_art_id}",
                        json.dumps({"artifact_kind": "selection_artifact",
                                    "selection_artifact_id": _sel_art_id,
                                    "artifact_sha256": _sel_art_sha,
                                    "training_partition": "scout",
                                    "run_id": run_record["id"]}, sort_keys=True),
                    ),
                    # Final-validation evidence
                    (
                        f"file://{_sel_art_path}",
                        f"final-validation-evidence|{evaluation_metric}={_fv}|sel_artifact:{_sel_art_id}|sha256:{_sel_art_sha[:16]}",
                        f"fv-evidence:{run_record['id']}:{_sel_cid}",
                        json.dumps({"artifact_kind": "final_validation_evidence",
                                    "fv_score": _fv,
                                    "metric": evaluation_metric,
                                    "higher_is_better": hib,
                                    "report_only": True,
                                    "selection_artifact_id": _sel_art_id,
                                    "selection_artifact_sha256": _sel_art_sha,
                                    "run_id": run_record["id"]}, sort_keys=True),
                    ),
                    # Deployment artifact
                    (
                        f"file://{_dep_art_path}",
                        f"deployment-artifact:{_dep_art_id}|sha256:{_dep_art_sha[:16]}",
                        f"dep-artifact:{_dep_art_id}",
                        json.dumps({"artifact_kind": "deployment_artifact",
                                    "model_artifact_id": _dep_art_id,
                                    "artifact_sha256": _dep_art_sha,
                                    "selection_artifact_id": _sel_art_id,
                                    "training_partition": "all_rows",
                                    "run_id": run_record["id"]}, sort_keys=True),
                    ),
                ]:
                    if not _sel_art_id and "selection" in _evi_key:
                        continue
                    if not _fv and "fv-evidence" in _evi_key:
                        continue
                    if not _dep_art_id and "dep-artifact" in _evi_key:
                        continue
                    try:
                        _evi = self.ledger.add_evidence(
                            _evi_uri,
                            _evi_excerpt,
                            source_kind=EvidenceKind.DATASET,
                            independence_key=_evi_key,
                            content=_evi_content,
                            metadata={"run_id": run_record["id"]},
                        )
                        self.ledger.attest(_claim_id, _evi, Stance.SUPPORT,
                                           actor="artifact-provenance-recorder",
                                           actor_role=ActorRole.TOOL)
                    except Exception:
                        pass

            graph = self.ledger.capture_graph(
                name=f"Case {case_id} after run {run_record['id']}",
                root_claim_ids=import_summary["claim_ids"],
                include_descendants=True,
            )
            engine_result["graph_snapshot_id"] = graph["id"]
            engine_result["belief_import"] = import_summary

            case_now = self.store.get_case(case_id)
            claim_rows = self.store.case_claims(case_id)
            reexam = [row for row in self.memory.list_reexamination("open") if row["claim_id"] in {c["claim_id"] for c in claim_rows}]
            report_bundle = self.reporter.write_bundle(
                run_dir / "report",
                case=case_now,
                plan=plan,
                result=engine_result,
                claim_rows=claim_rows,
                reexamination=reexam,
            )
            engine_result["reports"] = report_bundle
            for role, artifact in report_bundle.items():
                self.store.add_report(case_id, run_record["id"], format=role, path=artifact["path"], content_hash=artifact["sha256"])
            return self.store.finish_run(
                run_record["id"],
                result=engine_result,
                engine_run_id=run_record["id"],
                ledger_path=str(ledger_path.resolve()),
            )
        except Exception as exc:
            failure = {
                "run_id": run_record["id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "ledger_path": str(ledger_path.resolve()),
            }
            self.store.finish_run(
                run_record["id"],
                result=failure,
                engine_run_id=run_record["id"],
                ledger_path=str(ledger_path.resolve()),
                status="failed",
            )
            raise

    def _import_result(
        self,
        *,
        case_id: str,
        case_run_id: str,
        dataset_file: dict[str, Any],
        plan: dict[str, Any],
        result: dict[str, Any],
        dataframe: "pd.DataFrame | None" = None,
        domain: Any = None,
        composite_domain: Any = None,
    ) -> dict[str, Any]:
        from .influence import linear_influence_warning
        from .semantics import (
            apply_functional_form_overrides,
            classify_pairwise_finding,
            derive_finding_record,
        )

        evaluation_metric = result.get("evaluation_metric", plan.get("evaluation_metric", "r2"))
        target_transform = plan.get("target_transform")
        thresholds = plan.get("thresholds", {}) or {}
        min_reliable_partition_n = int(thresholds.get("min_reliable_partition_n", 8))
        hard_refutation_score_ceiling = float(thresholds.get("hard_refutation_score_ceiling", 0.0))

        findings_list = result.get("findings", [])

        # A candidate's own statement asserts predictive performance above
        # baseline only for composite claims ("Y can be predicted by a
        # composite of [...]"). linear_association/group_difference claims
        # assert an association or a group difference, not predictive power —
        # failing to predict well is not, by itself, a claim they contradict.
        def _is_predictive_claim(payload: dict[str, Any]) -> bool:
            return payload.get("kind") == "composite_linear"

        # A directional claim ("a stable positive/negative linear
        # association") is contradicted specifically when the fitted
        # relationship runs the other way. The fitted model is identical
        # across all cross-seed reseeds (train is always the scout partition,
        # unaffected by seed — see UploadedTableDomain.splits), so this only
        # needs one refit on scout, not a per-seed loop.
        def _direction_conflict(cid: str, payload: dict[str, Any]) -> bool:
            if payload.get("kind") != "linear_association":
                return False
            expected = payload.get("expected_direction")
            if expected not in ("positive", "negative"):
                return False
            if domain is None:
                return False
            try:
                cand_obj = Candidate(id=cid, statement="", payload=payload)
                model = domain.refit(cand_obj, domain.scout)
                if not model.get("valid"):
                    return False
                slope = model.get("slope")
                if slope is None:
                    return False
                actual = "positive" if slope > 0 else "negative" if slope < 0 else None
                return actual is not None and actual != expected
            except Exception:
                return False

        # --- Association evidence + missingness receipts (per candidate) -----
        # Computed from the full dataset, independent of predictive utility, so
        # a real association is never collapsed into a single predictive score.
        assoc_by_cid: dict[str, dict[str, Any] | None] = {}
        missing_by_cid: dict[str, dict[str, Any] | None] = {}
        if dataframe is not None:
            for finding in findings_list:
                cid = finding["candidate"]["id"]
                payload = finding["candidate"].get("payload", {}) or {}
                try:
                    assoc_by_cid[cid] = _association_evidence(dataframe, payload)
                except Exception:
                    assoc_by_cid[cid] = None
                try:
                    missing_by_cid[cid] = _missingness_receipt(dataframe, payload)
                except Exception:
                    missing_by_cid[cid] = None

        # --- Multivariable derived-variable / target-leakage detection -------
        # Bounded scout-select + held-out validation: flag targets that are
        # near-deterministically reconstructed from a small subset of columns.
        derived_by_target: dict[str, dict[str, Any]] = {}
        if domain is not None and getattr(domain, "scout", None) is not None and len(getattr(domain, "selection", [])) >= 5:
            gen_meta = plan.get("candidate_generation", {}) or {}
            numeric_cols = list(gen_meta.get("numeric_columns", []) or [])
            # Low-cardinality categorical/binary columns (already exclude
            # identifiers / repeated entities) are one-hot encoded inside the
            # detector so a constructed index that depends on a categorical/binary
            # term can be reconstructed.
            categorical_cols = list(gen_meta.get("categorical_columns", []) or [])
            if len(numeric_cols) >= 3:
                try:
                    from .derived import detect_multivariable_derived
                    derived_by_target = detect_multivariable_derived(
                        domain.scout, domain.selection, numeric_cols, numeric_cols,
                        categorical_columns=categorical_cols,
                    )
                except Exception:
                    derived_by_target = {}

        # Group derived-variable records into mutually-near-deterministic
        # clusters (connected components over target↔source links). Within a
        # collinear cluster the *constructed* column is NOT identifiable from
        # values alone, so we flag the whole set and state that explicitly
        # rather than singling out (or auto-contaminating) any one member.
        _parent: dict[str, str] = {}

        def _find(x: str) -> str:
            _parent.setdefault(x, x)
            while _parent[x] != x:
                _parent[x] = _parent[_parent[x]]
                x = _parent[x]
            return x

        def _union(a: str, b: str) -> None:
            _parent[_find(a)] = _find(b)

        for _tgt, _rec in derived_by_target.items():
            for _s in _rec["source_variables"]:
                _union(_tgt, _s)
        derived_clusters: dict[str, dict[str, Any]] = {}
        for _tgt, _rec in derived_by_target.items():
            root = _find(_tgt)
            cl = derived_clusters.setdefault(root, {"members": set(), "records": []})
            cl["members"].update([_tgt, *_rec["source_variables"]])
            cl["records"].append(_rec)

        # --- Subgroup-reversal / regime-dependence detection ----------------
        # For directional (linear_association) candidates, check whether the
        # pooled direction reverses inside stable major subgroups of an eligible
        # categorical conditioning variable (Simpson's paradox). If so, the
        # universal directional claim must NOT be committed.
        from .subgroup import detect_subgroup_reversal

        eligible_conditioning = list(
            (plan.get("candidate_generation", {}) or {}).get("categorical_columns", []) or []
        )
        min_group_n = int(thresholds.get("subgroup_min_group_n", 25))
        subgroup_by_cid: dict[str, dict[str, Any] | None] = {}
        if dataframe is not None and eligible_conditioning:
            for finding in findings_list:
                payload = finding["candidate"].get("payload", {}) or {}
                if payload.get("kind") != "linear_association":
                    continue
                try:
                    subgroup_by_cid[finding["candidate"]["id"]] = detect_subgroup_reversal(
                        dataframe,
                        str(payload.get("predictor")),
                        str(payload.get("outcome")),
                        eligible_conditioning,
                        min_group_n=min_group_n,
                    )
                except Exception:
                    subgroup_by_cid[finding["candidate"]["id"]] = None

        # --- Classification pre-pass ------------------------------------
        # Decide each finding's refined internal type (robust_relation /
        # promising_candidate / falsified_candidate / not_supported_candidate /
        # inconclusive_candidate / functional_form_rejected_candidate /
        # supported_association_candidate / regime_dependent_candidate /
        # untestable_candidate) before doing any claim/evidence writes, so the
        # functional-form pass below can see which candidates in the same run
        # actually survived.
        prelim: dict[str, tuple[str, dict[str, Any] | None]] = {}
        for finding in findings_list:
            cid = finding["candidate"]["id"]
            payload = finding["candidate"].get("payload", {}) or {}
            final_status = finding.get("final_status")
            support = final_status in {"supported", "challenged", "provisional"} and not any(
                attack.get("killed") for attack in finding.get("falsifications", [])
            )
            if support and final_status == "supported":
                prelim[cid] = ("robust_relation", None)
            elif support:
                prelim[cid] = ("promising_candidate", None)
            elif final_status == "refuted":
                ftype, diag = classify_pairwise_finding(
                    finding,
                    min_reliable_partition_n=min_reliable_partition_n,
                    hard_refutation_score_ceiling=hard_refutation_score_ceiling,
                    is_explicit_predictive_claim=_is_predictive_claim(payload),
                    direction_conflict=_direction_conflict(cid, payload),
                )
                # A composite whose ONLY killers are improvement/ablation did
                # predict adequately — it just added no incremental value over
                # the simpler model. That is redundancy, not a refutation, and
                # must be reported distinctly from a model that failed
                # validation (held_out / validation_resample).
                killer_names = {a.get("name") for a in finding.get("falsifications", []) if a.get("killed")}
                if payload.get("kind") == "composite_linear" and killer_names and killer_names <= {"improvement", "ablation"}:
                    reasons = []
                    if "improvement" in killer_names:
                        reasons.append(
                            "adding the extra predictor(s) did not beat the best single predictor by the "
                            "required margin (no incremental value)"
                        )
                    if "ablation" in killer_names:
                        reasons.append(
                            "at least one predictor can be removed with no meaningful performance loss "
                            "(redundant predictor)"
                        )
                    diag = dict(diag or {})
                    diag["reason"] = (
                        "The composite was not refuted; the simpler model dominates: " + "; ".join(reasons) + "."
                    )
                    diag["composite_failure_mode"] = sorted(killer_names)
                    prelim[cid] = ("no_incremental_value_candidate", diag)
                    continue
                # A generated group effect that fails only the standalone
                # predictive bar, but carries a real bootstrap-stable effect
                # size, is a *supported association* (not merely "not
                # supported"). Never upgrade an inconclusive (untrustworthy
                # sample) or a hard refutation.
                if (
                    ftype == "not_supported_candidate"
                    and payload.get("kind") == "group_difference"
                    and _qualifies_as_supported_association(assoc_by_cid.get(cid))
                ):
                    diag = dict(diag or {})
                    diag["reason"] = (
                        "The group effect is a real, bootstrap-stable association (effect size "
                        f"{assoc_by_cid.get(cid, {}).get('effect_size')}, omega² "
                        f"{assoc_by_cid.get(cid, {}).get('omega_squared')}) but did not clear the "
                        "standalone predictive-utility bar. Reported as a supported association with "
                        "limited standalone predictive utility."
                    )
                    prelim[cid] = ("supported_association_candidate", diag)
                else:
                    prelim[cid] = (ftype, diag)
            else:
                prelim[cid] = ("untestable_candidate", None)

            # Subgroup reversal blocks any pooled directional verdict: whether
            # or not the pooled fit would have committed, a universal directional
            # claim cannot stand when it reverses inside the major subgroups.
            reversal = subgroup_by_cid.get(cid)
            if reversal:
                prelim[cid] = ("regime_dependent_candidate", {"reason": reversal["reason"], "subgroup": reversal})

        overrides = apply_functional_form_overrides(
            [(finding, prelim[finding["candidate"]["id"]][0]) for finding in findings_list]
        )

        # --- Preferred-form selection within each relationship family --------
        # Linear + nonlinear forms of the same predictor→outcome pair form one
        # family. Among the SURVIVING forms pick a preferred one on held-out
        # score (never in-sample), preferring the SIMPLER form unless a more
        # complex form beats it by a configured margin — so a quadratic/log-log
        # is only "preferred" when it genuinely generalises better.
        from .semantics import _candidate_family_key
        from .table_domain import FORM_COMPLEXITY

        min_form_improvement = float(thresholds.get("preferred_form_min_improvement", 0.01))
        survivor_prelim = {"robust_relation", "promising_candidate"}

        def _held_out(finding: dict[str, Any]) -> float:
            for a in finding.get("falsifications", []):
                if a.get("name") == "held_out":
                    return float((a.get("detail", {}) or {}).get("score") or 0.0)
            return 0.0

        family_members: dict[Any, list[dict[str, Any]]] = {}
        for finding in findings_list:
            payload = finding["candidate"].get("payload", {}) or {}
            if payload.get("kind") not in ("linear_association", "nonlinear_association"):
                continue
            key = _candidate_family_key(payload)
            if key is None:
                continue
            cid = finding["candidate"]["id"]
            family_members.setdefault(key, []).append({
                "cid": cid,
                "form": payload.get("form", "linear"),
                "prelim": prelim[cid][0],
                "held_out": _held_out(finding),
                "assoc": assoc_by_cid.get(cid),
            })

        model_family_by_cid: dict[str, dict[str, Any]] = {}
        for key, members in family_members.items():
            if len(members) < 2:
                continue
            survivors_in_family = [m for m in members if m["prelim"] in survivor_prelim]
            preferred_cid = None
            preferred_form = None
            if survivors_in_family:
                top = max(m["held_out"] for m in survivors_in_family)
                within = [m for m in survivors_in_family if top - m["held_out"] <= min_form_improvement]
                pref = min(within, key=lambda m: (FORM_COMPLEXITY.get(m["form"], 2), -m["held_out"]))
                preferred_cid, preferred_form = pref["cid"], pref["form"]
            member_summ = [
                {"candidate_id": m["cid"], "form": m["form"], "verdict_group": m["prelim"],
                 "held_out_score": round(m["held_out"], 4)}
                for m in members
            ]
            for m in members:
                info: dict[str, Any] = {
                    "family_size": len(members),
                    "form": m["form"],
                    "members": member_summ,
                    "preferred_candidate_id": preferred_cid,
                    "preferred_form": preferred_form,
                    "is_preferred": m["cid"] == preferred_cid,
                }
                assoc = m.get("assoc") or {}
                if m["form"] == "log_log" and assoc.get("power_law_exponent") is not None:
                    info["power_law_exponent"] = assoc["power_law_exponent"]
                if preferred_form == "log_log" and m["cid"] == preferred_cid and assoc.get("power_law_exponent") is not None:
                    info["preferred_power_law_exponent"] = assoc["power_law_exponent"]
                model_family_by_cid[m["cid"]] = info

        # --- Full-data diagnostic score (report-only, never gates anything) ---
        full_data_scores: dict[str, float | None] = {}
        for finding in findings_list:
            cid = finding["candidate"]["id"]
            payload = finding["candidate"].get("payload", {}) or {}
            kind = payload.get("kind")
            dom = composite_domain if kind == "composite_linear" and composite_domain is not None else domain
            if dom is None:
                full_data_scores[cid] = None
                continue
            try:
                cand_obj = Candidate(id=cid, statement=finding["candidate"].get("statement", ""), payload=payload)
                model = dom.refit(cand_obj, dom.df)
                full_data_scores[cid] = dom.score_metric(cand_obj, model, dom.df) if model.get("valid") else None
            except Exception:
                full_data_scores[cid] = None

        candidate_to_claim: dict[str, str] = {}
        claim_ids: list[str] = []
        for finding in result.get("findings", []):
            candidate = finding["candidate"]
            payload = candidate.get("payload", {})
            scope = {
                "kind": payload.get("kind"),
                "predictor": payload.get("predictor"),
                "outcome": payload.get("outcome"),
                "group": payload.get("group"),
                "evaluation_metric": evaluation_metric,
                "target_transform": target_transform,
            }
            claim_id, _ = self.memory.resolve_or_create_claim(
                candidate["statement"],
                scope=scope,
                claim_type="research_finding",
                metadata={
                    "source_candidate_id": candidate["id"],
                    "generated_from_case": case_id,
                    "dataset_sha256": dataset_file["sha256"],
                    "evaluation_metric": evaluation_metric,
                    "target_transform": target_transform,
                    "composition_strategy": payload.get("composition_strategy"),
                },
            )
            candidate_to_claim[candidate["id"]] = claim_id
            claim_ids.append(claim_id)
            final_status = finding.get("final_status")
            support = final_status in {"supported", "challenged", "provisional"} and not any(
                attack.get("killed") for attack in finding.get("falsifications", [])
            )
            evidence_id = self.memory.attach_run_evidence(
                claim_id,
                run_id=case_run_id,
                finding=finding,
                source_uri=f"file://{result['ledger_path']}#{candidate['id']}",
                support=support,
            )
            self.memory.record_check(
                claim_id,
                name="governed_judge",
                passed=final_status in {"supported", "challenged", "provisional"},
                score=finding.get("verdict", {}).get("score"),
                detail={
                    **finding.get("verdict", {}).get("detail", {}),
                    "evaluation_metric": evaluation_metric,
                    "higher_is_better": result.get("higher_is_better", True),
                    "final_validation_metric_score": finding.get("final_validation_metric_score"),
                    "fitted_coefficients": self._extract_coefficients(finding),
                },
                run_id=case_run_id,
            )
            for attack in finding.get("falsifications", []):
                self.memory.record_check(
                    claim_id,
                    name=attack.get("name", "unknown_falsifier"),
                    passed=not bool(attack.get("killed")),
                    score=attack.get("metric"),
                    detail=attack.get("detail", {}),
                    run_id=case_run_id,
                )
            self.memory.synchronize_status(
                claim_id,
                rationale=f"Imported governed discovery result from run {case_run_id}; evidence {evidence_id}",
            )
            finding_type, classification_diagnostics = prelim[candidate["id"]]
            functional_form_override = None
            if candidate["id"] in overrides:
                finding_type, functional_form_override = overrides[candidate["id"]]
            # Informative-missingness reliability guard: a relationship on a column
            # whose missingness is not-missing-at-random has biased complete-case
            # evidence, so it is reported as provisional rather than committed.
            informative_missingness_warning = None
            _im_cols = set(plan.get("informative_missingness_columns", []) or [])
            if _im_cols:
                _hit = set(_candidate_columns(payload)) & _im_cols
                if _hit:
                    informative_missingness_warning = {
                        "type": "informative_missingness",
                        "columns": sorted(_hit),
                        "note": (
                            "A variable in this relationship has not-missing-at-random missingness; "
                            "complete-case evidence may be biased, so this is reported as provisional, "
                            "not committed."
                        ),
                    }
                    if finding_type == "robust_relation":
                        finding_type = "promising_candidate"
            influence_warning = None
            if (
                dataframe is not None
                and payload.get("kind") == "linear_association"
                and finding_type in {"robust_relation", "promising_candidate"}
            ):
                influence_warning = linear_influence_warning(
                    dataframe,
                    str(payload.get("predictor")),
                    str(payload.get("outcome")),
                )
            detail = derive_finding_record(
                finding,
                finding_type,
                influence_warning=influence_warning,
                classification_diagnostics=classification_diagnostics,
                functional_form_override=functional_form_override,
                full_data_score=full_data_scores.get(candidate["id"]),
                is_predictive_claim=_is_predictive_claim(payload),
                association_evidence=assoc_by_cid.get(candidate["id"]),
                missingness=missing_by_cid.get(candidate["id"]),
                model_family=model_family_by_cid.get(candidate["id"]),
                subgroup_warning=subgroup_by_cid.get(candidate["id"]),
            )
            if informative_missingness_warning:
                detail["informative_missingness_warning"] = informative_missingness_warning
            self.store.link_claim(
                case_id=case_id,
                run_id=case_run_id,
                claim_id=claim_id,
                finding_type=finding_type,
                source_candidate_id=candidate["id"],
                finding_detail=detail,
            )

        for finding in result.get("findings", []):
            candidate = finding["candidate"]
            child_id = candidate_to_claim[candidate["id"]]
            parents = [candidate_to_claim[p] for p in candidate.get("parents", []) if p in candidate_to_claim]
            if parents:
                self.ledger.add_proof(
                    child_id,
                    parents,
                    rule="approved_discovery_plan_derivation",
                    metadata={"case_id": case_id, "run_id": case_run_id},
                    actor="research-compiler",
                    actor_role=ActorRole.TOOL,
                )

        # Scoped per-subgroup claims for every detected reversal: the pooled
        # directional claim is blocked (regime_dependent) and the real
        # within-subgroup associations are recorded as supported scoped claims.
        for cid, reversal in subgroup_by_cid.items():
            if not reversal:
                continue
            for si, scoped in enumerate(reversal.get("scoped_claims", []), start=1):
                scoped_claim_id, _ = self.memory.resolve_or_create_claim(
                    scoped["statement"],
                    scope={
                        "dataset_sha256": dataset_file["sha256"],
                        "conditioning_variable": scoped["group_col"],
                        "group_value": scoped["group_value"],
                        "direction": scoped["direction"],
                        "type": "scoped_association",
                    },
                    claim_type="research_finding",
                    metadata={
                        "case_id": case_id,
                        "generated_from_case": case_id,
                        "parent_candidate_id": cid,
                        "conditioning_variable": scoped["group_col"],
                    },
                )
                evidence = self.ledger.add_evidence(
                    f"file://{dataset_file['stored_path']}",
                    scoped["statement"],
                    source_kind=EvidenceKind.DATASET,
                    independence_key=f"dataset:{dataset_file['sha256']}:scoped:{cid}:{scoped['group_value']}",
                    content=json.dumps(scoped, sort_keys=True),
                    metadata={"case_id": case_id, "conditioning_variable": scoped["group_col"]},
                )
                self.ledger.attest(scoped_claim_id, evidence, Stance.SUPPORT,
                                   actor="subgroup-reversal-detector", actor_role=ActorRole.TOOL)
                self.memory.synchronize_status(scoped_claim_id, rationale="Scoped within-subgroup association")
                claim_ids.append(scoped_claim_id)
                self.store.link_claim(
                    case_id=case_id,
                    run_id=case_run_id,
                    claim_id=scoped_claim_id,
                    finding_type="scoped_association",
                    source_candidate_id=f"scoped:{cid}:{si}",
                    finding_detail={
                        "hypothesis_text": scoped["statement"],
                        "finding_type": "scoped_association",
                        "verdict": "supported_association",
                        "is_candidate_hypothesis": False,
                        "association_evidence": {
                            "effect_size_metric": "within_group_slope_sign",
                            "direction": scoped["direction"],
                            "sign_stability": scoped["sign_stability"],
                            "n": scoped["n"],
                        },
                        "scope": {
                            "conditioning_variable": scoped["group_col"],
                            "group_value": scoped["group_value"],
                        },
                        "parent_candidate_id": cid,
                    },
                )

        artifact_count = 0
        for index, item in enumerate(plan.get("quality_findings", []), start=1):
            text = f"{item.get('title')}: {item.get('detail')}"
            claim_id, _ = self.memory.resolve_or_create_claim(
                text,
                scope={"dataset_sha256": dataset_file["sha256"], "type": item.get("type")},
                claim_type="data_quality",
                metadata={"severity": item.get("severity"), "case_id": case_id},
            )
            evidence = self.ledger.add_evidence(
                f"file://{dataset_file['stored_path']}",
                text,
                source_kind=EvidenceKind.DATASET,
                independence_key=f"dataset:{dataset_file['sha256']}",
                content=json.dumps(item, sort_keys=True),
                metadata={"case_id": case_id, "quality_finding_index": index},
            )
            self.ledger.attest(claim_id, evidence, Stance.SUPPORT, actor="data-profiler", actor_role=ActorRole.TOOL)
            self.memory.synchronize_status(claim_id, rationale="Deterministic data-profile finding")
            claim_ids.append(claim_id)
            quality_detail: dict[str, Any] = {
                "hypothesis_text": text,
                "finding_type": item.get("type", "data_quality"),
                "verdict": "artifact",
                "is_candidate_hypothesis": True,
                "title": item.get("title"),
                "detail": item.get("detail"),
                "column": item.get("column"),
            }
            if item.get("informative_missingness"):
                quality_detail["informative_missingness"] = item["informative_missingness"]
            self.store.link_claim(
                case_id=case_id,
                run_id=case_run_id,
                claim_id=claim_id,
                finding_type=item.get("type", "data_quality"),
                source_candidate_id=f"quality:{index}",
                finding_detail=quality_detail,
            )

        for artifact in plan.get("structural_relations", []):
            claim_id, _ = self.memory.resolve_or_create_claim(
                artifact["statement"],
                scope={
                    "dataset_sha256": dataset_file["sha256"],
                    "artifact_kind": artifact.get("artifact_kind"),
                    "columns": artifact.get("columns"),
                },
                claim_type="structural_artifact",
                metadata={"case_id": case_id, "artifact_kind": artifact.get("artifact_kind")},
            )
            evidence = self.ledger.add_evidence(
                f"file://{dataset_file['stored_path']}",
                artifact["statement"],
                source_kind=EvidenceKind.DATASET,
                independence_key=f"dataset:{dataset_file['sha256']}:{artifact['id']}",
                content=json.dumps(artifact, sort_keys=True),
                metadata={"case_id": case_id, "artifact_kind": artifact.get("artifact_kind")},
            )
            self.ledger.attest(claim_id, evidence, Stance.SUPPORT, actor="artifact-detector", actor_role=ActorRole.TOOL)
            claim_ids.append(claim_id)
            artifact_detail: dict[str, Any] = {
                "hypothesis_text": artifact["statement"],
                "finding_type": "artifact",
                "verdict": "artifact",
                "artifact_kind": artifact.get("artifact_kind"),
                "detail": artifact.get("detail", ""),
                "is_candidate_hypothesis": True,
            }
            if artifact.get("artifact_kind") in ("near_duplicate_copy", "near_copy_affine"):
                artifact_detail["artifact_warning"] = {
                    "type": "target_leakage_near_copy",
                    "leakage_risk": artifact.get("leakage_risk", "high"),
                    "similarity_metric": artifact.get("similarity_metric"),
                    "similarity": artifact.get("similarity"),
                    "correlation": artifact.get("correlation"),
                    "residual_variance_ratio": artifact.get("residual_variance_ratio"),
                    "suspected_source_column": artifact.get("suspected_source_column"),
                    "derived_column_candidate": artifact.get("derived_column_candidate"),
                    "disposition": artifact.get("disposition", "downgraded_to_artifact"),
                    "columns": artifact.get("columns"),
                    # For a symmetric near-copy the derivation direction cannot be
                    # determined from values alone, so neither column's other
                    # relationships are auto-contaminated (the source is not flagged).
                    "derivation_direction": "undetermined",
                }
            elif artifact.get("artifact_kind") in ("derived_field", "near_derived_field"):
                artifact_detail["artifact_warning"] = {
                    "type": "derived_dependency_field",
                    "leakage_risk": artifact.get("leakage_risk", "high"),
                    "op": artifact.get("op"),
                    "inputs": artifact.get("inputs"),
                    "similarity": artifact.get("similarity"),
                    "residual_variance_ratio": artifact.get("residual_variance_ratio"),
                    "disposition": artifact.get("disposition", "downgraded_to_artifact"),
                    "note": (
                        "This column is a (near-)deterministic algebraic function of other columns "
                        "(an accounting identity), so its relationships are derived, not independent "
                        "discoveries."
                    ),
                }
            self.store.link_claim(
                case_id=case_id,
                run_id=case_run_id,
                claim_id=claim_id,
                finding_type="artifact",
                source_candidate_id=artifact["id"],
                finding_detail=artifact_detail,
            )
            artifact_count += 1

        # Multivariable derived-variable artifacts: one per near-deterministic
        # dependency cluster. The constructed column cannot be identified from
        # values alone, so every member is named as a candidate and the
        # direction is stated as undetermined (no member is auto-contaminated).
        for _root, cluster in derived_clusters.items():
            members = sorted(cluster["members"])
            best_rec = max(cluster["records"], key=lambda r: r["held_out_r2"])
            statement = (
                f"Columns [{', '.join(members)}] are mutually near-deterministic — at least one is a "
                f"likely constructed/derived index ({best_rec['construction']}), reconstructed from the "
                f"others at held-out R²={best_rec['held_out_r2']} (residual variance ratio="
                f"{best_rec['residual_variance_ratio']}). Which column is the derived index cannot be "
                f"determined from the data alone; treat this set as artifact-contaminated, not as "
                f"independent discoveries."
            )
            d_claim_id, _ = self.memory.resolve_or_create_claim(
                statement,
                scope={
                    "dataset_sha256": dataset_file["sha256"],
                    "artifact_kind": "likely_derived_variable",
                    "members": members,
                },
                claim_type="structural_artifact",
                metadata={"case_id": case_id, "artifact_kind": "likely_derived_variable"},
            )
            evidence = self.ledger.add_evidence(
                f"file://{dataset_file['stored_path']}",
                statement,
                source_kind=EvidenceKind.DATASET,
                independence_key=f"dataset:{dataset_file['sha256']}:derived:{'|'.join(members)}",
                content=json.dumps({"members": members, "records": cluster["records"]}, sort_keys=True),
                metadata={"case_id": case_id, "artifact_kind": "likely_derived_variable"},
            )
            self.ledger.attest(d_claim_id, evidence, Stance.SUPPORT,
                               actor="multivariable-derived-detector", actor_role=ActorRole.TOOL)
            self.memory.synchronize_status(d_claim_id, rationale="Multivariable derived-variable artifact")
            claim_ids.append(d_claim_id)
            self.store.link_claim(
                case_id=case_id,
                run_id=case_run_id,
                claim_id=d_claim_id,
                finding_type="artifact",
                source_candidate_id=f"derived:{'|'.join(members)}",
                finding_detail={
                    "hypothesis_text": statement,
                    "finding_type": "artifact",
                    "verdict": "artifact",
                    "artifact_kind": "likely_derived_variable",
                    "is_candidate_hypothesis": True,
                    "artifact_warning": {
                        "type": "likely_derived_variable",
                        "leakage_risk": "high",
                        "member_columns": members,
                        "derivation_direction": "undetermined",
                        "best_reconstruction": {
                            "target_candidate": best_rec["target"],
                            "source_variables": best_rec["source_variables"],
                            "coefficients": best_rec["coefficients"],
                            "reconstruction_metric": "held_out_r2",
                            "held_out_r2": best_rec["held_out_r2"],
                            "residual_variance_ratio": best_rec["residual_variance_ratio"],
                            "best_single_predictor_r2": best_rec["best_single_predictor_r2"],
                            "margin_over_best_single": best_rec["margin_over_best_single"],
                            "refit_median_r2": best_rec["refit_median_r2"],
                            "valid_refit_count": best_rec.get("valid_refit_count"),
                            "refit_attempts": best_rec.get("refit_attempts"),
                            "construction": best_rec["construction"],
                        },
                        "disposition": "artifact_qualified",
                    },
                },
            )
            artifact_count += 1

        return {
            "claim_ids": list(dict.fromkeys(claim_ids)),
            "candidate_to_claim": candidate_to_claim,
            "artifact_count": artifact_count,
        }

    @staticmethod
    def _extract_coefficients(finding: dict[str, Any]) -> dict[str, Any] | None:
        """Return fitted model coefficients from a finding if available."""
        # Coefficients are NOT stored in the ledger finding dict (the engine
        # discards them after scoring).  They would need to be recomputed via
        # refit() on the training data.  This is a known gap; the reference is
        # stored as None here so the graph metadata records the absence.
        return None

    # ------------------------------------------------------------------
    # Belief memory facade
    # ------------------------------------------------------------------
    def claim_history(self, claim_id: str) -> dict[str, Any]:
        return self.memory.reconstruct_history(claim_id)

    def supersede_claim(self, claim_id: str, *, new_statement: str, rationale: str) -> dict[str, Any]:
        return self.memory.supersede(claim_id, new_statement, rationale=rationale)

    def revoke_evidence(self, evidence_id: str, *, rationale: str) -> dict[str, Any]:
        return self.memory.revoke_evidence(evidence_id, rationale=rationale)

    def reexamination_queue(self) -> list[dict[str, Any]]:
        return self.memory.list_reexamination("open")

    def add_contradiction(self, claim_a: str, claim_b: str, *, rationale: str) -> dict[str, Any]:
        return self.memory.add_contradiction(claim_a, claim_b, rationale=rationale)
