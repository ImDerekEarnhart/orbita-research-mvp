from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import validate_metric
from .table_domain import generate_table_candidates

PLAN_SCHEMA_V02 = "orbita-research-plan/0.2"
PLAN_SCHEMA_V03 = "orbita-research-plan/0.3"

# v0.2 field set — historical plans (Run 001, Run 002A). MUST NOT be extended.
_IMMUTABLE_FIELDS_V02 = (
    "target_transform",
    "outcome_domain",
    "evaluation_metric",
    "thresholds",
    "candidate_generation",
)

# v0.3 field set — new plans. Adds ablation policy and composition strategy.
_IMMUTABLE_FIELDS_V03 = (
    "target_transform",
    "outcome_domain",
    "evaluation_metric",
    "ablation_metric",
    "composition_strategy",
    "thresholds",
    "candidate_generation",
)

# Legacy alias — keep for any code that references IMMUTABLE_PLAN_FIELDS directly.
IMMUTABLE_PLAN_FIELDS = _IMMUTABLE_FIELDS_V03


def compute_plan_hash(plan: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON encoding of immutable fields for this plan's schema.

    Schema routing
    --------------
    orbita-research-plan/0.2  → original five-field set (Run 001, Run 002A compatible)
    orbita-research-plan/0.3  → extended set (ablation_metric, composition_strategy added)

    Historical plan hashes are preserved: a v0.2 plan always hashes with the v0.2
    field set, so stored hashes remain verifiable even after the engine is upgraded.
    """
    schema = plan.get("schema_version", PLAN_SCHEMA_V02)
    fields = _IMMUTABLE_FIELDS_V02 if schema == PLAN_SCHEMA_V02 else _IMMUTABLE_FIELDS_V03
    subset = {k: plan.get(k) for k in fields}
    canonical = json.dumps(subset, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_plan_schema_executable(plan: dict[str, Any]) -> None:
    """Raise if the plan's schema cannot be executed by this engine version.

    Historical v0.2 plans remain auditable and their hashes still verify, but
    they cannot be executed under the v0.3 engine semantics (different ablation
    policy, backward elimination, two-artifact pipeline).  Re-compile to proceed.
    """
    schema = plan.get("schema_version", PLAN_SCHEMA_V02)
    if schema != PLAN_SCHEMA_V03:
        raise ValueError(
            f"Plan schema {schema!r} cannot be executed by this engine. "
            "Historical plans remain auditable and their hashes still verify, "
            "but new execution requires plan schema orbita-research-plan/0.3. "
            "Re-compile the case to generate an executable plan."
        )


class ResearchCompiler:
    """Translate a case into an explicit, reviewable, frozen analysis plan."""

    def compile(
        self,
        case: dict[str, Any],
        *,
        max_candidates: int = 60,
        target_transform: str | None = None,
        outcome_domain: str | None = None,
        evaluation_metric: str = "r2",
        ablation_metric: str | None = None,
        confirmation_fraction: float = 0.25,
        final_validation_fraction: float = 0.15,
        target_column: str | None = None,
    ) -> dict[str, Any]:
        validate_metric(evaluation_metric)
        if ablation_metric is not None:
            validate_metric(ablation_metric)
        resolved_ablation_metric = ablation_metric if ablation_metric is not None else evaluation_metric
        files = case.get("files", [])
        tables = [f for f in files if f.get("artifact_kind") == "table" and f.get("extracted_path")]
        texts = [f for f in files if f.get("artifact_kind") == "text" and f.get("extracted_path")]
        if not tables:
            return {
                "schema_version": "orbita-research-plan/0.2",
                "mode": case.get("mode", "open_discovery"),
                "goal": case.get("goal", ""),
                "status": "needs_data",
                "blocking_questions": [
                    "No parsed tabular dataset was found. Upload CSV, Excel, JSON records, JSONL, or Parquet for automated discovery."
                ],
                "source_context": [self._source_summary(item) for item in texts],
                "routes": [],
                "candidates": [],
                "assumptions": [],
            }
        selected = max(tables, key=lambda item: int(item.get("profile", {}).get("rows", 0)))
        df = pd.read_csv(Path(selected["extracted_path"]))
        profile = selected.get("profile", {})

        scout_fraction = 1.0 - confirmation_fraction - final_validation_fraction
        if scout_fraction < 0.3:
            raise ValueError(
                f"scout_fraction ({scout_fraction:.3f}) is too small; "
                "reduce confirmation_fraction or final_validation_fraction"
            )

        assumptions = [
            {
                "id": "unit_of_analysis",
                "statement": "Each row is treated as one independent unit unless the researcher says otherwise.",
                "severity": "high",
                "requires_review": True,
            },
            {
                "id": "association_not_causation",
                "statement": "Automatically generated relationships are treated as associations, not causal effects.",
                "severity": "high",
                "requires_review": False,
            },
            {
                "id": "missing_values",
                "statement": "Each candidate is evaluated on rows containing the variables required by that candidate.",
                "severity": "medium",
                "requires_review": False,
            },
        ]
        identifier_columns = [
            c["name"] for c in profile.get("column_profiles", []) if c.get("inferred_role") == "identifier"
        ]
        candidates, generation = generate_table_candidates(
            df,
            goal=case.get("goal", ""),
            max_candidates=max_candidates,
            exclude_columns=identifier_columns,
            target_column=target_column,
        )
        # Augment generation dict with all partition fractions so the service
        # can reconstruct the exact same split as candidate generation used.
        generation["confirmation_fraction"] = confirmation_fraction
        generation["final_validation_fraction"] = final_validation_fraction
        generation["scout_fraction"] = scout_fraction

        quality_findings = self._quality_findings(profile)

        plan: dict[str, Any] = {
            "schema_version": PLAN_SCHEMA_V03,
            "mode": case.get("mode", "open_discovery"),
            "goal": case.get("goal", ""),
            "status": "ready_for_review" if candidates else "no_candidates",
            "selected_dataset": {
                "file_id": selected["id"],
                "name": selected["original_name"],
                "normalized_path": selected["extracted_path"],
                "sha256": selected["sha256"],
                "rows": profile.get("rows"),
                "columns": profile.get("columns"),
            },
            "source_context": [self._source_summary(item) for item in texts],
            "data_profile": profile,
            "quality_findings": quality_findings,
            "excluded_from_candidate_generation": identifier_columns,
            "candidate_generation": generation,
            "structural_relations": generation.get("structural_relations", []),
            "routes": ["uploaded_table_association", "data_quality_audit", "belief_graph_import"],
            "target_transform": target_transform,
            "outcome_domain": outcome_domain,
            "target_column": target_column,
            "evaluation_metric": evaluation_metric,
            "ablation_metric": resolved_ablation_metric,
            "composition_strategy": "composition_v1_1_backward_elimination",
            "thresholds": {
                "commit_at": 0.25,
                "baseline_margin": 0.05,
                "held_out_min": 0.15,
                "cross_seed_count": 9,
                "cross_seed_min": 0.15,
                "cross_seed_max_spread": 0.65,
                # Number of independent fresh-partition refits used by the
                # diagnostic-only RepeatedRefitValidator (model reproducibility).
                "repeated_refit_count": 12,
                # A more complex form (e.g. quadratic, log-log) is only chosen as
                # the preferred member of a relationship family when it beats the
                # simplest within-margin form's held-out score by at least this.
                "preferred_form_min_improvement": 0.01,
                "composite_min_predictors": 2,
                "composite_max_predictors": 10,
                "composite_min_improvement": 0.01,
                "ablation_min_contribution": 0.01,
                "ablation_min_absolute_improvement": 0.01,
                "ablation_min_relative_improvement": None,
                # Below this held-out/cross-seed test-partition size, a killed
                # falsifier is reclassified as "inconclusive" rather than
                # "refuted" — R² computed on a handful of rows is dominated by
                # whichever single row landed in the split and cannot be
                # trusted to reject a real relationship.
                "min_reliable_partition_n": 8,
                # Below this magnitude, a killed falsifier's score is treated
                # as "did not clear the bar" rather than "actively
                # contradicted" (hard evidence against requires a score
                # meaningfully worse than a trivial baseline, i.e. negative).
                "hard_refutation_score_ceiling": 0.0,
            },
            "candidates": candidates,
            "assumptions": assumptions,
            "blocking_questions": [],
            "report_modules": [
                "source_inventory",
                "data_interpretation",
                "quality_and_errors",
                "surviving_findings",
                "failed_candidates",
                "assumptions",
                "limitations",
                "recommended_tests",
                "provenance_and_receipts",
            ],
        }
        plan["plan_hash"] = compute_plan_hash(plan)
        return plan

    def validate_external_plan(self, case: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        required = {"schema_version", "selected_dataset", "candidates", "thresholds"}
        missing = sorted(required - set(plan))
        if missing:
            raise ValueError(f"Plan is missing required fields: {', '.join(missing)}")
        file_ids = {item["id"] for item in case.get("files", [])}
        if plan["selected_dataset"].get("file_id") not in file_ids:
            raise ValueError("selected_dataset.file_id does not belong to this case")
        seen: set[str] = set()
        for candidate in plan.get("candidates", []):
            for field in ("id", "statement", "kind"):
                if field not in candidate:
                    raise ValueError(f"Candidate missing {field}")
            if candidate["id"] in seen:
                raise ValueError(f"Duplicate candidate id: {candidate['id']}")
            seen.add(candidate["id"])
        # Validate metric if present; default to r2 for backward compatibility.
        metric = plan.get("evaluation_metric", "r2")
        validate_metric(metric)
        # Always recompute plan_hash after validation so revisions get a fresh hash.
        plan["plan_hash"] = compute_plan_hash(plan)
        return plan

    def _source_summary(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "file_id": item["id"],
            "name": item["original_name"],
            "sha256": item["sha256"],
            "profile": item.get("profile", {}),
            "role": "context_only_in_v0.1",
        }

    def _quality_findings(self, profile: dict[str, Any]) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        if profile.get("duplicates", 0):
            findings.append(
                {
                    "type": "data_error",
                    "severity": "medium",
                    "title": "Duplicate rows detected",
                    "detail": f"{profile['duplicates']} exact duplicate rows were found.",
                }
            )
        for column in profile.get("column_profiles", []):
            if column.get("missing_fraction", 0) >= 0.3:
                findings.append(
                    {
                        "type": "data_quality",
                        "severity": "medium",
                        "title": f"High missingness in {column['name']}",
                        "detail": f"{column['missing_fraction']:.1%} of values are missing.",
                    }
                )
            if column.get("inferred_role") == "identifier":
                signals = column.get("identifier_signals", {}) or {}
                shape = signals.get("shape", "near-unique")
                uniq = signals.get("uniqueness")
                uniq_txt = f" ({uniq:.0%} unique)" if isinstance(uniq, (int, float)) else ""
                findings.append(
                    {
                        "type": "artifact_guard",
                        "severity": "low",
                        "title": f"Identifier excluded: {column['name']}",
                        "detail": (
                            f"Detected as a likely row identifier (shape: {shape}{uniq_txt}) and "
                            f"excluded from automatic relation mining. It is recorded as a data-quality "
                            f"artifact rather than silently dropped."
                        ),
                        "identifier_signals": signals,
                        "column": column["name"],
                    }
                )
        return findings
