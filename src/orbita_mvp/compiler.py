from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .table_domain import generate_table_candidates


class ResearchCompiler:
    """Translate a case into an explicit, reviewable, frozen analysis plan."""

    def compile(self, case: dict[str, Any], *, max_candidates: int = 60) -> dict[str, Any]:
        files = case.get("files", [])
        tables = [f for f in files if f.get("artifact_kind") == "table" and f.get("extracted_path")]
        texts = [f for f in files if f.get("artifact_kind") == "text" and f.get("extracted_path")]
        if not tables:
            return {
                "schema_version": "orbita-research-plan/0.1",
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
        candidates, generation = generate_table_candidates(
            df,
            goal=case.get("goal", ""),
            max_candidates=max_candidates,
        )
        profile = selected.get("profile", {})
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
        quality_findings = self._quality_findings(profile)
        return {
            "schema_version": "orbita-research-plan/0.1",
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
            "routes": ["uploaded_table_association", "data_quality_audit", "belief_graph_import"],
            "thresholds": {
                "commit_at": 0.25,
                "baseline_margin": 0.05,
                "held_out_min": 0.15,
                "cross_seed_count": 9,
                "cross_seed_min": 0.15,
                "cross_seed_max_spread": 0.65,
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
                findings.append(
                    {
                        "type": "artifact_guard",
                        "severity": "low",
                        "title": f"Identifier excluded: {column['name']}",
                        "detail": "The column appears unique per row and is excluded from automatic relation mining.",
                    }
                )
        return findings
