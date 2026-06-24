from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from orbita import ActorRole, EpistemicLedger, EvidenceKind, Stance
from orbita_discovery.core import Engine, Ledger, finding_to_dict, survivors
from orbita_discovery.falsifiers import BaselineFalsifier, CrossSeedFalsifier, HeldOutFalsifier
from orbita_discovery.judges import GatedJudge

from .compiler import ResearchCompiler
from .ingestion import ArtifactIngestor
from .memory import BeliefMemory
from .reporting import ReportCompiler
from .storage import CaseStore
from .table_domain import UploadedTableDomain


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

    def compile_case(self, case_id: str, *, max_candidates: int = 60) -> dict[str, Any]:
        case = self.store.get_case(case_id)
        plan = self.compiler.compile(case, max_candidates=max_candidates)
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
    def run_case(self, case_id: str, *, plan_id: str | None = None, auto_approve: bool = False) -> dict[str, Any]:
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

        selected_file = self.store.get_file(plan["selected_dataset"]["file_id"])
        df = pd.read_csv(selected_file["extracted_path"])
        run_record = self.store.create_run(case_id, plan_record["id"])
        run_dir = self.store.case_dir(case_id) / "runs" / run_record["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "discovery_ledger.jsonl"

        try:
            thresholds = plan.get("thresholds", {})
            domain = UploadedTableDomain(
                df,
                plan["candidates"],
                scout_fraction=float(plan.get("candidate_generation", {}).get("scout_fraction", 0.6)),
                seed=int(plan.get("candidate_generation", {}).get("seed", 20260623)),
            )
            judge = GatedJudge(
                commit_at=float(thresholds.get("commit_at", 0.25)),
                baseline_margin=float(thresholds.get("baseline_margin", 0.05)),
            )
            falsifiers = [
                BaselineFalsifier(margin=float(thresholds.get("baseline_margin", 0.05))),
                HeldOutFalsifier(min_score=float(thresholds.get("held_out_min", 0.15))),
                CrossSeedFalsifier(
                    seeds=int(thresholds.get("cross_seed_count", 9)),
                    min_median=float(thresholds.get("cross_seed_min", 0.15)),
                    max_spread=thresholds.get("cross_seed_max_spread", 0.65),
                ),
            ]
            engine = Engine(judge, falsifiers, Ledger(ledger_path))
            engine.run(domain)
            engine_result = {
                "run_id": run_record["id"],
                "engine": "orbita-discovery-kit/0.2-compatible",
                "domain": "uploaded_table",
                "ledger_path": str(ledger_path.resolve()),
                "candidate_count": len(engine.ledger.entries),
                "survivor_count": len(survivors(engine.ledger)),
                "survivor_ids": [item.candidate.id for item in survivors(engine.ledger)],
                "findings": [finding_to_dict(item) for item in engine.ledger.entries],
            }

            import_summary = self._import_result(
                case_id=case_id,
                case_run_id=run_record["id"],
                dataset_file=selected_file,
                plan=plan,
                result=engine_result,
            )
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
    ) -> dict[str, Any]:
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
            }
            claim_id, _ = self.memory.resolve_or_create_claim(
                candidate["statement"],
                scope=scope,
                claim_type="research_finding",
                metadata={
                    "source_candidate_id": candidate["id"],
                    "generated_from_case": case_id,
                    "dataset_sha256": dataset_file["sha256"],
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
                detail=finding.get("verdict", {}).get("detail", {}),
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
            finding_type = (
                "robust_relation" if support and final_status == "supported"
                else "candidate_relation" if support
                else "falsified_candidate" if final_status == "refuted"
                else "unresolved_candidate"
            )
            self.store.link_claim(
                case_id=case_id,
                run_id=case_run_id,
                claim_id=claim_id,
                finding_type=finding_type,
                source_candidate_id=candidate["id"],
            )

        # Populate derivation edges after every candidate has a durable claim ID.
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

        # Store data-quality findings as supported claims with dataset provenance.
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
            self.store.link_claim(
                case_id=case_id,
                run_id=case_run_id,
                claim_id=claim_id,
                finding_type=item.get("type", "data_quality"),
                source_candidate_id=f"quality:{index}",
            )
        return {"claim_ids": list(dict.fromkeys(claim_ids)), "candidate_to_claim": candidate_to_claim}

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
