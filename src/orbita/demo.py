from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from .actions import SafeActionRuntime
from .analysis import AnalysisClaimTest, DatasetAnalysisSpec, MetricCondition
from .ledger import EpistemicLedger
from .execution import ContainerExecutionSpec, OutputObligation, ResourceLimits, StagedFile
from .discovery import DiscoverySpec
from .evaluation import ComparativeEvaluationRuntime, default_adversarial_suite
from .models import ActorRole, EvidenceKind, RiskLevel, Stance
from .planner import CandidatePlan, CandidateStep, EpistemicPlanSelector
from .proposals import ModelIdentity
from .support import SupportEngine


def run_demo(root: str | Path | None = None) -> dict:
    base = Path(root) if root else Path(tempfile.mkdtemp(prefix="orbita_demo_"))
    base.mkdir(parents=True, exist_ok=True)
    db_path = base / "orbita.db"
    workspace = base / "workspace"

    with EpistemicLedger(db_path) as ledger:
        support = SupportEngine(ledger)

        # v0.2 uses canonical entities and registered predicate schemas. These
        # remain ordinary ledger claims: evidence, proofs, collapse, and action
        # gating work without a separate knowledge system.
        ledger.add_predicate(
            "is_a",
            domain_type="taxon",
            range_kind="entity",
            range_type="taxon",
        )
        ledger.add_predicate(
            "has_property",
            domain_type="taxon",
            range_kind="entity",
            range_type="property",
        )
        ledger.add_predicate(
            "positively_correlates_with",
            domain_type="measurement",
            range_kind="entity",
            range_type="measurement",
        )
        dog_mammal = ledger.add_relation_claim(
            "Dog",
            "is_a",
            "Mammal",
            subject_type="taxon",
            object_type="taxon",
        )
        mammal_warm = ledger.add_relation_claim(
            "Mammal",
            "has_property",
            "Warm-blooded",
            subject_type="taxon",
            object_type="property",
        )
        dog_warm = ledger.add_relation_claim(
            "Dog",
            "has_property",
            "Warm-blooded",
            subject_type="taxon",
            object_type="property",
        )

        ev_a = ledger.add_evidence(
            "book://zoology-a",
            "Dogs are classified as mammals.",
            source_kind=EvidenceKind.WEB_SOURCE,
            independence_key="publisher:zoology-a",
        )
        ev_a2 = ledger.add_evidence(
            "dataset://taxonomy-a",
            "Canis familiaris belongs to Mammalia.",
            source_kind=EvidenceKind.DATASET,
            independence_key="dataset:taxonomy-a",
        )
        ev_b = ledger.add_evidence(
            "book://biology-b",
            "Mammals regulate a warm internal body temperature.",
            source_kind=EvidenceKind.WEB_SOURCE,
            independence_key="publisher:biology-b",
        )
        ev_b2 = ledger.add_evidence(
            "dataset://physiology-b",
            "Class Mammalia is endothermic.",
            source_kind=EvidenceKind.DATASET,
            independence_key="dataset:physiology-b",
        )
        for evidence_id in (ev_a, ev_a2):
            ledger.attest(dog_mammal, evidence_id, Stance.SUPPORT)
        for evidence_id in (ev_b, ev_b2):
            ledger.attest(mammal_warm, evidence_id, Stance.SUPPORT)

        ledger.add_proof(
            dog_warm,
            [dog_mammal, mammal_warm],
            rule="typed_taxonomy_property_inheritance",
        )

        # v0.5 captures the exact support structure before any evidence changes.
        graph_before = ledger.capture_graph(
            name="dog-warm-before-revocation",
            root_claim_ids=[dog_warm],
        )
        before = support.evaluate(dog_warm).as_dict()
        affected = ledger.revoke_evidence(
            ev_b,
            rationale="Source was withdrawn",
            actor="Derek",
            actor_role=ActorRole.HUMAN,
        )
        after_one_revoke = support.collapse_report(affected)
        ledger.revoke_evidence(
            ev_b2,
            rationale="Dataset label was invalid",
            actor="Derek",
            actor_role=ActorRole.HUMAN,
        )
        after_full_revoke = support.evaluate(dog_warm).as_dict()
        graph_collapsed = ledger.capture_graph(
            name="dog-warm-after-revocation",
            root_claim_ids=[dog_warm],
        )
        collapse_diff = ledger.compare_graphs(
            graph_before,
            graph_collapsed,
            name="dog-warm-collapse",
        )
        collapse_artifacts = ledger.graphs.render_diff(
            collapse_diff,
            output_dir=base / "graph_views" / "collapse",
        )

        direct = ledger.add_evidence(
            "experiment://canine-thermoregulation",
            "Direct physiology measurement supports canine endothermy.",
            source_kind=EvidenceKind.EXPERIMENT_RECEIPT,
            independence_key="experiment:canine-thermoregulation",
        )
        ledger.attest(dog_warm, direct, Stance.SUPPORT)
        after_alternate_support = support.evaluate(dog_warm).as_dict()
        graph_recovered = ledger.capture_graph(
            name="dog-warm-after-alternate-evidence",
            root_claim_ids=[dog_warm],
        )
        recovery_diff = ledger.compare_graphs(
            graph_collapsed,
            graph_recovered,
            name="dog-warm-recovery",
        )
        recovery_artifacts = ledger.graphs.render_diff(
            recovery_diff,
            output_dir=base / "graph_views" / "recovery",
        )

        # Exact structured negation is linked automatically as a contradiction.
        dog_not_warm = ledger.negate_relation_claim(
            dog_warm,
            actor="demo",
            actor_role=ActorRole.TOOL,
        )
        negation_evidence = ledger.add_evidence(
            "proposal://adversarial-negative",
            "Adversarial proposal for contradiction testing only.",
            source_kind=EvidenceKind.MODEL_PROPOSAL,
            independence_key="proposal:adversarial-negative",
        )
        ledger.attest(dog_not_warm, negation_evidence, Stance.SUPPORT)
        challenged = support.evaluate(dog_warm).as_dict()

        # v0.4: a model may propose an exact typed hypothesis, but its own
        # proposal is explicitly non-warranting. Repeating the model proposal
        # still leaves the claim unknown. Only the later dataset receipt can
        # support it.
        model_proposal_json = json.dumps(
            {
                "schema_version": "1.0",
                "task_summary": "Propose a testable dataset hypothesis",
                "proposals": [
                    {
                        "type": "claim",
                        "local_id": "marker_hypothesis",
                        "claim_format": "relation",
                        "relation": {
                            "subject": {"name": "marker_a", "entity_type": "measurement"},
                            "predicate": {"name": "positively_correlates_with"},
                            "object": {
                                "kind": "entity",
                                "entity": {"name": "response", "entity_type": "measurement"},
                            },
                            "qualifiers": {"scope": "demo_dataset"},
                        },
                        "confidence": 0.88,
                        "rationale": "The observed values suggest a positive relation that must be tested.",
                    }
                ],
            }
        )
        proposal_batch = ledger.ingest_model_response(
            model_proposal_json,
            identity=ModelIdentity("demo", "hypothesis-generator", "1"),
            system_prompt="Return typed, testable hypotheses only.",
            user_prompt="Propose a relationship between marker_a and response.",
            generation_parameters={"temperature": 0},
            response_id="demo-proposal-1",
        )
        repeated_proposal_batch = ledger.ingest_model_response(
            model_proposal_json,
            identity=ModelIdentity("demo", "hypothesis-generator", "1"),
            system_prompt="Return typed, testable hypotheses only.",
            user_prompt="Repeat the proposal to test warrant isolation.",
            generation_parameters={"temperature": 0},
            response_id="demo-proposal-2",
        )
        marker_claim = proposal_batch["items"][0]["durable_entity_id"]
        marker_support_before_analysis = support.evaluate(marker_claim).as_dict()

        # v0.3 receipt machinery now supplies external, hash-bound warrant for
        # the exact model-proposed claim. Replay confirms the result without
        # pretending to be an independent dataset.
        dataset_path = base / "marker_response.csv"
        dataset_path.write_text(
            "marker_a,response\n1,2\n2,4\n3,6\n4,8\n5,10\n",
            encoding="utf-8",
        )
        analysis_receipt = ledger.run_analysis(
            DatasetAnalysisSpec(
                dataset_path=dataset_path,
                analysis_type="pearson_correlation",
                parameters={"x": "marker_a", "y": "response"},
                claim_tests=(
                    AnalysisClaimTest(
                        claim_id=marker_claim,
                        metric_path="pearson_r",
                        support_condition=MetricCondition(">=", 0.8),
                        refute_condition=MetricCondition("<=", 0.1),
                        rationale="Predeclared positive-correlation threshold",
                    ),
                ),
                metadata={"demo": True},
            )
        )
        reproduction_receipt = ledger.reproduce_analysis(analysis_receipt["id"])
        marker_support = support.evaluate(marker_claim).as_dict()
        provenance_graph = ledger.capture_graph(
            name="model-proposal-to-dataset-warrant",
            root_claim_ids=[marker_claim],
        )
        provenance_artifacts = ledger.graphs.render_snapshot(
            provenance_graph,
            output_dir=base / "graph_views" / "proposal_receipt",
        )

        contradiction_edges = [
            dict(row)
            for row in ledger.db.conn.execute(
                "SELECT id, claim_a, claim_b, rationale, active FROM contradictions ORDER BY created_at"
            ).fetchall()
        ]

        safe_plan = CandidatePlan(
            name="verified_write",
            goal="Create an auditable result file",
            steps=[
                CandidateStep(
                    intent="Write the current support result",
                    action_type="write_text",
                    args={
                        "path": "results/support.json",
                        "text": json.dumps(after_alternate_support, indent=2),
                    },
                    required_claims=[dog_warm],
                    obligations=[{"type": "file_exists", "path": "results/support.json"}],
                    risk=RiskLevel.LOW,
                )
            ],
        )
        weak_plan = CandidatePlan(
            name="unverified_write",
            goal="Create a result without verification",
            steps=[
                CandidateStep(
                    intent="Write an unverified note",
                    action_type="write_text",
                    args={"path": "results/weak.txt", "text": "trust me"},
                    required_claims=[],
                    obligations=[],
                    risk=RiskLevel.MEDIUM,
                )
            ],
        )
        selector = EpistemicPlanSelector(support)
        chosen, scores = selector.choose([weak_plan, safe_plan])

        runtime = SafeActionRuntime(ledger, workspace)
        plan_id = runtime.persist_plan(chosen)
        receipts = runtime.execute_plan(plan_id)

        # v0.7 stages an exact OCI execution manifest. The demo deliberately
        # stops at the human approval gate because no code may run merely from
        # being generated or staged. Replace the example digest with a real,
        # locally available image digest before approving and executing it.
        # v0.8 creates a restart-safe, counterexample-first investigation.
        # Candidate mining is non-warranting; the exact holdout test is staged
        # behind a separate manifest-bound human approval.
        discovery_dataset = base / "governed_discovery.csv"
        discovery_dataset.write_text(
            "marker,response,noise\n"
            + "\n".join(
                f"{i},{2 * i + ((i % 5) - 2) * 0.1},{(i * 7) % 17}"
                for i in range(1, 81)
            )
            + "\n",
            encoding="utf-8",
        )
        discovery_investigation = ledger.create_discovery(
            DiscoverySpec(
                question="Does marker co-vary with response under held-out testing?",
                dataset_path=discovery_dataset,
                image="python@sha256:" + "0" * 64,
                min_rows=15,
                permutation_trials=39,
                bootstrap_trials=40,
            ),
            actor="demo",
            actor_role=ActorRole.HUMAN,
        )

        staged_execution = ledger.executions.submit(
            ContainerExecutionSpec(
                name="demo proof-carrying calculation",
                image="python@sha256:" + "0" * 64,
                command=("python", "main.py"),
                code_files=(
                    StagedFile(
                        "main.py",
                        text=(
                            "import json\n"
                            "from pathlib import Path\n"
                            "Path('/workspace/output/result.json').write_text("
                            "json.dumps({'supported': True}))\n"
                        ),
                        media_type="text/x-python",
                    ),
                ),
                outputs=(
                    OutputObligation(
                        "result.json",
                        media_type="application/json",
                        json_schema={
                            "type": "object",
                            "required": ["supported"],
                            "properties": {"supported": {"type": "boolean"}},
                        },
                    ),
                ),
                limits=ResourceLimits(
                    timeout_seconds=30, memory_mb=128, cpus=0.5, pids=32
                ),
                required_claims=(dog_warm,),
                metadata={"demo": True, "not_executed": True},
            ),
            actor="demo",
            actor_role=ActorRole.HUMAN,
        )

        evaluation_runtime = ComparativeEvaluationRuntime(ledger, base / "evaluation_workspace")
        evaluation_suite = evaluation_runtime.create_suite(default_adversarial_suite())
        evaluation_runs = {
            profile: evaluation_runtime.create_fixture_run(evaluation_suite["id"], profile)
            for profile in ("base_llm", "rag", "final_answer_verifier", "orbita")
        }
        evaluation_report = evaluation_runtime.compile_report(evaluation_suite["id"])

        return {
            "version": "1.0.0",
            "root": str(base),
            "claim_ids": {
                "dog_is_a_mammal": dog_mammal,
                "mammal_has_property_warm_blooded": mammal_warm,
                "dog_has_property_warm_blooded": dog_warm,
                "dog_not_warm_blooded": dog_not_warm,
                "marker_a_positively_correlates_with_response": marker_claim,
            },
            "typed_claim": ledger.get_relation_claim(dog_warm),
            "support_before_revocation": before,
            "affected_after_first_revocation": after_one_revoke,
            "support_after_both_premise_sources_revoked": after_full_revoke,
            "support_after_alternate_direct_evidence": after_alternate_support,
            "support_with_linked_unwarranted_negation": challenged,
            "auto_linked_contradictions": contradiction_edges,
            "model_proposal_batch": proposal_batch,
            "repeated_model_proposal_batch": repeated_proposal_batch,
            "model_only_support_before_analysis": marker_support_before_analysis,
            "dataset_analysis_receipt": analysis_receipt,
            "dataset_analysis_reproduction": reproduction_receipt,
            "dataset_claim_support": marker_support,
            "graph_snapshots": {
                "before_revocation": graph_before,
                "collapsed": graph_collapsed,
                "recovered": graph_recovered,
                "proposal_to_receipt": provenance_graph,
            },
            "graph_diffs": {
                "collapse": collapse_diff,
                "recovery": recovery_diff,
            },
            "graph_artifacts": {
                "collapse": collapse_artifacts,
                "recovery": recovery_artifacts,
                "proposal_to_receipt": provenance_artifacts,
            },
            "graph_integrity": {
                "collapse_diff": ledger.graphs.verify_diff(collapse_diff["id"]),
                "recovery_diff": ledger.graphs.verify_diff(recovery_diff["id"]),
                "collapse_artifacts": ledger.graphs.verify_artifacts(diff_id=collapse_diff["id"]),
                "recovery_artifacts": ledger.graphs.verify_artifacts(diff_id=recovery_diff["id"]),
                "provenance_snapshot": ledger.graphs.verify_snapshot(provenance_graph["id"]),
                "provenance_artifacts": ledger.graphs.verify_artifacts(snapshot_id=provenance_graph["id"]),
            },
            "selected_plan": chosen.name,
            "plan_scores": [asdict(score) for score in scores],
            "action_receipts": receipts,
            "governed_discovery": {
                "investigation_id": discovery_investigation["id"],
                "status": discovery_investigation["status"],
                "candidate_count": len(discovery_investigation["hypotheses"]),
                "confirmation_run_id": discovery_investigation["resume_cursor"]["confirmation_run_id"],
                "discovery_split_non_warranting": True,
                "executed": False,
                "reason": "The exact confirmation manifest requires separate human approval and an OCI engine",
            },
            "comparative_evaluation": {
                "suite_id": evaluation_suite["id"],
                "suite_integrity_valid": evaluation_runtime.verify_suite(evaluation_suite["id"]),
                "runs": {
                    profile: {
                        "run_id": run["id"],
                        "overall_score": run["metrics"]["overall_score"],
                        "integrity_valid": evaluation_runtime.verify_run(run["id"]),
                    }
                    for profile, run in evaluation_runs.items()
                },
                "report_hash": evaluation_report["report_hash"],
                "report_integrity_valid": evaluation_runtime.verify_report(evaluation_suite["id"]),
                "interpretation_boundary": evaluation_report["report"]["interpretation_boundary"],
            },
            "container_execution": {
                "runtime_status": ledger.executions.runtime_status(),
                "staged_run": staged_execution,
                "executed": False,
                "reason": "Exact human approval and an available OCI engine are required",
            },
        }
