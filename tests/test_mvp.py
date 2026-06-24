from __future__ import annotations

from pathlib import Path

from orbita import EvidenceKind, Stance
from orbita_mvp import ResearchMVP


def write_table(path: Path) -> None:
    rows = ["subject_id,group,x,y,noise"]
    for i in range(1, 31):
        group = "A" if i <= 15 else "B"
        x = i / 3
        y = 2.5 * x + (0.05 if i % 2 else -0.05)
        noise = ((i * 17) % 13) - 6
        rows.append(f"s{i:02d},{group},{x:.4f},{y:.4f},{noise}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_open_discovery_end_to_end(tmp_path: Path):
    table = tmp_path / "data.csv"
    write_table(table)
    with ResearchMVP(tmp_path / "orbita.db", tmp_path / "workspace") as service:
        case = service.create_case(name="Blank goal", goal="")
        uploaded = service.add_file(case["id"], table)
        assert uploaded["artifact_kind"] == "table"
        plan = service.compile_case(case["id"])
        assert plan["plan"]["mode"] == "open_discovery"
        assert plan["plan"]["candidates"]
        service.approve_plan(plan["id"], reviewer="tester")
        run = service.run_case(case["id"], plan_id=plan["id"])
        assert run["status"] == "completed"
        assert run["result"]["candidate_count"] > 0
        assert Path(run["result"]["reports"]["html"]["path"]).exists()
        assert service.store.case_claims(case["id"])


def test_supersession_creates_history_and_reexamination(tmp_path: Path):
    with ResearchMVP(tmp_path / "orbita.db", tmp_path / "workspace") as service:
        parent, _ = service.memory.resolve_or_create_claim("Parent claim")
        evidence = service.ledger.add_evidence(
            "source://parent",
            "Parent claim",
            source_kind=EvidenceKind.CODE_TEST,
            independence_key="parent-test",
        )
        service.ledger.attest(parent, evidence, Stance.SUPPORT)
        service.memory.synchronize_status(parent, rationale="test")
        child, _ = service.memory.resolve_or_create_claim("Child claim")
        service.ledger.add_proof(child, [parent], rule="parent_implies_child")
        assert service.claim_history(child)["current_support"]["state"] == "supported"

        result = service.supersede_claim(
            parent,
            new_statement="Narrower parent claim",
            rationale="New evidence narrows the scope",
        )
        assert result["newer_claim_id"] != parent
        history = service.claim_history(parent)
        assert len(history["supersession_chain"]) == 2
        assert service.claim_history(child)["current_support"]["state"] in {"unknown", "unsupported"}
        assert any(item["claim_id"] == child for item in service.reexamination_queue())


def test_alternate_derivation_prevents_total_collapse(tmp_path: Path):
    with ResearchMVP(tmp_path / "orbita.db", tmp_path / "workspace") as service:
        parents = []
        for label in ("A", "B"):
            claim, _ = service.memory.resolve_or_create_claim(label)
            evidence = service.ledger.add_evidence(
                f"source://{label}",
                label,
                source_kind=EvidenceKind.CODE_TEST,
                independence_key=label,
            )
            service.ledger.attest(claim, evidence, Stance.SUPPORT)
            service.memory.synchronize_status(claim, rationale="test")
            parents.append(claim)
        child, _ = service.memory.resolve_or_create_claim("C")
        service.ledger.add_proof(child, [parents[0]], rule="route_one")
        service.ledger.add_proof(child, [parents[1]], rule="route_two")
        service.supersede_claim(parents[0], new_statement="A narrowed", rationale="scope correction")
        report = service.claim_history(child)["current_support"]
        assert report["state"] == "supported"


def test_contradiction_propagates_and_queues_dependents(tmp_path: Path):
    with ResearchMVP(tmp_path / "orbita.db", tmp_path / "workspace") as service:
        a, _ = service.memory.resolve_or_create_claim("Treatment improves response")
        b, _ = service.memory.resolve_or_create_claim("Treatment does not improve response")
        for claim_id, label in ((a, "positive"), (b, "negative")):
            evidence = service.ledger.add_evidence(
                f"source://{label}",
                label,
                source_kind=EvidenceKind.CODE_TEST,
                independence_key=label,
            )
            service.ledger.attest(claim_id, evidence, Stance.SUPPORT)
            service.memory.synchronize_status(claim_id, rationale="test")

        child, _ = service.memory.resolve_or_create_claim("Use treatment in protocol")
        service.memory.add_derivation(child, [a], rule="action_from_effect")
        assert service.claim_history(child)["current_support"]["state"] == "supported"

        result = service.add_contradiction(a, b, rationale="Two supported conclusions are incompatible")
        assert result["contradiction_id"].startswith("ctr_")
        assert service.claim_history(a)["current_support"]["state"] == "challenged"
        assert any(item["claim_id"] in {a, child} for item in service.reexamination_queue())


def test_plan_revision_creates_new_immutable_version(tmp_path: Path):
    table = tmp_path / "data.csv"
    write_table(table)
    with ResearchMVP(tmp_path / "orbita.db", tmp_path / "workspace") as service:
        case = service.create_case(name="Revision test", goal="")
        service.add_file(case["id"], table)
        first = service.compile_case(case["id"])
        edited = dict(first["plan"])
        edited["thresholds"] = {**edited["thresholds"], "commit_at": 0.4}
        second = service.revise_plan(first["id"], edited, compiler="test-review")
        assert second["id"] != first["id"]
        assert second["version"] == first["version"] + 1
        assert second["plan_hash"] != first["plan_hash"]
        assert service.store.get_plan(first["id"])["plan"]["thresholds"]["commit_at"] == 0.25
