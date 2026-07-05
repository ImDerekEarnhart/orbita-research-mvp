"""Phase 2B tests — observation ledger, counterexample memory, receipts, scoping."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import orbita_mvp.api as api_module
from orbita_mvp import observations, receipts
from orbita_mvp.service import ResearchMVP


USER = "phase2b-user"
PASSWORD = "phase2b-pass"


def _auth(user: str = USER, password: str = PASSWORD) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


@pytest.fixture()
def svc(tmp_path: Path):
    service = ResearchMVP(tmp_path / "phase2b.db", tmp_path / "workspace")
    try:
        yield service
    finally:
        service.close()


@pytest.fixture()
def client(svc: ResearchMVP, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(api_module, "service", svc)
    monkeypatch.setattr(api_module, "_DEMO_USER", USER)
    monkeypatch.setattr(api_module, "_DEMO_PASS", PASSWORD)
    return TestClient(api_module.app, raise_server_exceptions=False)


def write_table(path: Path) -> None:
    """Small table with one real linear relation (x→y) and one noise column,
    so a real run produces both survivors and killed candidates."""
    rows = ["subject_id,group,x,y,noise"]
    for i in range(1, 31):
        group = "A" if i <= 15 else "B"
        x = i / 3
        y = 2.5 * x + (0.05 if i % 2 else -0.05)
        noise = ((i * 17) % 13) - 6
        rows.append(f"s{i:02d},{group},{x:.4f},{y:.4f},{noise}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _run_discovery(svc: ResearchMVP, tmp_path: Path, *, graph_id: str | None):
    table = tmp_path / "data.csv"
    if not table.exists():
        write_table(table)
    case = svc.create_case(name="Phase 2B case", goal="")
    svc.add_file(case["id"], table)
    plan = svc.compile_case(case["id"])
    svc.approve_plan(plan["id"], reviewer="tester")
    run = svc.run_case(case["id"], plan_id=plan["id"], graph_id=graph_id)
    return case, run


# ---------------------------------------------------------------------------
# A. Observation ledger
# ---------------------------------------------------------------------------

def test_ledger_append_creates_entries(svc: ResearchMVP, tmp_path: Path):
    case, run = _run_discovery(svc, tmp_path, graph_id="graph_obs")
    entries = observations.read_observations(svc.store.case_dir(case["id"]))
    kinds = [e["kind"] for e in entries]
    assert observations.KIND_DATASET_IMPORTED in kinds
    assert observations.KIND_RUN_STARTED in kinds
    assert observations.KIND_RUN_RECEIPTS in kinds
    assert observations.KIND_RUN_COMPLETED in kinds
    for entry in entries:
        assert entry["observation_id"].startswith("obs_")
        assert entry["case_id"] == case["id"]
        assert entry["timestamp"]
        assert entry["source"].startswith("orbita_mvp.")
        assert entry["engine_version"]
        assert entry["content_hash"]
    run_entries = [e for e in entries if e["kind"] == observations.KIND_RUN_STARTED]
    assert run_entries[0]["graph_id"] == "graph_obs"
    assert run_entries[0]["run_id"] == run["id"]
    assert run_entries[0]["dataset_ids"]


def test_ledger_is_append_only_and_hash_chained(svc: ResearchMVP, tmp_path: Path):
    case = svc.store.create_case(name="Chain", goal="")
    case_dir = svc.store.case_dir(case["id"])
    first = observations.record_observation(
        case_dir, case_id=case["id"], source="test", kind="manual_a", payload={"n": 1}
    )
    second = observations.record_observation(
        case_dir, case_id=case["id"], source="test", kind="manual_b", payload={"n": 2}
    )
    entries = observations.read_observations(case_dir)
    assert [e["kind"] for e in entries] == ["manual_a", "manual_b"]
    # The first entry is untouched by the second append; the chain links them.
    assert entries[0] == first
    assert entries[1]["previous_hash"] == first["content_hash"]
    assert entries[0]["previous_hash"] is None
    assert observations.verify_chain(case_dir)
    # The module deliberately exposes no update/delete operation.
    assert not any(name.startswith(("update", "delete", "remove")) for name in dir(observations))
    # Tampering with a stored entry breaks the chain verification.
    ledger_file = case_dir / observations.LEDGER_FILENAME
    lines = ledger_file.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["payload"] = {"n": 999}
    lines[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    ledger_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not observations.verify_chain(case_dir)


def test_case_delete_removes_observation_ledger_and_counterexamples(svc: ResearchMVP, tmp_path: Path):
    case = svc.store.create_case(name="Doomed", goal="")
    case_dir = svc.store.case_dir(case["id"])
    observations.record_observation(
        case_dir, case_id=case["id"], source="test", kind="manual", payload={}
    )
    svc.store.ledger.db.conn.execute(
        """INSERT INTO claims
           (id, canonical_text, claim_type, status, scope_json, metadata_json, created_at, updated_at)
           VALUES (?, ?, 'test', 'supported', '{}', '{}', datetime('now'), datetime('now'))""",
        ("claim_doomed", "doomed claim"),
    )
    svc.store.record_counterexample(
        claim_id="claim_doomed",
        case_id=case["id"],
        graph_id="graph_doomed",
        run_id=None,
        dataset_id="file_doomed",
        found_by="test",
        failure={"epistemic_effect": "challenges"},
    )
    assert (case_dir / observations.LEDGER_FILENAME).exists()
    assert len(svc.store.case_counterexamples(case["id"])) > 0
    assert svc.store.graph_memory_summary("graph_doomed")["counterexample_count"] > 0

    svc.delete_case(case["id"])

    with pytest.raises(KeyError):
        svc.store.get_case(case["id"])
    assert svc.store.case_counterexamples(case["id"]) == []
    assert svc.store.graph_counterexamples("graph_doomed") == []
    assert svc.store.graph_memory_summary("graph_doomed")["counterexample_count"] == 0
    assert not case_dir.exists()
    assert observations.read_observations(case_dir) == []


# ---------------------------------------------------------------------------
# B. Counterexample memory
# ---------------------------------------------------------------------------

def test_killed_claims_write_counterexamples(svc: ResearchMVP, tmp_path: Path):
    case, run = _run_discovery(svc, tmp_path, graph_id="graph_cx")
    cx_rows = svc.store.case_counterexamples(case["id"])
    # The noise column guarantees killed candidates in this fixture.
    assert cx_rows, "run with refuted candidates must write counterexamples"
    claim_ids = {c["claim_id"] for c in svc.store.case_claims(case["id"])}
    for cx in cx_rows:
        assert cx["claim_id"] in claim_ids
        assert cx["case_id"] == case["id"]
        assert cx["graph_id"] == "graph_cx"
        assert cx["run_id"] == run["id"]
        assert cx["dataset_id"]
        assert cx["found_by"]
        assert cx["minimal_known"] is False
        failure = cx["failure"]
        assert failure["epistemic_effect"] in {"refutes", "challenges"}
        assert failure["killer_stages"]
        assert failure["receipts"]
        world = cx["world"]
        assert world["statement"]
        assert world["payload"]


def test_survivor_claims_write_no_counterexample(svc: ResearchMVP, tmp_path: Path):
    case, _run = _run_discovery(svc, tmp_path, graph_id="graph_cx2")
    committed = [
        c for c in svc.store.case_claims(case["id"])
        if c.get("finding_type") == "robust_relation"
    ]
    assert committed, "fixture must commit at least one relation (x→y)"
    cx_claim_ids = {cx["claim_id"] for cx in svc.store.case_counterexamples(case["id"])}
    for claim in committed:
        assert claim["claim_id"] not in cx_claim_ids


def test_counterexample_query_excludes_other_graphs(svc: ResearchMVP, client: TestClient, tmp_path: Path):
    _run_discovery(svc, tmp_path, graph_id="graph_A")
    _run_discovery(svc, tmp_path, graph_id="graph_B")
    a_rows = svc.store.graph_counterexamples("graph_A")
    b_rows = svc.store.graph_counterexamples("graph_B")
    assert a_rows and b_rows
    assert all(cx["graph_id"] == "graph_A" for cx in a_rows)
    a_ids = {cx["id"] for cx in a_rows}
    assert not a_ids & {cx["id"] for cx in b_rows}

    resp = client.get("/graphs/graph_A/counterexamples", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert {cx["id"] for cx in body["counterexamples"]} == a_ids
    # No-graph runs never leak into graph-scoped queries either.
    assert client.get("/graphs/no_such_graph/counterexamples", headers=_auth()).json()["counterexamples"] == []


def test_counterexample_routes_require_auth(client: TestClient):
    assert client.get("/graphs/g/counterexamples").status_code == 401
    assert client.get("/graphs/g/summary").status_code == 401
    assert client.get("/graphs/g/summary", headers=_auth(password="bad")).status_code == 401


# ---------------------------------------------------------------------------
# C. Receipts and epistemic effect
# ---------------------------------------------------------------------------

def test_passed_check_without_commit_is_not_supports():
    """Unknown / no counterexample found is not proof."""
    passed_attack = {"name": "held_out", "killed": False, "metric": 0.2, "detail": {"score": 0.2}}
    receipt = receipts.falsifier_receipt(passed_attack, finding_type="promising_candidate")
    assert receipt["epistemic_effect"] == "none"
    assert receipt["status"] == "passed"

    unknown_attack = {"name": "held_out", "detail": {}}
    receipt = receipts.falsifier_receipt(unknown_attack, finding_type="promising_candidate")
    assert receipt["epistemic_effect"] == "unknown"


def test_receipt_effects_by_finding_type():
    killed = {"name": "held_out", "killed": True, "metric": -0.4, "detail": {"score": -0.4}}
    assert receipts.falsifier_receipt(killed, finding_type="falsified_candidate")["epistemic_effect"] == "refutes"
    assert receipts.falsifier_receipt(killed, finding_type="not_supported_candidate")["epistemic_effect"] == "challenges"
    assert receipts.falsifier_receipt(killed, finding_type="inconclusive_candidate")["epistemic_effect"] == "challenges"

    passed = {"name": "baseline", "killed": False, "metric": 0.9, "detail": {}}
    assert receipts.falsifier_receipt(passed, finding_type="robust_relation")["epistemic_effect"] == "supports"

    skipped = {"name": "ablation", "killed": False, "detail": {"skipped": "not a composite candidate"}}
    assert receipts.falsifier_receipt(skipped, finding_type="robust_relation")["epistemic_effect"] == "none"
    assert receipts.falsifier_receipt(skipped, finding_type="robust_relation")["status"] == "skipped"


def test_receipts_stored_in_finding_detail(svc: ResearchMVP, tmp_path: Path):
    case, _run = _run_discovery(svc, tmp_path, graph_id="graph_rcpt")
    claims = svc.store.case_claims(case["id"])
    with_receipts = [c for c in claims if c["finding_detail"].get("falsifier_receipts")]
    assert with_receipts, "candidate claims must carry falsifier receipts"
    for claim in with_receipts:
        for receipt in claim["finding_detail"]["falsifier_receipts"]:
            assert receipt["stage"]
            assert receipt["epistemic_effect"] in receipts.EPISTEMIC_EFFECTS


# ---------------------------------------------------------------------------
# D. Graph query/export summaries
# ---------------------------------------------------------------------------

def test_graph_claims_include_counterexample_counts_and_summary(
    svc: ResearchMVP, client: TestClient, tmp_path: Path
):
    case, _run = _run_discovery(svc, tmp_path, graph_id="graph_sum")
    resp = client.get("/graphs/graph_sum/claims", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert all("counterexample_count" in c for c in body["claims"])
    killed_counts = [c["counterexample_count"] for c in body["claims"] if c["counterexample_count"] > 0]
    assert killed_counts, "killed claims must show counterexample counts"

    summary = body["summary"]
    assert summary["graph_id"] == "graph_sum"
    assert summary["claim_count"] == len(body["claims"])
    assert summary["counterexample_count"] == len(svc.store.graph_counterexamples("graph_sum"))
    assert summary["observation_count"] >= 4  # import + run start + receipts + completed
    assert case["id"] in summary["observations_by_case"]

    dataset_relations = summary["dataset_relations"]
    assert dataset_relations, "dataset relation summary must be present"
    for _dataset_id, relation in dataset_relations.items():
        assert set(relation) == {"supports", "refutes", "challenges"}
    assert any(r["challenges"] + r["refutes"] > 0 for r in dataset_relations.values())


def test_graph_summary_scoped_per_graph(svc: ResearchMVP, client: TestClient, tmp_path: Path):
    _run_discovery(svc, tmp_path, graph_id="graph_S1")
    _run_discovery(svc, tmp_path, graph_id="graph_S2")
    s1 = client.get("/graphs/graph_S1/summary", headers=_auth()).json()["summary"]
    s2 = client.get("/graphs/graph_S2/summary", headers=_auth()).json()["summary"]
    assert s1["graph_id"] == "graph_S1"
    assert set(s1["observations_by_case"]) != set(s2["observations_by_case"])
    empty = client.get("/graphs/never_used/summary", headers=_auth()).json()["summary"]
    assert empty["claim_count"] == 0
    assert empty["counterexample_count"] == 0
    assert empty["observation_count"] == 0
