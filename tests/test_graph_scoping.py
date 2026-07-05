from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import orbita_mvp.api as api_module
from orbita_mvp.service import ResearchMVP


USER = "graph-user"
PASSWORD = "graph-pass"


def _auth(user: str = USER, password: str = PASSWORD) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


@pytest.fixture()
def svc(tmp_path: Path):
    service = ResearchMVP(tmp_path / "graphs.db", tmp_path / "workspace")
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


def _make_run(svc: ResearchMVP, case_id: str) -> str:
    """Create a real plan + run so case_claims FK constraints hold."""
    plan = svc.store.save_plan(case_id, {"candidates": [], "status": "test"}, compiler="test")
    run = svc.store.create_run(case_id, plan["id"])
    return run["id"]


def _link_claim(svc: ResearchMVP, case_id: str, claim_id: str, run_id: str) -> None:
    # Insert a minimal claims row so graph_claims' JOIN finds it.
    svc.store.ledger.db.conn.execute(
        """INSERT OR IGNORE INTO claims
           (id, canonical_text, claim_type, status, scope_json, metadata_json, created_at, updated_at)
           VALUES (?, ?, 'test', 'supported', '{}', '{}', datetime('now'), datetime('now'))""",
        (claim_id, f"test claim {claim_id}"),
    )
    svc.store.link_claim(
        case_id=case_id,
        run_id=run_id,
        claim_id=claim_id,
        finding_type="committed",
        source_candidate_id=f"cand_{claim_id}",
        finding_detail={},
    )


def test_stamped_claims_returned_by_graph_query(svc: ResearchMVP, client: TestClient):
    case = svc.store.create_case(name="Graph scoping", goal="")
    run_id = _make_run(svc, case["id"])
    _link_claim(svc, case["id"], "claim_g1", run_id)
    svc.store.stamp_run_claims(
        case_id=case["id"], run_id=run_id, graph_id="graph_A",
        origin={"dataset_ids": ["f1"], "engine_version": "test", "plan_hash": "h", "operators": []},
    )

    resp = client.get("/graphs/graph_A/claims", headers=_auth())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["graph_id"] == "graph_A"
    ids = [c["claim_id"] for c in body["claims"]]
    assert "claim_g1" in ids
    row = body["claims"][0]
    assert row["origin"]["operators"] == []
    assert row["origin"]["dataset_ids"] == ["f1"]


def test_other_graph_claims_excluded(svc: ResearchMVP, client: TestClient):
    case = svc.store.create_case(name="Two graphs", goal="")
    run_a = _make_run(svc, case["id"])
    run_b = _make_run(svc, case["id"])
    _link_claim(svc, case["id"], "claim_a", run_a)
    _link_claim(svc, case["id"], "claim_b", run_b)
    svc.store.stamp_run_claims(case_id=case["id"], run_id=run_a, graph_id="graph_A")
    svc.store.stamp_run_claims(case_id=case["id"], run_id=run_b, graph_id="graph_B")

    resp = client.get("/graphs/graph_A/claims", headers=_auth())
    ids = [c["claim_id"] for c in resp.json()["claims"]]
    assert "claim_a" in ids
    assert "claim_b" not in ids


def test_legacy_null_graph_claims_unaffected_and_excluded(svc: ResearchMVP, client: TestClient):
    case = svc.store.create_case(name="Legacy", goal="")
    run_legacy = _make_run(svc, case["id"])
    _link_claim(svc, case["id"], "claim_legacy", run_legacy)
    # No stamp — graph_id stays NULL, exactly like pre-2A rows.

    resp = client.get("/graphs/any_graph/claims", headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["claims"] == []

    # Legacy row is still visible through the case-level query.
    legacy = svc.store.case_claims(case["id"])
    assert any(c["claim_id"] == "claim_legacy" for c in legacy)
    assert all(c.get("graph_id") is None for c in legacy if c["claim_id"] == "claim_legacy")


def test_graph_claims_requires_basic_auth(client: TestClient):
    assert client.get("/graphs/graph_A/claims").status_code == 401
    assert client.get("/graphs/graph_A/claims", headers=_auth(password="wrong")).status_code == 401


def test_counterexamples_table_exists_and_empty(svc: ResearchMVP):
    rows = svc.store.ledger.db.conn.execute("SELECT COUNT(*) AS n FROM counterexamples").fetchone()
    assert rows["n"] == 0
    cols = {r["name"] for r in svc.store.ledger.db.conn.execute("PRAGMA table_info(counterexamples)").fetchall()}
    assert {"id", "claim_id", "case_id", "graph_id", "run_id", "dataset_id",
            "world_json", "measurements_json", "failure_json", "found_by",
            "minimal_known", "created_at"} <= cols


def test_case_claims_columns_added(svc: ResearchMVP):
    cols = {r["name"] for r in svc.store.ledger.db.conn.execute("PRAGMA table_info(case_claims)").fetchall()}
    assert {"graph_id", "origin_json", "epistemic_status"} <= cols
