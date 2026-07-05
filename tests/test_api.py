from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import orbita_mvp.api as api_module
from orbita_mvp import ResearchMVP


def _csv_bytes() -> bytes:
    rows = ["subject_id,group,x,y"]
    for i in range(1, 25):
        group = "A" if i <= 12 else "B"
        x = i / 2
        y = 3 * x + (0.1 if i % 2 else -0.1)
        rows.append(f"s{i:02d},{group},{x:.3f},{y:.3f}")
    return ("\n".join(rows) + "\n").encode()


def test_browser_api_flow(tmp_path: Path, monkeypatch):
    old_service = api_module.service
    replacement = ResearchMVP(tmp_path / "api.db", tmp_path / "workspace")
    api_module.service = replacement
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setattr(api_module, "_DEMO_USER", "")
    monkeypatch.setattr(api_module, "_DEMO_PASS", "")
    try:
        with TestClient(api_module.app) as client:
            assert client.get("/").status_code == 200
            assert client.get("/health").json()["status"] == "ok"

            case = client.post("/cases", json={"name": "API case", "goal": ""}).json()
            upload = client.post(
                f"/cases/{case['id']}/files",
                files={"file": ("data.csv", _csv_bytes(), "text/csv")},
            )
            assert upload.status_code == 200
            assert upload.json()["artifact_kind"] == "table"

            plan_response = client.post(f"/cases/{case['id']}/compile", json={"max_candidates": 20})
            assert plan_response.status_code == 200
            plan = plan_response.json()
            revised_plan = dict(plan["plan"])
            revised_plan["thresholds"] = {**revised_plan["thresholds"], "commit_at": 0.3}
            revised = client.post(
                f"/plans/{plan['id']}/revise",
                json={"plan": revised_plan, "compiler": "api-test"},
            ).json()
            assert revised["version"] == plan["version"] + 1

            assert client.post(
                f"/plans/{revised['id']}/approve", json={"reviewer": "tester"}
            ).status_code == 200
            run = client.post(
                f"/cases/{case['id']}/run", json={"plan_id": revised["id"]}
            )
            assert run.status_code == 200
            result = run.json()["result"]
            assert result["candidate_count"] > 0

            claims = client.get(f"/cases/{case['id']}/claims").json()["claims"]
            assert claims
            claim_id = claims[0]["claim_id"]
            assert client.get(f"/claims/{claim_id}/history").status_code == 200
            assert client.get(f"/claims/{claim_id}/impact").status_code == 200
            assert client.get(f"/cases/{case['id']}/report").status_code == 200
    finally:
        api_module.service = old_service
        replacement.close()
