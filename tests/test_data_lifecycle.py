from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import orbita_mvp.api as api_module
from orbita_mvp.service import ResearchMVP


USER = "delete-user"
PASSWORD = "delete-pass"


def _auth(user: str = USER, password: str = PASSWORD) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _csv_bytes() -> bytes:
    return b"subject_id,x,y\ns01,1,2\ns02,2,4\n"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    svc = ResearchMVP(tmp_path / "lifecycle.db", tmp_path / "workspace")
    monkeypatch.setattr(api_module, "service", svc)
    monkeypatch.setattr(api_module, "_DEMO_USER", USER)
    monkeypatch.setattr(api_module, "_DEMO_PASS", PASSWORD)
    try:
        yield TestClient(api_module.app, raise_server_exceptions=False)
    finally:
        svc.close()


def _create_case(client: TestClient, name: str = "Lifecycle case") -> str:
    response = client.post("/cases", json={"name": name, "goal": ""}, headers=_auth())
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _upload(client: TestClient, case_id: str, filename: str = "data.csv") -> dict:
    response = client.post(
        f"/cases/{case_id}/files",
        headers=_auth(),
        files={"file": (filename, _csv_bytes(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_delete_case_requires_basic_auth_and_health_remains_public(client: TestClient):
    case_id = _create_case(client)

    assert client.get("/health").status_code == 200
    assert client.delete(f"/cases/{case_id}").status_code == 401
    assert client.delete(f"/cases/{case_id}", headers=_auth(password="wrong")).status_code == 401
    assert client.get(f"/cases/{case_id}", headers=_auth()).status_code == 200


def test_delete_missing_case_returns_404(client: TestClient):
    response = client.delete("/cases/case_missing", headers=_auth())
    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown resource"


def test_delete_existing_case_removes_backend_case_and_artifacts(client: TestClient):
    case_id = _create_case(client)
    file_record = _upload(client, case_id)
    stored_path = Path(file_record["stored_path"])
    extracted_path = Path(file_record["extracted_path"])

    assert stored_path.exists()
    assert extracted_path.exists()

    response = client.delete(f"/cases/{case_id}", headers=_auth())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["deleted"] is True
    assert body["case_id"] == case_id
    assert body["artifacts_removed"] >= 2

    assert client.get(f"/cases/{case_id}", headers=_auth()).status_code == 404
    assert not stored_path.exists()
    assert not extracted_path.exists()


def test_delete_case_does_not_remove_unrelated_case_artifacts(client: TestClient):
    case_a = _create_case(client, "Delete me")
    case_b = _create_case(client, "Keep me")
    _upload(client, case_a, "a.csv")
    b_file = _upload(client, case_b, "b.csv")
    b_stored = Path(b_file["stored_path"])

    response = client.delete(f"/cases/{case_a}", headers=_auth())
    assert response.status_code == 200, response.text

    assert client.get(f"/cases/{case_a}", headers=_auth()).status_code == 404
    assert client.get(f"/cases/{case_b}", headers=_auth()).status_code == 200
    assert b_stored.exists()
