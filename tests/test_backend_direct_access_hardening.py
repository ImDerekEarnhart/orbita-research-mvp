from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import orbita_mvp.api as api_module
import orbita_mvp.upload_safety as upload_safety
from orbita_mvp.service import ResearchMVP


USER = "backend-user"
PASSWORD = "backend-pass"


def _auth(user: str = USER, password: str = PASSWORD) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _csv_bytes() -> bytes:
    return b"subject_id,x,y\ns01,1,2\ns02,2,4\n"


@pytest.fixture()
def isolated_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    svc = ResearchMVP(tmp_path / "hardening.db", tmp_path / "workspace")
    monkeypatch.setattr(api_module, "service", svc)
    for key in (
        "APP_ENV",
        "ORBITA_APP_ENV",
        "ORBITA_ENV",
        "RAILWAY_ENVIRONMENT",
        "RAILWAY_ENVIRONMENT_NAME",
    ):
        monkeypatch.delenv(key, raising=False)
    try:
        yield svc
    finally:
        svc.close()


def _client(monkeypatch: pytest.MonkeyPatch, user: str = USER, password: str = PASSWORD) -> TestClient:
    monkeypatch.setattr(api_module, "_DEMO_USER", user)
    monkeypatch.setattr(api_module, "_DEMO_PASS", password)
    return TestClient(api_module.app, raise_server_exceptions=False)


def _create_case(client: TestClient) -> str:
    response = client.post("/cases", json={"name": "upload hardening", "goal": ""}, headers=_auth())
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _upload(
    client: TestClient,
    case_id: str,
    filename: str,
    content: bytes | str = _csv_bytes(),
    content_type: str = "text/csv",
):
    if isinstance(content, str):
        content = content.encode()
    return client.post(
        f"/cases/{case_id}/files",
        headers=_auth(),
        files={"file": (filename, content, content_type)},
    )


def _upload_raw_filename(
    client: TestClient,
    case_id: str,
    filename: str,
    content: bytes | str = _csv_bytes(),
    content_type: str = "text/csv",
):
    if isinstance(content, str):
        content = content.encode()
    boundary = "----orbita-hardening-test"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    footer = f"\r\n--{boundary}--\r\n".encode()
    headers = {
        **_auth(),
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    return client.post(f"/cases/{case_id}/files", headers=headers, content=header + content + footer)


def test_basic_auth_fails_closed_for_staging_like_missing_credentials(
    isolated_service, monkeypatch: pytest.MonkeyPatch
):
    client = _client(monkeypatch, user="", password="")

    assert client.get("/health").status_code == 200
    assert client.get("/cases").status_code == 401
    monkeypatch.setenv("APP_ENV", "staging")
    assert client.get("/cases").status_code == 401
    denied = client.post(
        "/cases/any-case/files",
        files={"file": ("data.csv", _csv_bytes(), "text/csv")},
    )
    assert denied.status_code == 401


def test_basic_auth_wrong_and_correct_credentials(isolated_service, monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)

    assert client.get("/health").status_code == 200
    assert client.get("/cases").status_code == 401
    assert client.get("/cases", headers=_auth(password="wrong")).status_code == 401
    assert client.get("/cases", headers=_auth()).status_code == 200


def test_valid_tiny_csv_upload_accepted(isolated_service, monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    case_id = _create_case(client)

    response = _upload(client, case_id, "measurements.csv")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["artifact_kind"] == "table"
    assert body["original_name"] == "measurements.csv"


def test_upload_rejects_unsafe_filenames(isolated_service, monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    case_id = _create_case(client)

    names = [
        "../evil.csv",
        "..\\evil.csv",
        "/tmp/evil.csv",
        "C:\\evil.csv",
        ".hidden.csv",
        "evil.exe",
        "evil.csv.exe",
        "evil.exe.csv",
        "evil.zip",
        "evil.sh",
        "evil.js",
    ]
    for name in names:
        response = _upload_raw_filename(client, case_id, name)
        assert response.status_code == 400, f"{name} returned {response.status_code}: {response.text}"
        assert response.json()["detail"] == "Invalid upload"


def test_upload_rejects_unsafe_mime_and_disguised_content(
    isolated_service, monkeypatch: pytest.MonkeyPatch
):
    client = _client(monkeypatch)
    case_id = _create_case(client)

    cases = [
        ("mime.csv", _csv_bytes(), "application/x-msdownload"),
        ("pe.csv", b"MZ\x00\x01not a csv", "text/csv"),
        ("zip.csv", b"PK\x03\x04\x14\x00not a csv", "text/csv"),
        ("script.csv", b"#!/bin/sh\necho nope\n", "text/csv"),
        ("html.csv", b"<script>alert(1)</script>\n", "text/csv"),
        ("binary.csv", b"a,b\n1,\x00\n", "text/csv"),
    ]
    for filename, content, content_type in cases:
        response = _upload(client, case_id, filename, content, content_type)
        assert response.status_code == 400, f"{filename} returned {response.status_code}: {response.text}"
        assert response.json()["detail"] == "Invalid upload"


def test_upload_rejects_oversized_file(isolated_service, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(api_module, "MAX_CSV_UPLOAD_BYTES", 32)
    monkeypatch.setattr(upload_safety, "MAX_CSV_UPLOAD_BYTES", 32)
    client = _client(monkeypatch)
    case_id = _create_case(client)

    response = _upload(client, case_id, "big.csv", b"a,b\n" + (b"1,2\n" * 20))
    assert response.status_code == 413
    assert response.json()["detail"] == "File too large"
