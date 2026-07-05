"""Security tests for /admin/test/tamper-artifact and /admin/test/restore-artifact.

Invariants:
  1. Routes work in test-enabled mode with the correct token
  2. Routes return 404 when ORBITA_ENABLE_TEST_ENDPOINTS is not set (production-safe)
  3. Routes return 404 when ORBITA_ENABLE_TEST_ENDPOINTS is explicitly false
  4. ORBITA_TEST_TOKEN alone (without ORBITA_ENABLE_TEST_ENDPOINTS=true) cannot activate routes
  5. No credential or token value appears in any successful response
  6. Routes return 403 (not 404) when enabled but wrong token provided
"""
from __future__ import annotations

import io
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

import orbita_mvp.api as api_module
from orbita_mvp.service import ResearchMVP
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixture: isolated service + client with patched module-level state
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_client(tmp_path, monkeypatch):
    """Return a TestClient backed by an isolated tmp_path service.

    Yields (client, run_id) after creating a minimal run through the API.
    """
    svc = ResearchMVP(tmp_path / "sec.db", tmp_path / "ws")
    monkeypatch.setattr(api_module, "service", svc)
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setattr(api_module, "_DEMO_USER", "")
    monkeypatch.setattr(api_module, "_DEMO_PASS", "")

    client = TestClient(api_module.app, raise_server_exceptions=False)

    # Build and upload a tiny dataset
    rng = np.random.default_rng(77)
    n = 120
    x = rng.uniform(1, 5, n)
    y = 3.0 * x + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"row_id": range(n), "x": x, "y": y})
    csv_bytes = df.to_csv(index=False).encode()

    r_case = client.post("/cases", json={"name": "sec-test", "goal": ""})
    assert r_case.status_code == 200, r_case.text
    case_id = r_case.json()["id"]

    r_file = client.post(
        f"/cases/{case_id}/files",
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert r_file.status_code == 200, r_file.text

    r_compile = client.post(
        f"/cases/{case_id}/compile",
        json={"evaluation_metric": "r2", "max_candidates": 20},
    )
    assert r_compile.status_code == 200, r_compile.text

    r_run = client.post(f"/cases/{case_id}/run", json={"auto_approve": True})
    assert r_run.status_code == 200, r_run.text
    run_id = r_run.json()["result"]["run_id"]

    yield client, run_id


# ---------------------------------------------------------------------------
# Test 1: Routes work with correct token when test endpoints are enabled
# ---------------------------------------------------------------------------

def test_tamper_restore_work_when_enabled(isolated_client, monkeypatch):
    """With ORBITA_ENABLE_TEST_ENDPOINTS=true and correct token, tamper+restore cycle succeeds."""
    client, run_id = isolated_client
    token = "test-tok-abc123"
    monkeypatch.setattr(api_module, "_TEST_ENDPOINTS_ENABLED", True)
    monkeypatch.setenv("ORBITA_TEST_TOKEN", token)

    r_tamper = client.post(
        "/admin/test/tamper-artifact",
        params={"run_id": run_id, "target_column": "y", "test_token": token},
    )
    assert r_tamper.status_code == 200, f"expected 200; got {r_tamper.status_code}: {r_tamper.text[:300]}"
    body = r_tamper.json()
    assert body["status"] == "corrupted"
    assert "corrupted_field" in body

    r_restore = client.post(
        "/admin/test/restore-artifact",
        params={"run_id": run_id, "target_column": "y", "test_token": token},
    )
    assert r_restore.status_code == 200, f"restore failed: {r_restore.status_code}: {r_restore.text[:200]}"
    assert r_restore.json()["status"] == "restored"


# ---------------------------------------------------------------------------
# Test 2: Routes return 404 when ORBITA_ENABLE_TEST_ENDPOINTS is not set
# ---------------------------------------------------------------------------

def test_tamper_returns_404_when_disabled_by_absence(isolated_client, monkeypatch):
    """When _TEST_ENDPOINTS_ENABLED is False (env var absent), both routes return 404."""
    client, run_id = isolated_client
    token = "some-token"
    monkeypatch.setattr(api_module, "_TEST_ENDPOINTS_ENABLED", False)
    monkeypatch.setenv("ORBITA_TEST_TOKEN", token)

    r_tamper = client.post(
        "/admin/test/tamper-artifact",
        params={"run_id": run_id, "target_column": "y", "test_token": token},
    )
    assert r_tamper.status_code == 404, (
        f"disabled tamper endpoint must return 404; got {r_tamper.status_code}"
    )

    r_restore = client.post(
        "/admin/test/restore-artifact",
        params={"run_id": run_id, "target_column": "y", "test_token": token},
    )
    assert r_restore.status_code == 404, (
        f"disabled restore endpoint must return 404; got {r_restore.status_code}"
    )


# ---------------------------------------------------------------------------
# Test 3: Routes return 404 when ORBITA_ENABLE_TEST_ENDPOINTS is explicitly false
# ---------------------------------------------------------------------------

def test_tamper_returns_404_when_explicitly_disabled(isolated_client, monkeypatch):
    """ORBITA_ENABLE_TEST_ENDPOINTS=false (any non-true value) must produce 404."""
    client, run_id = isolated_client
    monkeypatch.setattr(api_module, "_TEST_ENDPOINTS_ENABLED", False)
    monkeypatch.setenv("ORBITA_TEST_TOKEN", "tok")

    assert client.post(
        "/admin/test/tamper-artifact",
        params={"run_id": run_id, "target_column": "y", "test_token": "tok"},
    ).status_code == 404

    assert client.post(
        "/admin/test/restore-artifact",
        params={"run_id": run_id, "target_column": "y", "test_token": "tok"},
    ).status_code == 404


# ---------------------------------------------------------------------------
# Test 4: ORBITA_TEST_TOKEN alone cannot activate routes
# ---------------------------------------------------------------------------

def test_token_alone_cannot_activate_disabled_routes(isolated_client, monkeypatch):
    """Setting ORBITA_TEST_TOKEN without ORBITA_ENABLE_TEST_ENDPOINTS=true must still return 404.

    This is the production-safety invariant: even a valid token cannot enable
    destructive test endpoints when the enable flag is absent.
    """
    client, run_id = isolated_client
    secret_token = "super-secret-production-token"
    # _TEST_ENDPOINTS_ENABLED=False but ORBITA_TEST_TOKEN is set with the real value
    monkeypatch.setattr(api_module, "_TEST_ENDPOINTS_ENABLED", False)
    monkeypatch.setenv("ORBITA_TEST_TOKEN", secret_token)

    r = client.post(
        "/admin/test/tamper-artifact",
        params={"run_id": run_id, "target_column": "y", "test_token": secret_token},
    )
    assert r.status_code == 404, (
        "ORBITA_TEST_TOKEN alone must not activate disabled test endpoints. "
        f"Got {r.status_code}: {r.text[:200]}"
    )


# ---------------------------------------------------------------------------
# Test 5: No credential or token appears in any response body
# ---------------------------------------------------------------------------

def test_no_token_in_response_bodies(isolated_client, monkeypatch):
    """Successful tamper response must not echo the test token or any secret."""
    client, run_id = isolated_client
    token = "my-secret-staging-token-XYZ789"
    monkeypatch.setattr(api_module, "_TEST_ENDPOINTS_ENABLED", True)
    monkeypatch.setenv("ORBITA_TEST_TOKEN", token)

    r_tamper = client.post(
        "/admin/test/tamper-artifact",
        params={"run_id": run_id, "target_column": "y", "test_token": token},
    )
    assert r_tamper.status_code == 200
    assert token not in r_tamper.text, "Token must not appear in tamper response body"

    r_restore = client.post(
        "/admin/test/restore-artifact",
        params={"run_id": run_id, "target_column": "y", "test_token": token},
    )
    assert r_restore.status_code == 200
    assert token not in r_restore.text, "Token must not appear in restore response body"


# ---------------------------------------------------------------------------
# Test 6: Enabled but wrong token → 403 (not 404)
# ---------------------------------------------------------------------------

def test_wrong_token_returns_403_not_404(isolated_client, monkeypatch):
    """When test endpoints are enabled, a wrong token must return 403, not 404.
    This distinguishes 'route exists but denied' from 'route does not exist'.
    """
    client, run_id = isolated_client
    monkeypatch.setattr(api_module, "_TEST_ENDPOINTS_ENABLED", True)
    monkeypatch.setenv("ORBITA_TEST_TOKEN", "correct-token")

    r = client.post(
        "/admin/test/tamper-artifact",
        params={"run_id": run_id, "target_column": "y", "test_token": "wrong-token"},
    )
    assert r.status_code == 403, (
        f"wrong token with enabled endpoints should return 403; got {r.status_code}"
    )
