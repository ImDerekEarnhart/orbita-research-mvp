from __future__ import annotations

import html
import json
import mimetypes
import re
import secrets
import threading
import uuid
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .analysis import DatasetAnalysisSpec
from .agent_os import AutonomyMode, ComputerAgentRuntime
from .discovery import DiscoverySpec, GovernedDiscoveryRuntime
from .execution import CliOCIEngine, ContainerExecutionRuntime, ContainerExecutionSpec
from .evaluation import ComparativeEvaluationRuntime, default_adversarial_suite
from .ledger import EpistemicLedger
from .models import (
    ActorRole,
    EvidenceKind,
    LiteralDatatype,
    ObjectKind,
    ReviewDecision,
    Stance,
    TypedLiteral,
)
from .proposals import ModelIdentity
from .support import SupportEngine

UI_VERSION = "1.5.0"
_MAX_JSON_BYTES = 8 * 1024 * 1024
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_UNSAFE_SVG = re.compile(
    r"<(?:script|foreignObject)\b|\bon[a-z]+\s*=|\bjavascript\s*:", re.IGNORECASE
)


@dataclass(slots=True)
class UIConfig:
    db_path: str | Path
    workspace: str | Path | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    allow_remote: bool = False
    open_browser: bool = True
    max_upload_bytes: int = 5 * 1024 * 1024
    access_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    csrf_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path).expanduser().resolve()
        self.workspace = (
            Path(self.workspace).expanduser().resolve()
            if self.workspace is not None
            else self.db_path.parent / "orbita_ui_workspace"
        )
        if not (0 <= int(self.port) <= 65535):
            raise ValueError("port must be between 0 and 65535")
        self.port = int(self.port)
        if self.host not in {"127.0.0.1", "localhost", "::1"} and not self.allow_remote:
            raise ValueError(
                "Refusing a non-loopback bind without allow_remote=True. "
                "Remote mode should be used only on a trusted private network or tunnel."
            )
        if self.max_upload_bytes < 1024:
            raise ValueError("max_upload_bytes is unreasonably small")
        Path(self.workspace).mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class UIResponse:
    status: int
    body: bytes
    content_type: str = "application/json; charset=utf-8"
    headers: dict[str, str] = field(default_factory=dict)


class APIError(Exception):
    def __init__(self, status: int, message: str, *, details: Any = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.details = details


class OrbitaUIApp:
    """Local, dependency-light web interface over the epistemic runtime.

    The UI never grants arbitrary filesystem access or arbitrary code execution.
    CSV uploads are copied into a confined workspace and may only be processed by
    the built-in, receipt-producing analysis vocabulary.
    """

    def __init__(self, config: UIConfig):
        self.config = config
        self.assets = files("orbita").joinpath("ui_assets")

    def handle(
        self,
        method: str,
        raw_path: str,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> UIResponse:
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        parsed = urlparse(raw_path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/" and method == "GET":
                return self._index(query, headers)
            if path.startswith("/assets/") and method == "GET":
                self._require_session(headers)
                return self._asset(path.removeprefix("/assets/"))
            if path == "/api/health" and method == "GET":
                self._require_session(headers)
                return self._ok(
                    {
                        "ok": True,
                        "ui_version": UI_VERSION,
                        "database": self.config.db_path.name,
                        "remote_mode": bool(self.config.allow_remote),
                    }
                )
            if path.startswith("/api/"):
                self._require_session(headers)
                if method in {"POST", "PUT", "PATCH", "DELETE"}:
                    self._require_csrf(headers)
                    self._require_same_origin(headers)
                payload = self._decode_json(body) if method != "GET" else {}
                return self._api(method, path, query, payload)
            raise APIError(HTTPStatus.NOT_FOUND, "Route not found")
        except APIError as exc:
            return self._error(exc.status, exc.message, exc.details)
        except KeyError as exc:
            return self._error(HTTPStatus.NOT_FOUND, str(exc).strip("'"))
        except (ValueError, TypeError, PermissionError, FileNotFoundError, RuntimeError) as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # do not leak tracebacks into the browser
            return self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "The request failed inside the local runtime",
                {"error_type": type(exc).__name__, "message": str(exc)},
            )

    # ------------------------------------------------------------------
    # Authentication and static shell
    # ------------------------------------------------------------------
    def _index(self, query: dict[str, list[str]], headers: dict[str, str]) -> UIResponse:
        supplied = query.get("token", [None])[0]
        session_ok = self._session_token(headers) == self.config.access_token
        if supplied != self.config.access_token and not session_ok:
            return UIResponse(
                HTTPStatus.UNAUTHORIZED,
                b"Orbita UI access token required.",
                "text/plain; charset=utf-8",
            )
        template = self.assets.joinpath("index.html").read_text(encoding="utf-8")
        rendered = template.replace("__ORBITA_CSRF_TOKEN__", html.escape(self.config.csrf_token))
        cookie = (
            f"orbita_session={self.config.access_token}; Path=/; HttpOnly; "
            "SameSite=Strict; Max-Age=43200"
        )
        return UIResponse(
            HTTPStatus.OK,
            rendered.encode("utf-8"),
            "text/html; charset=utf-8",
            {"Set-Cookie": cookie},
        )

    def _asset(self, name: str) -> UIResponse:
        if name not in {"app.js", "style.css"}:
            raise APIError(HTTPStatus.NOT_FOUND, "Asset not found")
        asset = self.assets.joinpath(name)
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return UIResponse(HTTPStatus.OK, asset.read_bytes(), f"{content_type}; charset=utf-8")

    def _session_token(self, headers: dict[str, str]) -> str | None:
        raw = headers.get("cookie", "")
        cookie = SimpleCookie()
        try:
            cookie.load(raw)
        except Exception:
            return None
        morsel = cookie.get("orbita_session")
        return morsel.value if morsel else None

    def _require_session(self, headers: dict[str, str]) -> None:
        if not secrets.compare_digest(
            self._session_token(headers) or "", self.config.access_token
        ):
            raise APIError(HTTPStatus.UNAUTHORIZED, "Valid Orbita session required")

    def _require_csrf(self, headers: dict[str, str]) -> None:
        if not secrets.compare_digest(
            headers.get("x-orbita-csrf", ""), self.config.csrf_token
        ):
            raise APIError(HTTPStatus.FORBIDDEN, "Invalid CSRF token")

    @staticmethod
    def _require_same_origin(headers: dict[str, str]) -> None:
        origin = headers.get("origin")
        host = headers.get("host")
        if origin and host:
            parsed = urlparse(origin)
            if parsed.netloc != host:
                raise APIError(HTTPStatus.FORBIDDEN, "Cross-origin mutation refused")

    # ------------------------------------------------------------------
    # API dispatcher
    # ------------------------------------------------------------------
    def _api(
        self,
        method: str,
        path: str,
        query: dict[str, list[str]],
        payload: dict[str, Any],
    ) -> UIResponse:
        parts = [part for part in path.split("/") if part]
        if path == "/api/dashboard" and method == "GET":
            return self._ok(self._dashboard())
        if path == "/api/claims" and method == "GET":
            return self._ok(self._claims(query))
        if path == "/api/claims" and method == "POST":
            return self._ok(self._create_claim(payload), status=HTTPStatus.CREATED)
        if len(parts) == 3 and parts[:2] == ["api", "claims"] and method == "GET":
            return self._ok(self._claim_detail(parts[2]))
        if (
            len(parts) == 4
            and parts[:2] == ["api", "claims"]
            and parts[3] == "evidence"
            and method == "POST"
        ):
            return self._ok(self._add_claim_evidence(parts[2], payload), status=HTTPStatus.CREATED)
        if (
            len(parts) == 4
            and parts[:2] == ["api", "evidence"]
            and parts[3] == "revoke"
            and method == "POST"
        ):
            return self._ok(self._revoke_evidence(parts[2], payload))

        if path == "/api/analyses" and method == "GET":
            return self._ok(self._analyses())
        if path == "/api/analyses" and method == "POST":
            return self._ok(self._run_analysis(payload), status=HTTPStatus.CREATED)
        if len(parts) == 3 and parts[:2] == ["api", "analyses"] and method == "GET":
            return self._ok(self._analysis_detail(parts[2]))
        if (
            len(parts) == 4
            and parts[:2] == ["api", "analyses"]
            and parts[3] == "reproduce"
            and method == "POST"
        ):
            return self._ok(self._reproduce_analysis(parts[2], payload), status=HTTPStatus.CREATED)

        if path == "/api/discoveries" and method == "GET":
            return self._ok(self._discoveries())
        if path == "/api/discoveries" and method == "POST":
            return self._ok(self._create_discovery(payload), status=HTTPStatus.CREATED)
        if len(parts) == 3 and parts[:2] == ["api", "discoveries"] and method == "GET":
            return self._ok(self._discovery_detail(parts[2]))
        if (len(parts) == 4 and parts[:2] == ["api", "discoveries"]
                and parts[3] == "approve" and method == "POST"):
            return self._ok(self._approve_discovery(parts[2], payload))
        if (len(parts) == 4 and parts[:2] == ["api", "discoveries"]
                and parts[3] == "advance" and method == "POST"):
            return self._ok(self._advance_discovery(parts[2], payload), status=HTTPStatus.CREATED)
        if (len(parts) == 4 and parts[:2] == ["api", "discoveries"]
                and parts[3] == "report" and method == "POST"):
            return self._ok(self._compile_discovery_report(parts[2]))


        if path == "/api/evaluations" and method == "GET":
            return self._ok(self._evaluations())
        if path == "/api/evaluations/default" and method == "POST":
            return self._ok(self._create_default_evaluation(), status=HTTPStatus.CREATED)
        if len(parts) == 3 and parts[:2] == ["api", "evaluations"] and method == "GET":
            return self._ok(self._evaluation_detail(parts[2]))
        if (len(parts) == 4 and parts[:2] == ["api", "evaluations"]
                and parts[3] == "fixture" and method == "POST"):
            return self._ok(self._create_evaluation_fixture(parts[2], payload), status=HTTPStatus.CREATED)
        if (len(parts) == 4 and parts[:2] == ["api", "evaluations"]
                and parts[3] == "report" and method == "POST"):
            return self._ok(self._compile_evaluation_report(parts[2]))
        if len(parts) == 3 and parts[:2] == ["api", "evaluation-runs"] and method == "GET":
            return self._ok(self._evaluation_run_detail(parts[2]))

        if path == "/api/executions" and method == "GET":
            return self._ok(self._executions())
        if path == "/api/executions" and method == "POST":
            return self._ok(self._submit_execution(payload), status=HTTPStatus.CREATED)
        if len(parts) == 3 and parts[:2] == ["api", "executions"] and method == "GET":
            return self._ok(self._execution_detail(parts[2]))
        if (
            len(parts) == 4
            and parts[:2] == ["api", "executions"]
            and parts[3] == "approve"
            and method == "POST"
        ):
            return self._ok(self._approve_execution(parts[2], payload))
        if (
            len(parts) == 4
            and parts[:2] == ["api", "executions"]
            and parts[3] == "reject"
            and method == "POST"
        ):
            return self._ok(self._reject_execution(parts[2], payload))
        if (
            len(parts) == 4
            and parts[:2] == ["api", "executions"]
            and parts[3] == "run"
            and method == "POST"
        ):
            return self._ok(self._run_execution(parts[2], payload), status=HTTPStatus.CREATED)
        if (
            len(parts) == 4
            and parts[:2] == ["api", "executions"]
            and parts[3] == "reproduce"
            and method == "POST"
        ):
            return self._ok(self._reproduce_execution(parts[2]), status=HTTPStatus.CREATED)

        if path == "/api/proposals" and method == "GET":
            return self._ok(self._proposals(query))
        if path == "/api/proposals/ingest" and method == "POST":
            return self._ok(self._ingest_proposal(payload), status=HTTPStatus.CREATED)
        if len(parts) == 3 and parts[:2] == ["api", "proposals"] and method == "GET":
            return self._ok(self._proposal_detail(parts[2]))
        if (
            len(parts) == 4
            and parts[:2] == ["api", "proposal-items"]
            and parts[3] == "review"
            and method == "POST"
        ):
            return self._ok(self._review_proposal(parts[2], payload))

        if path == "/api/graphs" and method == "GET":
            return self._ok(self._graphs())
        if path == "/api/graphs/snapshots" and method == "POST":
            return self._ok(self._capture_graph(payload), status=HTTPStatus.CREATED)
        if (
            len(parts) == 4
            and parts[:3] == ["api", "graphs", "snapshots"]
            and method == "GET"
        ):
            return self._ok(self._graph_snapshot(parts[3]))
        if path == "/api/graphs/diffs" and method == "POST":
            return self._ok(self._compare_graphs(payload), status=HTTPStatus.CREATED)
        if (
            len(parts) == 4
            and parts[:3] == ["api", "graphs", "diffs"]
            and method == "GET"
        ):
            return self._ok(self._graph_diff(parts[3]))

        if path == "/api/agent/skills" and method == "GET":
            return self._ok(self._agent_skills())
        if path == "/api/agent/state" and method == "GET":
            return self._ok(self._agent_state(query.get("goal_id", [None])[0]))
        if path == "/api/agent/compile" and method == "POST":
            return self._ok(self._agent_compile(payload))
        if path == "/api/agent/goals" and method == "GET":
            return self._ok(self._agent_goals())
        if path == "/api/agent/goals" and method == "POST":
            return self._ok(self._agent_create_goal(payload), status=HTTPStatus.CREATED)
        if len(parts) == 4 and parts[:3] == ["api", "agent", "goals"] and method == "GET":
            return self._ok(self._agent_goal(parts[3]))
        if (len(parts) == 5 and parts[:3] == ["api", "agent", "goals"]
                and parts[4] == "plan" and method == "POST"):
            return self._ok(self._agent_plan_goal(parts[3]), status=HTTPStatus.CREATED)
        if len(parts) == 4 and parts[:3] == ["api", "agent", "plans"] and method == "GET":
            return self._ok(self._agent_plan(parts[3]))
        if (len(parts) == 5 and parts[:3] == ["api", "agent", "plans"]
                and parts[4] == "run" and method == "POST"):
            return self._ok(self._agent_run_plan(parts[3]))
        if (len(parts) == 5 and parts[:3] == ["api", "agent", "approvals"]
                and parts[4] == "approve" and method == "POST"):
            return self._ok(self._agent_approve(parts[3], payload))

        if path == "/api/adaptive/status" and method == "GET":
            return self._ok(self._adaptive_status())
        if path == "/api/desktop/observations" and method == "GET":
            limit = int(query.get("limit", ["100"])[0])
            return self._ok(self._desktop_observations(limit))
        if len(parts) == 4 and parts[:3] == ["api", "desktop", "observations"] and method == "GET":
            return self._ok(self._desktop_observation(parts[3]))
        if path == "/api/adaptive/workflows" and method == "GET":
            return self._ok(self._adaptive_workflows(query.get("status", [None])[0]))
        if len(parts) == 4 and parts[:3] == ["api", "adaptive", "workflows"] and method == "GET":
            return self._ok(self._adaptive_workflow(parts[3]))

        if path == "/api/integrations/status" and method == "GET":
            return self._ok(self._integration_status())
        if path == "/api/integrations/drafts" and method == "GET":
            return self._ok(self._integration_drafts())
        if path == "/api/integrations/drafts" and method == "POST":
            return self._ok(self._create_integration_draft(payload), status=HTTPStatus.CREATED)
        if len(parts) == 4 and parts[:3] == ["api", "integrations", "drafts"] and method == "GET":
            return self._ok(self._integration_draft(parts[3]))
        if (len(parts) == 5 and parts[:3] == ["api", "integrations", "drafts"]
                and parts[4] == "request-approval" and method == "POST"):
            return self._ok(self._request_integration_approval(parts[3]))
        if (len(parts) == 5 and parts[:3] == ["api", "integrations", "approvals"]
                and parts[4] == "decide" and method == "POST"):
            return self._ok(self._decide_integration_approval(parts[3], payload))
        if path == "/api/schedules" and method == "GET":
            return self._ok(self._schedules())
        if path == "/api/schedules" and method == "POST":
            return self._ok(self._create_schedule(payload), status=HTTPStatus.CREATED)
        if len(parts) == 3 and parts[:2] == ["api", "schedules"] and method == "GET":
            return self._ok(self._schedule(parts[2]))
        if (len(parts) == 4 and parts[:2] == ["api", "schedules"]
                and parts[3] == "resume" and method == "POST"):
            return self._ok(self._resume_schedule(parts[2], payload))
        if (len(parts) == 4 and parts[:2] == ["api", "schedules"]
                and parts[3] in {"pause", "activate", "cancel"} and method == "POST"):
            return self._ok(self._set_schedule_state(parts[2], parts[3]))
        if path == "/api/scheduler/tick" and method == "POST":
            return self._ok(self._scheduler_tick(payload))

        if path == "/api/language/ask" and method == "POST":
            return self._ok(self._language_ask(payload), status=HTTPStatus.CREATED)
        if path == "/api/language/interpret" and method == "POST":
            return self._ok(self._language_interpret(payload))
        if len(parts) == 3 and parts[:2] == ["api", "language"] and method == "GET":
            return self._ok(self._language_response(parts[2]))
        if (
            len(parts) == 4
            and parts[:2] == ["api", "language"]
            and parts[3] == "verify"
            and method == "GET"
        ):
            return self._ok({"response_id": parts[2], "integrity_valid": self._language_verify(parts[2])})

        if path == "/api/events" and method == "GET":
            return self._ok(self._events(query))
        raise APIError(HTTPStatus.NOT_FOUND, "API route not found")

    # ------------------------------------------------------------------
    # Dashboard and claims
    # ------------------------------------------------------------------
    def _dashboard(self) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            claims = ledger.list_claims()
            reports = SupportEngine(ledger).evaluate_many([c["id"] for c in claims])
            claim_status: dict[str, int] = {}
            support_state: dict[str, int] = {}
            for claim in claims:
                claim_status[claim["status"]] = claim_status.get(claim["status"], 0) + 1
                state = reports[claim["id"]].state.value
                support_state[state] = support_state.get(state, 0) + 1
            receipts = ledger.list_analysis_receipts()
            batches = ledger.list_proposal_batches()
            snapshots = ledger.graphs.list_snapshots()
            diffs = ledger.graphs.list_diffs()
            execution_runtime = ContainerExecutionRuntime(
                ledger, Path(self.config.workspace) / "executions"
            )
            executions = execution_runtime.list()
            execution_status = execution_runtime.runtime_status()
            discoveries = GovernedDiscoveryRuntime(ledger, Path(self.config.workspace) / "discoveries").list()
            evaluations = ComparativeEvaluationRuntime(ledger, Path(self.config.workspace) / "evaluations").list_suites()
            integration_status = ledger.integrations.status()
            schedules = ledger.scheduler.list()
            pending_review = sum(
                1
                for batch in batches
                for item in batch["items"]
                if item["status"] == "quarantined" and item["requires_human_review"]
            )
            integrity_failures = sum(
                1
                for receipt in receipts
                if not (
                    receipt["integrity_valid"]
                    and receipt["artifact_integrity_valid"]
                    and receipt["evidence_binding_valid"]
                )
            )
            event_rows = ledger.db.conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT 12"
            ).fetchall()
            recent = [self._event_row(row) for row in event_rows]
            return {
                "claims": {
                    "total": len(claims),
                    "by_status": claim_status,
                    "by_support": support_state,
                },
                "analyses": {
                    "total": len(receipts),
                    "integrity_failures": integrity_failures,
                    "reproduced": sum(r["status"] == "reproduced" for r in receipts),
                },
                "proposals": {
                    "batches": len(batches),
                    "pending_review": pending_review,
                },
                "graphs": {"snapshots": len(snapshots), "diffs": len(diffs)},
                "discoveries": {
                    "total": len(discoveries),
                    "awaiting_approval": sum("awaiting" in d["status"] for d in discoveries),
                    "concluded": sum(d["status"] == "concluded" for d in discoveries),
                    "failed": sum(d["status"] in {"failed", "budget_exhausted"} for d in discoveries),
                },
                "evaluations": {
                    "suites": len(evaluations),
                    "runs": sum(len(ComparativeEvaluationRuntime(ledger, Path(self.config.workspace) / "evaluations").list_runs(item["id"])) for item in evaluations),
                    "reports": sum(bool(item.get("report_hash")) for item in evaluations),
                },
                "executions": {
                    "total": len(executions),
                    "waiting_approval": sum(r["status"] == "waiting_approval" for r in executions),
                    "succeeded": sum(r["status"] == "succeeded" for r in executions),
                    "failed": sum(r["status"] == "failed" for r in executions),
                    "engines": execution_status["engines"],
                },
                "automation": {
                    "drafts": integration_status["drafts"],
                    "receipts": integration_status["receipts"],
                    "schedules": len(schedules),
                    "blocked": sum(item["status"] == "blocked" for item in schedules),
                    "active": sum(item["status"] == "active" for item in schedules),
                },
                "recent_events": recent,
            }

    def _claims(self, query: dict[str, list[str]]) -> dict[str, Any]:
        text = query.get("q", [""])[0].strip().casefold()
        state_filter = query.get("state", [""])[0].strip()
        status_filter = query.get("status", [""])[0].strip()
        limit = min(max(int(query.get("limit", ["200"])[0]), 1), 1000)
        with EpistemicLedger(self.config.db_path) as ledger:
            claims = ledger.list_claims()
            reports = SupportEngine(ledger).evaluate_many([c["id"] for c in claims])
            items = []
            for claim in reversed(claims):
                state = reports[claim["id"]].state.value
                if text and text not in claim["canonical_text"].casefold() and text not in claim["id"]:
                    continue
                if state_filter and state != state_filter:
                    continue
                if status_filter and claim["status"] != status_filter:
                    continue
                relation = claim.get("relation")
                items.append(
                    {
                        "id": claim["id"],
                        "canonical_text": claim["canonical_text"],
                        "claim_type": claim["claim_type"],
                        "status": claim["status"],
                        "support_state": state,
                        "created_at": claim["created_at"],
                        "relation": relation,
                    }
                )
                if len(items) >= limit:
                    break
            return {"items": items, "count": len(items)}

    def _claim_detail(self, claim_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            claim = ledger.get_claim(claim_id)
            report = SupportEngine(ledger).evaluate(claim_id).as_dict()
            evidence_rows = ledger.db.conn.execute(
                """SELECT e.*, a.id AS attestation_id, a.stance, a.confidence,
                          a.created_at AS attested_at
                   FROM attestations a JOIN evidence e ON e.id = a.evidence_id
                   WHERE a.claim_id = ? ORDER BY a.created_at DESC""",
                (claim_id,),
            ).fetchall()
            evidence = []
            for row in evidence_rows:
                item = dict(row)
                item["active"] = bool(item["active"])
                item["metadata"] = json.loads(item.pop("metadata_json"))
                evidence.append(item)
            proof_rows = ledger.db.conn.execute(
                "SELECT * FROM proofs WHERE conclusion_claim_id = ? ORDER BY created_at",
                (claim_id,),
            ).fetchall()
            proofs = []
            for row in proof_rows:
                item = dict(row)
                item["active"] = bool(item["active"])
                item["metadata"] = json.loads(item.pop("metadata_json"))
                premise_rows = ledger.db.conn.execute(
                    """SELECT pp.position, c.id, c.canonical_text
                       FROM proof_premises pp JOIN claims c ON c.id = pp.premise_claim_id
                       WHERE pp.proof_id = ? ORDER BY pp.position""",
                    (item["id"],),
                ).fetchall()
                item["premises"] = [dict(p) for p in premise_rows]
                proofs.append(item)
            contradiction_rows = ledger.db.conn.execute(
                """SELECT c.*, ca.canonical_text AS claim_a_text,
                          cb.canonical_text AS claim_b_text
                   FROM contradictions c
                   JOIN claims ca ON ca.id = c.claim_a
                   JOIN claims cb ON cb.id = c.claim_b
                   WHERE c.claim_a = ? OR c.claim_b = ? ORDER BY c.created_at""",
                (claim_id, claim_id),
            ).fetchall()
            contradictions = []
            for row in contradiction_rows:
                item = dict(row)
                item["active"] = bool(item["active"])
                contradictions.append(item)
            return {
                "claim": claim,
                "support": report,
                "evidence": evidence,
                "proofs": proofs,
                "contradictions": contradictions,
                "history": ledger.history("claim", claim_id),
                "descendants": ledger.descendants_of_claim(claim_id),
            }

    def _create_claim(self, payload: dict[str, Any]) -> dict[str, Any]:
        for field_name in ("subject", "predicate", "object_value"):
            if field_name not in payload or str(payload[field_name]).strip() == "":
                raise ValueError(f"{field_name} is required")
        object_kind = ObjectKind(payload.get("object_kind", "entity"))
        object_value: Any = payload["object_value"]
        object_type = payload.get("object_type") or "thing"
        if object_kind == ObjectKind.LITERAL:
            datatype = LiteralDatatype(payload.get("literal_datatype", "string"))
            object_value = TypedLiteral(
                self._parse_literal(payload["object_value"], datatype),
                datatype,
                payload.get("unit") or None,
            )
            object_type = None
        qualifiers = payload.get("qualifiers") or {}
        if not isinstance(qualifiers, dict):
            raise ValueError("qualifiers must be a JSON object")
        with EpistemicLedger(self.config.db_path) as ledger:
            claim_id = ledger.add_relation_claim(
                str(payload["subject"]),
                str(payload["predicate"]),
                object_value,
                subject_type=str(payload.get("subject_type") or "thing"),
                object_kind=object_kind,
                object_type=object_type,
                polarity=bool(payload.get("polarity", True)),
                valid_from=payload.get("valid_from") or None,
                valid_to=payload.get("valid_to") or None,
                qualifiers=qualifiers,
                actor=str(payload.get("actor") or "ui-user"),
                actor_role=ActorRole.HUMAN,
            )
            return self._claim_detail(claim_id)

    def _add_claim_evidence(self, claim_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        for field_name in ("source_uri", "excerpt", "independence_key"):
            if not str(payload.get(field_name, "")).strip():
                raise ValueError(f"{field_name} is required")
        kind = EvidenceKind(payload.get("source_kind", EvidenceKind.HUMAN_TESTIMONY.value))
        stance = Stance(payload.get("stance", Stance.SUPPORT.value))
        confidence = float(payload.get("confidence", 1.0))
        with EpistemicLedger(self.config.db_path) as ledger:
            ledger._require_claim(claim_id)
            evidence_id = ledger.add_evidence(
                str(payload["source_uri"]),
                str(payload["excerpt"]),
                source_kind=kind,
                independence_key=str(payload["independence_key"]),
                content=payload.get("content") or str(payload["excerpt"]),
                metadata=payload.get("metadata") or {},
                actor=str(payload.get("actor") or "ui-user"),
                actor_role=ActorRole.HUMAN,
            )
            ledger.attest(
                claim_id,
                evidence_id,
                stance,
                confidence=confidence,
                actor=str(payload.get("actor") or "ui-user"),
                actor_role=ActorRole.HUMAN,
            )
        return self._claim_detail(claim_id)

    def _revoke_evidence(self, evidence_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        rationale = str(payload.get("rationale", "")).strip()
        if not rationale:
            raise ValueError("rationale is required")
        with EpistemicLedger(self.config.db_path) as ledger:
            affected = ledger.revoke_evidence(
                evidence_id,
                rationale=rationale,
                actor=str(payload.get("actor") or "ui-user"),
                actor_role=ActorRole.HUMAN,
            )
            reports = SupportEngine(ledger).collapse_report(affected)
            return {"evidence_id": evidence_id, "affected_claim_ids": affected, "reports": reports}

    # ------------------------------------------------------------------
    # Analyses
    # ------------------------------------------------------------------
    def _analyses(self) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            receipts = list(reversed(ledger.list_analysis_receipts()))
            return {"items": receipts, "count": len(receipts)}

    def _analysis_detail(self, receipt_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.get_analysis_receipt(receipt_id)

    def _run_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        filename = self._clean_filename(str(payload.get("filename") or "dataset.csv"))
        csv_text = payload.get("csv_text")
        if not isinstance(csv_text, str) or not csv_text.strip():
            raise ValueError("csv_text is required")
        raw = csv_text.encode("utf-8")
        if len(raw) > self.config.max_upload_bytes:
            raise ValueError(
                f"CSV exceeds the configured {self.config.max_upload_bytes} byte upload limit"
            )
        run_dir = Path(self.config.workspace) / "uploads" / f"upl_{uuid.uuid4().hex[:16]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        dataset_path = run_dir / filename
        dataset_path.write_bytes(raw)
        spec = DatasetAnalysisSpec.from_dict(
            {
                "dataset_path": str(dataset_path),
                "analysis_type": payload.get("analysis_type"),
                "parameters": payload.get("parameters") or {},
                "preprocessing": payload.get("preprocessing") or {},
                "claim_tests": payload.get("claim_tests") or [],
                "metadata": {
                    "submitted_via": "orbita_ui",
                    "original_filename": filename,
                    **(payload.get("metadata") or {}),
                },
            }
        )
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.run_analysis(spec, actor="orbita-ui", actor_role=ActorRole.TOOL)

    def _reproduce_analysis(self, receipt_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            dataset_path = payload.get("dataset_path") or None
            return ledger.reproduce_analysis(receipt_id, dataset_path=dataset_path)

    # ------------------------------------------------------------------
    # Governed discovery investigations
    # ------------------------------------------------------------------
    def _discovery_runtime(self, ledger: EpistemicLedger) -> GovernedDiscoveryRuntime:
        return GovernedDiscoveryRuntime(ledger, Path(self.config.workspace) / "discoveries")

    def _discoveries(self) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            items = list(reversed(self._discovery_runtime(ledger).list()))
            return {"items": items, "count": len(items)}

    def _discovery_detail(self, investigation_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._discovery_runtime(ledger).get(investigation_id)

    def _create_discovery(self, payload: dict[str, Any]) -> dict[str, Any]:
        csv_text = payload.get("csv_text")
        if not isinstance(csv_text, str) or not csv_text.strip():
            raise ValueError("csv_text is required")
        raw = csv_text.encode("utf-8")
        if len(raw) > self.config.max_upload_bytes:
            raise APIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Dataset exceeds upload limit")
        filename = self._clean_filename(str(payload.get("filename") or "dataset.csv"))
        run_dir = Path(self.config.workspace) / "discoveries" / "uploads" / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=False)
        dataset_path = run_dir / filename
        dataset_path.write_bytes(raw)
        replication_path = None
        replication_text = payload.get("replication_csv_text")
        if replication_text not in {None, ""}:
            if not isinstance(replication_text, str):
                raise ValueError("replication_csv_text must be text")
            replication_raw = replication_text.encode("utf-8")
            if len(replication_raw) > self.config.max_upload_bytes:
                raise APIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Replication dataset exceeds upload limit")
            replication_name = self._clean_filename(str(payload.get("replication_filename") or "replication.csv"))
            replication_path = run_dir / replication_name
            replication_path.write_bytes(replication_raw)
        spec_value = {
            "question": payload.get("question"),
            "dataset_path": str(dataset_path),
            "replication_dataset_path": str(replication_path) if replication_path else None,
            "image": payload.get("image"),
            "candidate_hypotheses": payload.get("candidate_hypotheses") or [],
            "seed": payload.get("seed", 20260619),
            "discovery_fraction": payload.get("discovery_fraction", 0.5),
            "min_rows": payload.get("min_rows", 12),
            "min_discovery_abs_r": payload.get("min_discovery_abs_r", 0.45),
            "min_confirmation_abs_r": payload.get("min_confirmation_abs_r", 0.35),
            "permutation_trials": payload.get("permutation_trials", 199),
            "bootstrap_trials": payload.get("bootstrap_trials", 200),
            "budget": payload.get("budget") or {},
            "metadata": {"submitted_via": "orbita_ui", **(payload.get("metadata") or {})},
        }
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._discovery_runtime(ledger).create(
                DiscoverySpec.from_dict(spec_value), actor="orbita-ui", actor_role=ActorRole.HUMAN
            )

    def _approve_discovery(self, investigation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        reviewer = str(payload.get("reviewer", "")).strip()
        rationale = str(payload.get("rationale", "")).strip()
        with EpistemicLedger(self.config.db_path) as ledger:
            runtime = self._discovery_runtime(ledger)
            inv = runtime.get(investigation_id)
            run_id = inv["resume_cursor"].get(f"{inv['current_phase']}_run_id")
            if not run_id:
                raise ValueError("Investigation has no current execution manifest")
            return runtime.executions.approve(run_id, reviewer=reviewer, rationale=rationale)

    def _advance_discovery(self, investigation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        requested = payload.get("engine")
        engine = CliOCIEngine(str(requested)) if requested else None
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._discovery_runtime(ledger).advance(investigation_id, engine=engine)

    def _compile_discovery_report(self, investigation_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            runtime = self._discovery_runtime(ledger)
            runtime.compile_report(investigation_id)
            return runtime.get(investigation_id)


    # ------------------------------------------------------------------
    # Comparative evaluations
    # ------------------------------------------------------------------
    def _evaluation_runtime(self, ledger: EpistemicLedger) -> ComparativeEvaluationRuntime:
        return ComparativeEvaluationRuntime(ledger, Path(self.config.workspace) / "evaluations")

    def _evaluations(self) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            runtime = self._evaluation_runtime(ledger)
            items = runtime.list_suites()
            for item in items:
                item["runs"] = runtime.list_runs(item["id"])
            return {"items": items, "count": len(items)}

    def _evaluation_detail(self, suite_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._evaluation_runtime(ledger).get_suite(suite_id)

    def _create_default_evaluation(self) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._evaluation_runtime(ledger).create_suite(default_adversarial_suite())

    def _create_evaluation_fixture(self, suite_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        profile = str(payload.get("profile", "")).strip()
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._evaluation_runtime(ledger).create_fixture_run(suite_id, profile)

    def _compile_evaluation_report(self, suite_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            runtime = self._evaluation_runtime(ledger)
            compiled = runtime.compile_report(suite_id)
            return {"compiled": compiled, "suite": runtime.get_suite(suite_id)}

    def _evaluation_run_detail(self, run_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._evaluation_runtime(ledger).get_run(run_id)

    # ------------------------------------------------------------------
    # Manifest-bound container executions
    # ------------------------------------------------------------------
    def _execution_runtime(self, ledger: EpistemicLedger) -> ContainerExecutionRuntime:
        return ContainerExecutionRuntime(ledger, Path(self.config.workspace) / "executions")

    def _executions(self) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            runtime = self._execution_runtime(ledger)
            items = runtime.list()
            return {"items": items, "count": len(items), "runtime": runtime.runtime_status()}

    def _execution_detail(self, run_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._execution_runtime(ledger).get(run_id)

    def _submit_execution(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw = payload.get("spec", payload)
        if not isinstance(raw, dict):
            raise ValueError("spec must be a JSON object")
        spec = ContainerExecutionSpec.from_dict(raw)
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._execution_runtime(ledger).submit(
                spec, actor="orbita-ui", actor_role=ActorRole.HUMAN
            )

    def _approve_execution(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        reviewer = str(payload.get("reviewer", "")).strip()
        rationale = str(payload.get("rationale", "")).strip()
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._execution_runtime(ledger).approve(
                run_id, reviewer=reviewer, rationale=rationale
            )

    def _reject_execution(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        reviewer = str(payload.get("reviewer", "")).strip()
        rationale = str(payload.get("rationale", "")).strip()
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._execution_runtime(ledger).reject(
                run_id, reviewer=reviewer, rationale=rationale
            )

    def _run_execution(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        requested = payload.get("engine")
        engine = CliOCIEngine(str(requested)) if requested else None
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._execution_runtime(ledger).execute(run_id, engine=engine)

    def _reproduce_execution(self, run_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._execution_runtime(ledger).prepare_reproduction(
                run_id, actor="orbita-ui", actor_role=ActorRole.HUMAN
            )

    # ------------------------------------------------------------------
    # Model proposals
    # ------------------------------------------------------------------
    def _proposals(self, query: dict[str, list[str]]) -> dict[str, Any]:
        status = query.get("status", [None])[0]
        with EpistemicLedger(self.config.db_path) as ledger:
            batches = list(reversed(ledger.list_proposal_batches(status=status)))
            return {"items": batches, "count": len(batches)}

    def _proposal_detail(self, batch_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.get_proposal_batch(batch_id)

    def _ingest_proposal(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_response = payload.get("raw_response")
        if isinstance(raw_response, dict):
            raw_response = json.dumps(raw_response, ensure_ascii=False)
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise ValueError("raw_response is required")
        identity = ModelIdentity(
            str(payload.get("provider") or "manual-import"),
            str(payload.get("model_name") or "unknown-model"),
            str(payload.get("model_version")) if payload.get("model_version") else None,
        )
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.ingest_model_response(
                raw_response,
                identity=identity,
                system_prompt=str(payload.get("system_prompt") or "Imported through Orbita UI"),
                user_prompt=str(payload.get("user_prompt") or "Model proposal import"),
                generation_parameters=payload.get("generation_parameters") or {},
                response_id=payload.get("response_id") or None,
                usage=payload.get("usage") or {},
                metadata={"submitted_via": "orbita_ui", **(payload.get("metadata") or {})},
            )

    def _review_proposal(self, item_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        reviewer = str(payload.get("reviewer", "")).strip()
        rationale = str(payload.get("rationale", "")).strip()
        if not reviewer or not rationale:
            raise ValueError("reviewer and rationale are required")
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.review_proposal_item(
                item_id,
                ReviewDecision(payload.get("decision")),
                reviewer=reviewer,
                rationale=rationale,
            )

    # ------------------------------------------------------------------
    # Graphs and audit
    # ------------------------------------------------------------------
    def _graphs(self) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return {
                "snapshots": list(reversed(ledger.graphs.list_snapshots())),
                "diffs": list(reversed(ledger.graphs.list_diffs())),
            }

    def _capture_graph(self, payload: dict[str, Any]) -> dict[str, Any]:
        roots = payload.get("root_claim_ids") or []
        if not isinstance(roots, list):
            raise ValueError("root_claim_ids must be an array")
        with EpistemicLedger(self.config.db_path) as ledger:
            snapshot = ledger.capture_graph(
                name=str(payload.get("name") or "UI epistemic snapshot"),
                root_claim_ids=[str(item) for item in roots],
                include_descendants=bool(payload.get("include_descendants", False)),
                actor="orbita-ui",
            )
            output_dir = Path(self.config.workspace) / "graphs" / snapshot["id"]
            artifacts = ledger.graphs.render_snapshot(
                snapshot, output_dir=output_dir, formats=("json", "svg", "html")
            )
            return {
                "snapshot": snapshot,
                "artifacts": artifacts,
                "svg": self._svg_from_artifacts(artifacts, "snapshot_svg"),
            }

    def _graph_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            snapshot = ledger.get_graph_snapshot(snapshot_id)
            artifacts = ledger.graphs.list_artifacts(snapshot_id=snapshot_id)
            svg = self._svg_from_artifacts(artifacts, "snapshot_svg", required=False)
            if svg is None:
                output_dir = Path(self.config.workspace) / "graphs" / snapshot_id
                artifacts = ledger.graphs.render_snapshot(
                    snapshot, output_dir=output_dir, formats=("svg", "html")
                )
                svg = self._svg_from_artifacts(artifacts, "snapshot_svg")
            return {
                "snapshot": snapshot,
                "artifacts": artifacts,
                "integrity_valid": ledger.graphs.verify_snapshot(snapshot_id),
                "artifacts_valid": ledger.graphs.verify_artifacts(snapshot_id=snapshot_id),
                "svg": svg,
            }

    def _compare_graphs(self, payload: dict[str, Any]) -> dict[str, Any]:
        before = str(payload.get("before_snapshot_id") or "")
        after = str(payload.get("after_snapshot_id") or "")
        if not before or not after:
            raise ValueError("before_snapshot_id and after_snapshot_id are required")
        with EpistemicLedger(self.config.db_path) as ledger:
            diff = ledger.compare_graphs(
                before,
                after,
                name=str(payload.get("name") or "UI epistemic collapse diff"),
                actor="orbita-ui",
            )
            output_dir = Path(self.config.workspace) / "graphs" / diff["id"]
            artifacts = ledger.graphs.render_diff(
                diff, output_dir=output_dir, formats=("json", "svg", "html")
            )
            return {
                "diff": diff,
                "artifacts": artifacts,
                "svg": self._svg_from_artifacts(artifacts, "diff_svg"),
            }

    def _graph_diff(self, diff_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            diff = ledger.get_graph_diff(diff_id)
            artifacts = ledger.graphs.list_artifacts(diff_id=diff_id)
            svg = self._svg_from_artifacts(artifacts, "diff_svg", required=False)
            if svg is None:
                output_dir = Path(self.config.workspace) / "graphs" / diff_id
                artifacts = ledger.graphs.render_diff(
                    diff, output_dir=output_dir, formats=("svg", "html")
                )
                svg = self._svg_from_artifacts(artifacts, "diff_svg")
            return {
                "diff": diff,
                "artifacts": artifacts,
                "integrity_valid": ledger.graphs.verify_diff(diff_id),
                "artifacts_valid": ledger.graphs.verify_artifacts(diff_id=diff_id),
                "svg": svg,
            }

    # ------------------------------------------------------------------
    # Computer agent
    # ------------------------------------------------------------------
    def _agent_runtime(self, ledger: EpistemicLedger) -> ComputerAgentRuntime:
        return ComputerAgentRuntime(ledger, Path(self.config.workspace) / "computer")

    def _agent_skills(self) -> list[dict[str, Any]]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._agent_runtime(ledger).registry.list_contracts()

    def _agent_state(self, goal_id: str | None = None) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._agent_runtime(ledger).machine_state(goal_id=goal_id)

    def _agent_compile(self, payload: dict[str, Any]) -> dict[str, Any]:
        utterance = str(payload.get("utterance", "")).strip()
        if not utterance:
            raise ValueError("utterance is required")
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._agent_runtime(ledger).compile_goal(utterance)

    def _agent_goals(self) -> list[dict[str, Any]]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._agent_runtime(ledger).list_goals()

    def _agent_create_goal(self, payload: dict[str, Any]) -> dict[str, Any]:
        utterance = str(payload.get("utterance", "")).strip()
        if not utterance:
            raise ValueError("utterance is required")
        mode = AutonomyMode(str(payload.get("autonomy_mode", AutonomyMode.VERIFIED.value)))
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._agent_runtime(ledger).create_goal(utterance, autonomy_mode=mode)

    def _agent_goal(self, goal_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._agent_runtime(ledger).get_goal(goal_id)

    def _agent_plan_goal(self, goal_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._agent_runtime(ledger).plan_goal(goal_id)

    def _agent_plan(self, plan_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._agent_runtime(ledger).get_plan(plan_id)

    def _agent_run_plan(self, plan_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._agent_runtime(ledger).run_until_blocked(plan_id)

    def _agent_approve(self, approval_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        reviewer = str(payload.get("reviewer", "")).strip()
        rationale = str(payload.get("rationale", "")).strip()
        with EpistemicLedger(self.config.db_path) as ledger:
            return self._agent_runtime(ledger).approve(approval_id, reviewer=reviewer, rationale=rationale)

    # ------------------------------------------------------------------
    # Governed integrations and schedules
    # ------------------------------------------------------------------
    def _adaptive_status(self) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.adaptive.status()

    def _desktop_observations(self, limit: int = 100) -> list[dict[str, Any]]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.adaptive.list_observations(limit)

    def _desktop_observation(self, observation_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.adaptive.get_observation(observation_id)

    def _adaptive_workflows(self, status: str | None = None) -> list[dict[str, Any]]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.adaptive.list_workflows(status)

    def _adaptive_workflow(self, workflow_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.adaptive.get_workflow(workflow_id)

    def _integration_status(self) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.integrations.status()

    def _integration_drafts(self) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            items = ledger.integrations.list_drafts()
            return {"count": len(items), "items": items}

    def _create_integration_draft(self, payload: dict[str, Any]) -> dict[str, Any]:
        kind = str(payload.get("kind", "email"))
        with EpistemicLedger(self.config.db_path) as ledger:
            if kind == "email":
                return ledger.integrations.create_email_draft(
                    to=payload.get("to", []),
                    subject=str(payload.get("subject", "")),
                    body=str(payload.get("body", "")),
                    cc=payload.get("cc"),
                    bcc=payload.get("bcc"),
                    actor=str(payload.get("actor", "local-ui")),
                )
            if kind == "calendar":
                return ledger.integrations.create_calendar_draft(
                    title=str(payload.get("title", "")),
                    start=str(payload.get("start", "")),
                    end=str(payload.get("end", "")),
                    timezone_name=str(payload.get("timezone", "UTC")),
                    attendees=list(payload.get("attendees", [])),
                    location=payload.get("location"),
                    description=payload.get("description"),
                    actor=str(payload.get("actor", "local-ui")),
                )
            raise ValueError("kind must be email or calendar")

    def _integration_draft(self, draft_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.integrations.get_draft(draft_id)

    def _request_integration_approval(self, draft_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.integrations.request_draft_approval(draft_id)

    def _decide_integration_approval(self, approval_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.integrations.decide_draft_approval(
                approval_id,
                approved=str(payload.get("decision", "approve")) == "approve",
                reviewer=str(payload.get("reviewer", "")),
                rationale=str(payload.get("rationale", "")),
            )

    def _schedules(self) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            items = ledger.scheduler.list()
            return {"count": len(items), "items": items}

    def _create_schedule(self, payload: dict[str, Any]) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            common = {
                "name": str(payload.get("name", "")),
                "goal_utterance": str(payload.get("goal", "")),
                "workspace": payload.get("workspace") or self.config.workspace,
                "autonomy_mode": str(payload.get("autonomy_mode", AutonomyMode.VERIFIED.value)),
                "actor": str(payload.get("actor", "local-ui")),
            }
            if str(payload.get("schedule_kind", "once")) == "interval":
                return ledger.scheduler.create_interval(
                    every_seconds=int(payload.get("every_seconds", 3600)),
                    first_run_at=payload.get("first_run_at"),
                    max_runs=payload.get("max_runs"),
                    **common,
                )
            return ledger.scheduler.create_once(run_at=str(payload.get("run_at", "")), **common)

    def _schedule(self, schedule_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.scheduler.get(schedule_id)

    def _resume_schedule(self, schedule_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.scheduler.resume(schedule_id, str(payload.get("worker", "local-ui")))

    def _set_schedule_state(self, schedule_id: str, action: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            if action == "pause":
                return ledger.scheduler.pause(schedule_id)
            if action == "activate":
                return ledger.scheduler.activate(schedule_id)
            if action == "cancel":
                return ledger.scheduler.cancel(schedule_id)
            raise ValueError("Unknown schedule action")

    def _scheduler_tick(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.scheduler.tick(
                str(payload.get("worker", "local-ui")),
                max_jobs=int(payload.get("max_jobs", 10)),
                now=payload.get("now"),
            )

    # ------------------------------------------------------------------
    # Warranted language
    # ------------------------------------------------------------------
    def _language_ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        utterance = str(payload.get("utterance", "")).strip()
        if not utterance:
            raise ValueError("utterance is required")
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.language.ask(utterance)

    def _language_interpret(self, payload: dict[str, Any]) -> dict[str, Any]:
        utterance = str(payload.get("utterance", "")).strip()
        if not utterance:
            raise ValueError("utterance is required")
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.language.interpret(utterance)

    def _language_response(self, response_id: str) -> dict[str, Any]:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.language.get_response(response_id)

    def _language_verify(self, response_id: str) -> bool:
        with EpistemicLedger(self.config.db_path) as ledger:
            return ledger.language.verify_response(response_id)

    def _events(self, query: dict[str, list[str]]) -> dict[str, Any]:
        limit = min(max(int(query.get("limit", ["200"])[0]), 1), 1000)
        with EpistemicLedger(self.config.db_path) as ledger:
            rows = ledger.db.conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return {"items": [self._event_row(row) for row in rows], "count": len(rows)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _event_row(row: Any) -> dict[str, Any]:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        return item

    @staticmethod
    def _parse_literal(value: Any, datatype: LiteralDatatype) -> Any:
        if datatype == LiteralDatatype.INTEGER:
            return int(value)
        if datatype == LiteralDatatype.FLOAT:
            return float(value)
        if datatype == LiteralDatatype.BOOLEAN:
            if isinstance(value, bool):
                return value
            lowered = str(value).strip().casefold()
            if lowered not in {"true", "false"}:
                raise ValueError("Boolean literals must be true or false")
            return lowered == "true"
        if datatype == LiteralDatatype.JSON:
            return value if isinstance(value, (dict, list)) else json.loads(str(value))
        return str(value)

    @staticmethod
    def _clean_filename(value: str) -> str:
        name = Path(value).name
        name = _SAFE_FILENAME.sub("_", name).strip("._") or "dataset.csv"
        if not name.casefold().endswith(".csv"):
            name += ".csv"
        return name[:160]

    def _svg_from_artifacts(
        self, artifacts: list[dict[str, Any]], role: str, *, required: bool = True
    ) -> str | None:
        for artifact in reversed(artifacts):
            if artifact["role"] != role:
                continue
            path = Path(artifact["path"]).expanduser().resolve()
            allowed_roots = [
                Path(self.config.workspace).resolve(),
                (self.config.db_path.parent / "graph_artifacts").resolve(),
            ]
            if not any(path.is_relative_to(root) for root in allowed_roots):
                raise ValueError("Graph artifact escaped the configured workspace")
            text = path.read_text(encoding="utf-8")
            if _UNSAFE_SVG.search(text):
                raise ValueError("Unsafe SVG content detected")
            return text
        if required:
            raise FileNotFoundError(f"No {role} artifact found")
        return None

    @staticmethod
    def _decode_json(body: bytes) -> dict[str, Any]:
        if len(body) > _MAX_JSON_BYTES:
            raise APIError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request body is too large")
        if not body:
            return {}
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise APIError(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise APIError(HTTPStatus.BAD_REQUEST, "Top-level JSON must be an object")
        return payload

    @staticmethod
    def _ok(value: Any, *, status: int = HTTPStatus.OK) -> UIResponse:
        return UIResponse(
            status,
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        )

    @staticmethod
    def _error(status: int, message: str, details: Any = None) -> UIResponse:
        return UIResponse(
            int(status),
            json.dumps(
                {"ok": False, "error": message, "details": details},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"),
        )


class _OrbitaHandler(BaseHTTPRequestHandler):
    app: OrbitaUIApp
    server_version = "OrbitaUI/0.8"

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self._security_headers()
        self.end_headers()

    def _dispatch(self, method: str) -> None:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length > _MAX_JSON_BYTES:
            response = OrbitaUIApp._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request body is too large"
            )
        else:
            body = self.rfile.read(content_length) if content_length else b""
            response = self.app.handle(method, self.path, dict(self.headers.items()), body)
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        for key, value in response.headers.items():
            self.send_header(key, value)
        self._security_headers()
        self.end_headers()
        self.wfile.write(response.body)

    def _security_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[orbita-ui] {self.address_string()} - {fmt % args}")


def build_ui_server(config: UIConfig) -> ThreadingHTTPServer:
    app = OrbitaUIApp(config)

    class Handler(_OrbitaHandler):
        pass

    Handler.app = app
    server = ThreadingHTTPServer((config.host, config.port), Handler)
    server.daemon_threads = True
    return server


def serve_ui(config: UIConfig) -> None:
    server = build_ui_server(config)
    actual_port = server.server_address[1]
    display_host = "127.0.0.1" if config.host in {"0.0.0.0", "::"} else config.host
    display_host = f"[{display_host}]" if ":" in display_host and not display_host.startswith("[") else display_host
    url = f"http://{display_host}:{actual_port}/?token={config.access_token}"
    print("Orbita Epistemic Intelligence UI")
    print(f"Database:  {config.db_path}")
    print(f"Workspace: {config.workspace}")
    print(f"Open:      {url}")
    if config.allow_remote:
        print("Remote mode is enabled. Share the tokenized URL only over a trusted private tunnel.")
    if config.open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping Orbita UI...")
    finally:
        server.server_close()
