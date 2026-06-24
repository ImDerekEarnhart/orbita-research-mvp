from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .agent_os import (
    AutonomyMode,
    ComputerAgentRuntime,
    ComputerStepStatus,
    SkillContract,
    SkillResult,
)
from .models import ActorRole, RiskLevel

if TYPE_CHECKING:  # pragma: no cover
    from .ledger import EpistemicLedger


INTEGRATION_API_VERSION = "1.4"
MAX_PROVIDER_RESPONSE_BYTES = 2 * 1024 * 1024


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_time(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        raw = value.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


class CapabilityProvider(Protocol):
    name: str

    def invoke(
        self,
        capability: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]: ...


class CallableCapabilityProvider:
    """Small provider adapter useful for local bridges and deterministic tests."""

    def __init__(
        self,
        name: str,
        callback: Callable[[str, dict[str, Any], str], dict[str, Any]],
    ) -> None:
        self.name = name
        self._callback = callback

    def invoke(
        self,
        capability: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        result = self._callback(capability, dict(payload), idempotency_key)
        if not isinstance(result, dict):
            raise TypeError("Capability provider must return a JSON object")
        return result


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class OpenClawBridge:
    """HMAC-authenticated bridge to an OpenClaw-compatible local broker.

    The bridge is loopback-only by default. It sends no user credentials; the
    broker owns its own account sessions and returns a structured receipt.
    """

    name = "openclaw"

    def __init__(
        self,
        endpoint: str,
        shared_secret: str,
        *,
        allow_remote: bool = False,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not shared_secret:
            raise ValueError("OpenClaw shared secret is required")
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("OpenClaw endpoint must use http or https")
        if not parsed.hostname:
            raise ValueError("OpenClaw endpoint must include a hostname")
        if parsed.username or parsed.password:
            raise ValueError("Credentials must not be embedded in the OpenClaw URL")
        if not allow_remote and not self._is_loopback(parsed.hostname):
            raise ValueError("Remote OpenClaw endpoints are disabled by default")
        self.endpoint = endpoint.rstrip("/")
        self.shared_secret = shared_secret.encode("utf-8")
        self.timeout_seconds = float(timeout_seconds)
        self._opener = urllib.request.build_opener(_NoRedirect())

    @staticmethod
    def _is_loopback(hostname: str) -> bool:
        if hostname.casefold() == "localhost":
            return True
        try:
            return ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            # Do not trust DNS resolution for a local-only policy; this avoids
            # DNS-rebinding an accepted name to a remote endpoint later.
            return False

    def invoke(
        self,
        capability: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        envelope = {
            "api_version": INTEGRATION_API_VERSION,
            "capability": capability,
            "payload": payload,
            "idempotency_key": idempotency_key,
        }
        body = stable_json(envelope).encode("utf-8")
        request_hash = hashlib.sha256(body).hexdigest()
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        nonce = secrets.token_hex(16)
        signature_payload = f"{timestamp}\n{nonce}\n{request_hash}".encode("utf-8")
        signature = hmac.new(self.shared_secret, signature_payload, hashlib.sha256).hexdigest()
        request = urllib.request.Request(
            f"{self.endpoint}/v1/capabilities/invoke",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Orbita-Timestamp": timestamp,
                "X-Orbita-Nonce": nonce,
                "X-Orbita-Request-Hash": request_hash,
                "X-Orbita-Signature": signature,
            },
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
                response_signature = response.headers.get("X-Orbita-Response-Signature", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenClaw broker returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenClaw broker is unavailable: {exc.reason}") from exc
        if len(raw) > MAX_PROVIDER_RESPONSE_BYTES:
            raise RuntimeError("OpenClaw response exceeds 2 MiB")
        response_hash = hashlib.sha256(raw).hexdigest()
        expected_response_signature = hmac.new(
            self.shared_secret,
            f"{request_hash}\n{response_hash}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not response_signature or not hmac.compare_digest(
            response_signature, expected_response_signature
        ):
            raise RuntimeError("OpenClaw response signature is missing or invalid")
        result = json.loads(raw.decode("utf-8"))
        if not isinstance(result, dict):
            raise RuntimeError("OpenClaw response must be a JSON object")
        if result.get("request_hash") != request_hash:
            raise RuntimeError("OpenClaw response is not bound to the request hash")
        return result


class DraftKind(StrEnum):
    EMAIL = "email"
    CALENDAR = "calendar"


class DraftStatus(StrEnum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    EXECUTED = "executed"
    REJECTED = "rejected"
    FAILED = "failed"


class ScheduleKind(StrEnum):
    ONCE = "once"
    INTERVAL = "interval"


class ScheduleStatus(StrEnum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, attrs):  # noqa: ANN001
        low = tag.casefold()
        if low == "title":
            self._in_title = True
        if low in {"script", "style", "noscript"}:
            self._ignore_depth += 1

    def handle_endtag(self, tag: str) -> None:
        low = tag.casefold()
        if low == "title":
            self._in_title = False
        if low in {"script", "style", "noscript"} and self._ignore_depth:
            self._ignore_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignore_depth:
            return
        value = " ".join(data.split())
        if not value:
            return
        self.parts.append(value)
        if self._in_title:
            self.title_parts.append(value)

    @property
    def text(self) -> str:
        return " ".join(self.parts)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts)


@dataclass(slots=True)
class BrowserVerification:
    expected_url_prefix: str | None = None
    title_contains: str | None = None
    required_text: list[str] = field(default_factory=list)
    forbidden_text: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "BrowserVerification":
        raw = raw or {}
        return cls(
            expected_url_prefix=raw.get("expected_url_prefix"),
            title_contains=raw.get("title_contains"),
            required_text=[str(v) for v in raw.get("required_text", [])],
            forbidden_text=[str(v) for v in raw.get("forbidden_text", [])],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_url_prefix": self.expected_url_prefix,
            "title_contains": self.title_contains,
            "required_text": list(self.required_text),
            "forbidden_text": list(self.forbidden_text),
        }


class IntegrationRuntime:
    """Governed external integrations and draft-first side effects."""

    def __init__(self, ledger: "EpistemicLedger") -> None:
        self.ledger = ledger
        self.provider: CapabilityProvider | None = None
        self.install_agent_contracts()

    def bind_provider(self, provider: CapabilityProvider) -> None:
        self.provider = provider
        self.install_agent_contracts(self.ledger.agent)
        adaptive = getattr(self.ledger, "adaptive", None)
        if adaptive is not None:
            adaptive.bind_provider(provider)

    def install_agent_contracts(self, agent: ComputerAgentRuntime | None = None) -> None:
        registry = (agent or self.ledger.agent).registry
        contracts = [
            (
                SkillContract(
                    "integration.email.create_draft",
                    "Create a local email draft; this never sends mail",
                    RiskLevel.LOW,
                    True,
                    True,
                    ("to", "subject", "body"),
                    ({"type": "output_key", "key": "draft_id"},),
                ),
                self._email_draft_skill,
            ),
            (
                SkillContract(
                    "integration.calendar.create_draft",
                    "Create a local calendar-event draft; this never changes a calendar",
                    RiskLevel.LOW,
                    True,
                    True,
                    ("title", "start", "end", "timezone"),
                    ({"type": "output_key", "key": "draft_id"},),
                ),
                self._calendar_draft_skill,
            ),
            (
                SkillContract(
                    "external.browser.navigate_verified",
                    "Navigate through a provider and verify returned page content locally",
                    RiskLevel.MEDIUM,
                    False,
                    True,
                    ("url", "verification"),
                    ({"type": "output_value", "key": "verified", "equals": True},),
                    external_side_effect=True,
                ),
                None,
            ),
            (
                SkillContract(
                    "external.windows.launch_app",
                    "Launch one allowlisted Windows application through a capability provider",
                    RiskLevel.HIGH,
                    True,
                    True,
                    ("app_id",),
                    ({"type": "output_value", "key": "verified", "equals": True},),
                    external_side_effect=True,
                ),
                None,
            ),
        ]
        for contract, handler in contracts:
            if not registry.has_contract(contract.name):
                registry.register(contract, handler)
            elif handler is not None:
                registry.bind_handler(contract.name, handler)
        if self.provider is not None:
            registry.bind_handler("external.browser.navigate_verified", self._browser_skill)
            registry.bind_handler("external.windows.launch_app", self._windows_launch_skill)

    # Drafts -----------------------------------------------------------------

    def create_email_draft(
        self,
        *,
        to: str | list[str],
        subject: str,
        body: str,
        cc: str | list[str] | None = None,
        bcc: str | list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        actor: str = "user",
    ) -> dict[str, Any]:
        recipients = self._addresses(to)
        if not recipients:
            raise ValueError("At least one email recipient is required")
        payload = {
            "to": recipients,
            "cc": self._addresses(cc),
            "bcc": self._addresses(bcc),
            "subject": subject.strip(),
            "body": body,
        }
        if not payload["subject"]:
            raise ValueError("Email subject is required")
        return self._create_draft(DraftKind.EMAIL, payload, metadata or {}, actor)

    def create_calendar_draft(
        self,
        *,
        title: str,
        start: str,
        end: str,
        timezone_name: str,
        attendees: list[str] | None = None,
        location: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        actor: str = "user",
    ) -> dict[str, Any]:
        timezone_clean = timezone_name.strip()
        try:
            event_zone = ZoneInfo(timezone_clean)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone: {timezone_clean}") from exc

        def event_time(value: str) -> datetime:
            raw = value.strip().replace("Z", "+00:00")
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=event_zone)
            return parsed.astimezone(timezone.utc)

        start_dt, end_dt = event_time(start), event_time(end)
        if end_dt <= start_dt:
            raise ValueError("Calendar event end must be after start")
        payload = {
            "title": title.strip(),
            "start": start,
            "end": end,
            "timezone": timezone_clean,
            "attendees": self._addresses(attendees),
            "location": location or "",
            "description": description or "",
        }
        if not payload["title"] or not payload["timezone"]:
            raise ValueError("Calendar title and timezone are required")
        return self._create_draft(DraftKind.CALENDAR, payload, metadata or {}, actor)

    def _create_draft(
        self,
        kind: DraftKind,
        payload: dict[str, Any],
        metadata: dict[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        draft_id = new_id("drf")
        now = utcnow()
        payload_hash = sha256_json(payload)
        self.ledger.db.conn.execute(
            """INSERT INTO integration_drafts
               (id, kind, payload_json, payload_hash, metadata_json, status, created_by,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                draft_id,
                kind.value,
                stable_json(payload),
                payload_hash,
                stable_json(metadata),
                DraftStatus.DRAFT.value,
                actor,
                now,
                now,
            ),
        )
        self.ledger._event(
            "integration_draft",
            draft_id,
            "INTEGRATION_DRAFT_CREATED",
            {"kind": kind.value, "payload_hash": payload_hash},
            actor,
            ActorRole.HUMAN,
        )
        self.ledger.db.conn.commit()
        return self.get_draft(draft_id)

    def request_draft_approval(self, draft_id: str) -> dict[str, Any]:
        draft = self._draft_row(draft_id)
        self._require_draft_integrity(draft)
        if draft["status"] in {DraftStatus.EXECUTED.value, DraftStatus.REJECTED.value}:
            raise ValueError("Draft can no longer be approved")
        approval_id = new_id("iap")
        self.ledger.db.conn.execute(
            """INSERT INTO integration_approvals
               (id, draft_id, payload_hash, status, created_at)
               VALUES (?, ?, ?, 'pending', ?)""",
            (approval_id, draft_id, draft["payload_hash"], utcnow()),
        )
        self.ledger.db.conn.execute(
            "UPDATE integration_drafts SET status = ?, updated_at = ? WHERE id = ?",
            (DraftStatus.PENDING_APPROVAL.value, utcnow(), draft_id),
        )
        self.ledger.db.conn.commit()
        return self.get_approval(approval_id)

    def decide_draft_approval(
        self,
        approval_id: str,
        *,
        approved: bool,
        reviewer: str,
        rationale: str,
    ) -> dict[str, Any]:
        if not reviewer.strip() or not rationale.strip():
            raise ValueError("Reviewer and rationale are required")
        row = self.ledger.db.conn.execute(
            "SELECT * FROM integration_approvals WHERE id = ?", (approval_id,)
        ).fetchone()
        if row is None or row["status"] != "pending":
            raise ValueError("Approval not found or already decided")
        draft = self._draft_row(row["draft_id"])
        self._require_draft_integrity(draft)
        if row["payload_hash"] != draft["payload_hash"]:
            raise ValueError("Approval no longer matches the exact draft payload")
        status = "approved" if approved else "rejected"
        draft_status = DraftStatus.APPROVED.value if approved else DraftStatus.REJECTED.value
        now = utcnow()
        self.ledger.db.conn.execute(
            """UPDATE integration_approvals
               SET status = ?, reviewer = ?, rationale = ?, decided_at = ? WHERE id = ?""",
            (status, reviewer, rationale, now, approval_id),
        )
        self.ledger.db.conn.execute(
            "UPDATE integration_drafts SET status = ?, updated_at = ? WHERE id = ?",
            (draft_status, now, draft["id"]),
        )
        self.ledger._event(
            "integration_draft",
            draft["id"],
            "INTEGRATION_DRAFT_APPROVED" if approved else "INTEGRATION_DRAFT_REJECTED",
            {"approval_id": approval_id, "payload_hash": draft["payload_hash"], "rationale": rationale},
            reviewer,
            ActorRole.HUMAN,
        )
        self.ledger.db.conn.commit()
        return self.get_approval(approval_id)

    def execute_draft(self, draft_id: str) -> dict[str, Any]:
        if self.provider is None:
            raise RuntimeError("No capability provider is bound")
        draft = self._draft_row(draft_id)
        self._require_draft_integrity(draft)
        if draft["status"] == DraftStatus.EXECUTED.value:
            receipt = self.ledger.db.conn.execute(
                "SELECT id FROM integration_receipts WHERE draft_id = ? ORDER BY created_at DESC LIMIT 1",
                (draft_id,),
            ).fetchone()
            return self.get_receipt(receipt["id"]) if receipt else self.get_draft(draft_id)
        approval = self.ledger.db.conn.execute(
            """SELECT * FROM integration_approvals
               WHERE draft_id = ? AND payload_hash = ? AND status = 'approved' AND consumed_at IS NULL
               ORDER BY created_at DESC LIMIT 1""",
            (draft_id, draft["payload_hash"]),
        ).fetchone()
        if approval is None:
            raise PermissionError("The exact draft has not been approved")
        capability = "gmail.send_email" if draft["kind"] == DraftKind.EMAIL.value else "calendar.create_event"
        payload = json.loads(draft["payload_json"])
        idempotency_key = f"draft:{draft_id}:{draft['payload_hash']}"
        started = utcnow()
        ok = False
        error: str | None = None
        response: dict[str, Any] = {}
        try:
            response = self.provider.invoke(capability, payload, idempotency_key=idempotency_key)
            required_id = "message_id" if capability == "gmail.send_email" else "event_id"
            ok = response.get("ok") is True and bool(response.get(required_id))
            if not ok:
                error = str(
                    response.get("error")
                    or f"Provider did not return explicit ok=true and {required_id}"
                )
        except Exception as exc:  # provider boundary
            error = f"{type(exc).__name__}: {exc}"
        completed = utcnow()
        receipt_id = new_id("irc")
        receipt_payload = {
            "draft_id": draft_id,
            "kind": draft["kind"],
            "payload_hash": draft["payload_hash"],
            "provider": self.provider.name,
            "capability": capability,
            "idempotency_key": idempotency_key,
            "ok": ok,
            "response": response,
            "error": error,
            "started_at": started,
            "completed_at": completed,
        }
        receipt_hash = sha256_json(receipt_payload)
        self.ledger.db.conn.execute(
            """INSERT INTO integration_receipts
               (id, draft_id, action_kind, provider, capability, request_hash,
                response_json, ok, error, receipt_hash, started_at, completed_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                receipt_id,
                draft_id,
                draft["kind"],
                self.provider.name,
                capability,
                draft["payload_hash"],
                stable_json(response),
                int(ok),
                error,
                receipt_hash,
                started,
                completed,
                completed,
            ),
        )
        self.ledger.db.conn.execute(
            "UPDATE integration_approvals SET consumed_at = ? WHERE id = ?",
            (completed, approval["id"]),
        )
        self.ledger.db.conn.execute(
            "UPDATE integration_drafts SET status = ?, updated_at = ? WHERE id = ?",
            (DraftStatus.EXECUTED.value if ok else DraftStatus.FAILED.value, completed, draft_id),
        )
        self.ledger._event(
            "integration_draft",
            draft_id,
            "INTEGRATION_DRAFT_EXECUTED" if ok else "INTEGRATION_DRAFT_FAILED",
            {"receipt_id": receipt_id, "receipt_hash": receipt_hash, "error": error},
            "integration_runtime",
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()
        return self.get_receipt(receipt_id)

    # Browser ---------------------------------------------------------------

    def navigate_verified(
        self,
        url: str,
        verification: BrowserVerification | dict[str, Any],
    ) -> dict[str, Any]:
        if self.provider is None:
            raise RuntimeError("No capability provider is bound")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Browser URL must be http or https")
        spec = verification if isinstance(verification, BrowserVerification) else BrowserVerification.from_dict(verification)
        request_payload = {"url": url, "capture_html": True, "capture_screenshot_hash": True}
        request_hash = sha256_json({"capability": "browser.navigate", "payload": request_payload, "verification": spec.as_dict()})
        response = self.provider.invoke("browser.navigate", request_payload, idempotency_key=f"browser:{request_hash}")
        provider_ok = response.get("ok") is True
        final_url = str(response.get("final_url") or "")
        html = str(response.get("html") or "")
        if len(html.encode("utf-8")) > MAX_PROVIDER_RESPONSE_BYTES:
            raise ValueError("Browser HTML snapshot exceeds 2 MiB")
        parser = _TextExtractor()
        parser.feed(html)
        title = str(response.get("title") or parser.title)
        text = parser.text
        checks: list[dict[str, Any]] = []

        def add(name: str, ok: bool, detail: str) -> None:
            checks.append({"name": name, "ok": bool(ok), "detail": detail})

        if spec.expected_url_prefix:
            add(
                "expected_url_prefix",
                final_url.startswith(spec.expected_url_prefix),
                f"final_url={final_url!r}",
            )
        if spec.title_contains:
            add(
                "title_contains",
                spec.title_contains.casefold() in title.casefold(),
                f"title={title!r}",
            )
        for expected in spec.required_text:
            add(
                f"required_text:{expected}",
                expected.casefold() in text.casefold(),
                f"required text {'found' if expected.casefold() in text.casefold() else 'missing'}",
            )
        for forbidden in spec.forbidden_text:
            add(
                f"forbidden_text:{forbidden}",
                forbidden.casefold() not in text.casefold(),
                f"forbidden text {'absent' if forbidden.casefold() not in text.casefold() else 'present'}",
            )
        verified = provider_ok and bool(final_url) and bool(checks) and all(item["ok"] for item in checks)
        output = {
            "verified": verified,
            "provider_ok": provider_ok,
            "requested_url": url,
            "final_url": final_url,
            "title": title,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
            "screenshot_sha256": response.get("screenshot_sha256"),
            "checks": checks,
            "provider": self.provider.name,
            "provider_receipt": response.get("receipt"),
        }
        receipt_id = self._store_direct_receipt(
            "browser", "browser.navigate", request_hash, output, verified,
            None if verified else "Browser verification failed"
        )
        output["receipt_id"] = receipt_id
        return output

    # Windows ---------------------------------------------------------------

    def register_windows_app(
        self,
        app_id: str,
        *,
        display_name: str,
        executable_hint: str,
        allowed_argument_patterns: list[str] | None = None,
        actor: str = "user",
    ) -> dict[str, Any]:
        normalized = app_id.strip().casefold()
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,63}", normalized):
            raise ValueError("app_id must be a stable lowercase identifier")
        patterns = allowed_argument_patterns or []
        for pattern in patterns:
            re.compile(pattern)
        payload = {
            "app_id": normalized,
            "display_name": display_name.strip(),
            "executable_hint": executable_hint.strip(),
            "allowed_argument_patterns": patterns,
        }
        manifest_hash = sha256_json(payload)
        now = utcnow()
        self.ledger.db.conn.execute(
            """INSERT INTO windows_app_registry
               (app_id, display_name, executable_hint, allowed_argument_patterns_json,
                manifest_hash, active, created_by, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
               ON CONFLICT(app_id) DO UPDATE SET
                 display_name=excluded.display_name,
                 executable_hint=excluded.executable_hint,
                 allowed_argument_patterns_json=excluded.allowed_argument_patterns_json,
                 manifest_hash=excluded.manifest_hash,
                 active=1,
                 created_by=excluded.created_by,
                 updated_at=excluded.updated_at""",
            (
                normalized,
                payload["display_name"],
                payload["executable_hint"],
                stable_json(patterns),
                manifest_hash,
                actor,
                now,
                now,
            ),
        )
        self.ledger.db.conn.commit()
        return self.get_windows_app(normalized)

    def launch_windows_app(self, app_id: str, arguments: list[str] | None = None) -> dict[str, Any]:
        if self.provider is None:
            raise RuntimeError("No capability provider is bound")
        app = self.get_windows_app(app_id)
        manifest_payload = {
            "app_id": app["app_id"],
            "display_name": app["display_name"],
            "executable_hint": app["executable_hint"],
            "allowed_argument_patterns": app["allowed_argument_patterns"],
        }
        if sha256_json(manifest_payload) != app["manifest_hash"]:
            raise ValueError("Windows app allowlist manifest integrity failed")
        if not app["active"]:
            raise PermissionError("Windows application is disabled")
        args = [str(v) for v in (arguments or [])]
        patterns = app["allowed_argument_patterns"]
        if args and not patterns:
            raise PermissionError("This application has no approved argument patterns")
        for value in args:
            if not any(re.fullmatch(pattern, value) for pattern in patterns):
                raise PermissionError(f"Argument is not allowlisted: {value}")
        payload = {
            "app_id": app["app_id"],
            "manifest_hash": app["manifest_hash"],
            "arguments": args,
        }
        request_hash = sha256_json(payload)
        response = self.provider.invoke(
            "windows.launch_app", payload, idempotency_key=f"windows:{request_hash}"
        )
        verified = response.get("ok") is True and response.get("app_id") == app["app_id"] and bool(response.get("pid"))
        output = {
            "verified": verified,
            "app_id": app["app_id"],
            "pid": response.get("pid"),
            "provider": self.provider.name,
            "provider_receipt": response.get("receipt"),
        }
        receipt_id = self._store_direct_receipt(
            "windows", "windows.launch_app", request_hash, output, verified,
            None if verified else "Windows launch verification failed"
        )
        output["receipt_id"] = receipt_id
        return output

    # Agent handlers --------------------------------------------------------

    def _email_draft_skill(self, args: dict[str, Any]) -> SkillResult:
        try:
            draft = self.create_email_draft(
                to=args["to"],
                subject=str(args["subject"]),
                body=str(args["body"]),
                cc=args.get("cc"),
                bcc=args.get("bcc"),
                actor="computer_agent",
            )
            return SkillResult(True, {"draft_id": draft["id"], "payload_hash": draft["payload_hash"], "status": draft["status"]})
        except Exception as exc:
            return SkillResult(False, error=f"{type(exc).__name__}: {exc}")

    def _calendar_draft_skill(self, args: dict[str, Any]) -> SkillResult:
        try:
            draft = self.create_calendar_draft(
                title=str(args["title"]),
                start=str(args["start"]),
                end=str(args["end"]),
                timezone_name=str(args["timezone"]),
                attendees=list(args.get("attendees", [])),
                location=args.get("location"),
                description=args.get("description"),
                actor="computer_agent",
            )
            return SkillResult(True, {"draft_id": draft["id"], "payload_hash": draft["payload_hash"], "status": draft["status"]})
        except Exception as exc:
            return SkillResult(False, error=f"{type(exc).__name__}: {exc}")

    def _browser_skill(self, args: dict[str, Any]) -> SkillResult:
        try:
            output = self.navigate_verified(str(args["url"]), dict(args["verification"]))
            return SkillResult(bool(output["verified"]), output, None if output["verified"] else "Browser verification failed")
        except Exception as exc:
            return SkillResult(False, error=f"{type(exc).__name__}: {exc}")

    def _windows_launch_skill(self, args: dict[str, Any]) -> SkillResult:
        try:
            output = self.launch_windows_app(str(args["app_id"]), list(args.get("arguments", [])))
            return SkillResult(bool(output["verified"]), output, None if output["verified"] else "Application launch verification failed")
        except Exception as exc:
            return SkillResult(False, error=f"{type(exc).__name__}: {exc}")

    # Queries and integrity -------------------------------------------------

    def get_draft(self, draft_id: str) -> dict[str, Any]:
        row = self._draft_row(draft_id)
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        result["metadata"] = json.loads(result.pop("metadata_json"))
        result["integrity_valid"] = sha256_json(result["payload"]) == result["payload_hash"]
        result["approvals"] = [dict(item) for item in self.ledger.db.conn.execute(
            "SELECT * FROM integration_approvals WHERE draft_id = ? ORDER BY created_at", (draft_id,)
        ).fetchall()]
        result["receipts"] = [self.get_receipt(item["id"]) for item in self.ledger.db.conn.execute(
            "SELECT id FROM integration_receipts WHERE draft_id = ? ORDER BY created_at", (draft_id,)
        ).fetchall()]
        return result

    def list_drafts(self) -> list[dict[str, Any]]:
        return [self.get_draft(row["id"]) for row in self.ledger.db.conn.execute(
            "SELECT id FROM integration_drafts ORDER BY created_at DESC"
        ).fetchall()]

    def get_approval(self, approval_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM integration_approvals WHERE id = ?", (approval_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown integration approval: {approval_id}")
        return dict(row)

    def get_receipt(self, receipt_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM integration_receipts WHERE id = ?", (receipt_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown integration receipt: {receipt_id}")
        result = dict(row)
        result["response"] = json.loads(result.pop("response_json"))
        return result

    def verify_receipt(self, receipt_id: str) -> bool:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM integration_receipts WHERE id = ?", (receipt_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown integration receipt: {receipt_id}")
        if row["draft_id"]:
            draft = self._draft_row(row["draft_id"])
            payload = {
                "draft_id": row["draft_id"],
                "kind": row["action_kind"],
                "payload_hash": draft["payload_hash"],
                "provider": row["provider"],
                "capability": row["capability"],
                "idempotency_key": f"draft:{row['draft_id']}:{draft['payload_hash']}",
                "ok": bool(row["ok"]),
                "response": json.loads(row["response_json"]),
                "error": row["error"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
            }
        else:
            payload = {
                "draft_id": None,
                "kind": row["action_kind"],
                "request_hash": row["request_hash"],
                "provider": row["provider"],
                "capability": row["capability"],
                "ok": bool(row["ok"]),
                "response": json.loads(row["response_json"]),
                "error": row["error"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
            }
        return sha256_json(payload) == row["receipt_hash"]

    def get_windows_app(self, app_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM windows_app_registry WHERE app_id = ?", (app_id.strip().casefold(),)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown Windows application: {app_id}")
        result = dict(row)
        result["active"] = bool(result["active"])
        result["allowed_argument_patterns"] = json.loads(result.pop("allowed_argument_patterns_json"))
        return result

    def status(self) -> dict[str, Any]:
        provider_name = self.provider.name if self.provider else None
        return {
            "provider": provider_name,
            "provider_name": provider_name,
            "provider_bound": self.provider is not None,
            "drafts": self.ledger.db.conn.execute(
                "SELECT COUNT(*) AS n FROM integration_drafts"
            ).fetchone()["n"],
            "pending_draft_approvals": self.ledger.db.conn.execute(
                "SELECT COUNT(*) AS n FROM integration_approvals WHERE status = 'pending'"
            ).fetchone()["n"],
            "receipts": self.ledger.db.conn.execute(
                "SELECT COUNT(*) AS n FROM integration_receipts"
            ).fetchone()["n"],
            "windows_apps": self.ledger.db.conn.execute(
                "SELECT COUNT(*) AS n FROM windows_app_registry WHERE active = 1"
            ).fetchone()["n"],
            "agent_skills": [
                item for item in self.ledger.agent.registry.list_contracts()
                if item["name"].startswith(("integration.", "external.browser", "external.windows"))
            ],
        }

    def _store_direct_receipt(
        self,
        action_kind: str,
        capability: str,
        request_hash: str,
        response: dict[str, Any],
        ok: bool,
        error: str | None,
    ) -> str:
        started = completed = utcnow()
        receipt_id = new_id("irc")
        payload = {
            "draft_id": None,
            "kind": action_kind,
            "request_hash": request_hash,
            "provider": self.provider.name if self.provider else "unbound",
            "capability": capability,
            "ok": ok,
            "response": response,
            "error": error,
            "started_at": started,
            "completed_at": completed,
        }
        receipt_hash = sha256_json(payload)
        self.ledger.db.conn.execute(
            """INSERT INTO integration_receipts
               (id, draft_id, action_kind, provider, capability, request_hash,
                response_json, ok, error, receipt_hash, started_at, completed_at, created_at)
               VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                receipt_id,
                action_kind,
                payload["provider"],
                capability,
                request_hash,
                stable_json(response),
                int(ok),
                error,
                receipt_hash,
                started,
                completed,
                completed,
            ),
        )
        self.ledger.db.conn.commit()
        return receipt_id

    @staticmethod
    def _addresses(value: str | list[str] | None) -> list[str]:
        if value is None:
            return []
        values = [value] if isinstance(value, str) else list(value)
        result: list[str] = []
        for item in values:
            for address in str(item).split(","):
                clean = address.strip()
                if not clean:
                    continue
                if "@" not in clean or clean.startswith("@") or clean.endswith("@"):
                    raise ValueError(f"Invalid email address: {clean}")
                result.append(clean)
        return list(dict.fromkeys(result))

    @staticmethod
    def _require_draft_integrity(row) -> None:
        if sha256_json(json.loads(row["payload_json"])) != row["payload_hash"]:
            raise ValueError("Draft payload integrity verification failed")

    def _draft_row(self, draft_id: str):
        row = self.ledger.db.conn.execute(
            "SELECT * FROM integration_drafts WHERE id = ?", (draft_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown integration draft: {draft_id}")
        return row


class ScheduledTaskRuntime:
    """SQLite-backed scheduler with leases and restart-safe plan continuation."""

    def __init__(self, ledger: "EpistemicLedger") -> None:
        self.ledger = ledger

    def create_once(
        self,
        *,
        name: str,
        run_at: str | datetime,
        goal_utterance: str,
        workspace: str | Path | None = None,
        autonomy_mode: AutonomyMode | str = AutonomyMode.VERIFIED,
        actor: str = "user",
    ) -> dict[str, Any]:
        return self._create(
            name=name,
            kind=ScheduleKind.ONCE,
            next_run_at=parse_time(run_at),
            interval_seconds=None,
            goal_utterance=goal_utterance,
            workspace=workspace,
            autonomy_mode=autonomy_mode,
            max_runs=1,
            actor=actor,
        )

    def create_interval(
        self,
        *,
        name: str,
        every_seconds: int,
        goal_utterance: str,
        first_run_at: str | datetime | None = None,
        workspace: str | Path | None = None,
        autonomy_mode: AutonomyMode | str = AutonomyMode.VERIFIED,
        max_runs: int | None = None,
        actor: str = "user",
    ) -> dict[str, Any]:
        if every_seconds < 60:
            raise ValueError("Scheduled intervals must be at least 60 seconds")
        next_run = parse_time(first_run_at or datetime.now(timezone.utc))
        return self._create(
            name=name,
            kind=ScheduleKind.INTERVAL,
            next_run_at=next_run,
            interval_seconds=every_seconds,
            goal_utterance=goal_utterance,
            workspace=workspace,
            autonomy_mode=autonomy_mode,
            max_runs=max_runs,
            actor=actor,
        )

    def _create(
        self,
        *,
        name: str,
        kind: ScheduleKind,
        next_run_at: datetime,
        interval_seconds: int | None,
        goal_utterance: str,
        workspace: str | Path | None,
        autonomy_mode: AutonomyMode | str,
        max_runs: int | None,
        actor: str,
    ) -> dict[str, Any]:
        if not name.strip() or not goal_utterance.strip():
            raise ValueError("Schedule name and goal are required")
        mode = AutonomyMode(autonomy_mode)
        schedule_id = new_id("sch")
        now = utcnow()
        payload = {
            "name": name.strip(),
            "kind": kind.value,
            "goal_utterance": goal_utterance.strip(),
            "workspace": str(Path(workspace).resolve()) if workspace else None,
            "autonomy_mode": mode.value,
            "next_run_at": next_run_at.isoformat(),
            "interval_seconds": interval_seconds,
            "max_runs": max_runs,
        }
        schedule_hash = sha256_json(payload)
        self.ledger.db.conn.execute(
            """INSERT INTO scheduled_jobs
               (id, name, schedule_kind, goal_utterance, workspace, autonomy_mode,
                next_run_at, interval_seconds, max_runs, run_count, status,
                schedule_hash, created_by, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)""",
            (
                schedule_id,
                payload["name"],
                kind.value,
                payload["goal_utterance"],
                payload["workspace"],
                mode.value,
                payload["next_run_at"],
                interval_seconds,
                max_runs,
                ScheduleStatus.ACTIVE.value,
                schedule_hash,
                actor,
                now,
                now,
            ),
        )
        self.ledger._event(
            "scheduled_job",
            schedule_id,
            "SCHEDULE_CREATED",
            {"schedule_hash": schedule_hash, **payload},
            actor,
            ActorRole.HUMAN,
        )
        self.ledger.db.conn.commit()
        return self.get(schedule_id)

    def claim_due(
        self,
        worker_id: str,
        *,
        now: str | datetime | None = None,
        lease_seconds: int = 300,
        exclude_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        if not worker_id.strip():
            raise ValueError("worker_id is required")
        current = parse_time(now or datetime.now(timezone.utc))
        expiry = current + timedelta(seconds=max(30, int(lease_seconds)))
        conn = self.ledger.db.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            excluded = sorted(exclude_ids or set())
            exclusion_sql = ""
            params: list[Any] = [current.isoformat(), current.isoformat()]
            if excluded:
                exclusion_sql = " AND id NOT IN (" + ",".join("?" for _ in excluded) + ")"
                params.extend(excluded)
            row = conn.execute(
                """SELECT * FROM scheduled_jobs
                   WHERE status IN ('active', 'blocked')
                     AND next_run_at <= ?
                     AND (lease_expires_at IS NULL OR lease_expires_at < ?)"""
                + exclusion_sql
                + " ORDER BY next_run_at, created_at LIMIT 1",
                tuple(params),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            conn.execute(
                """UPDATE scheduled_jobs
                   SET lease_owner = ?, lease_expires_at = ?, updated_at = ?
                   WHERE id = ?""",
                (worker_id, expiry.isoformat(), current.isoformat(), row["id"]),
            )
            conn.commit()
            return self.get(row["id"])
        except Exception:
            conn.rollback()
            raise

    def run_claimed(self, schedule_id: str, worker_id: str) -> dict[str, Any]:
        schedule = self._row(schedule_id)
        if not self.verify_schedule(schedule_id):
            raise ValueError("Schedule integrity verification failed")
        if schedule["lease_owner"] != worker_id:
            raise PermissionError("Worker does not own the schedule lease")
        if schedule["lease_expires_at"] and parse_time(schedule["lease_expires_at"]) < datetime.now(timezone.utc):
            raise PermissionError("Schedule lease has expired")
        workspace = Path(schedule["workspace"]) if schedule["workspace"] else self.ledger.agent.boundary.root
        runtime = ComputerAgentRuntime(self.ledger, workspace)
        self.ledger.integrations.install_agent_contracts(runtime)
        runtime.recover_interrupted()
        active_run = None
        if schedule["active_run_id"]:
            active_run = self.ledger.db.conn.execute(
                "SELECT * FROM scheduled_job_runs WHERE id = ?", (schedule["active_run_id"],)
            ).fetchone()
        if active_run is None:
            run_id = new_id("sjr")
            goal = runtime.create_goal(
                schedule["goal_utterance"],
                autonomy_mode=schedule["autonomy_mode"],
                actor="scheduler",
            )
            plan = runtime.plan_goal(goal["id"])
            now = utcnow()
            self.ledger.db.conn.execute(
                """INSERT INTO scheduled_job_runs
                   (id, schedule_id, scheduled_for, goal_id, plan_id, status,
                    started_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)""",
                (run_id, schedule_id, schedule["next_run_at"], goal["id"], plan["id"], now, now, now),
            )
            self.ledger.db.conn.execute(
                "UPDATE scheduled_jobs SET active_run_id = ?, updated_at = ? WHERE id = ?",
                (run_id, now, schedule_id),
            )
            self.ledger.db.conn.commit()
            active_run = self.ledger.db.conn.execute(
                "SELECT * FROM scheduled_job_runs WHERE id = ?", (run_id,)
            ).fetchone()
        plan = runtime.run_until_blocked(active_run["plan_id"])
        plan_status = plan["status"]
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        if plan_status == "succeeded":
            run_status = "succeeded"
            run_count = int(schedule["run_count"]) + 1
            completed = now
            if schedule["schedule_kind"] == ScheduleKind.ONCE.value or (
                schedule["max_runs"] is not None and run_count >= int(schedule["max_runs"])
            ):
                schedule_status = ScheduleStatus.COMPLETED.value
                next_run = schedule["next_run_at"]
            else:
                schedule_status = ScheduleStatus.ACTIVE.value
                next_run = (parse_time(schedule["next_run_at"]) + timedelta(seconds=int(schedule["interval_seconds"]))).isoformat()
            active_run_id = None
        elif plan_status in {"waiting_approval", "waiting_input", "needs_configuration", "blocked"}:
            run_status = plan_status
            run_count = int(schedule["run_count"])
            completed = None
            schedule_status = ScheduleStatus.BLOCKED.value
            next_run = schedule["next_run_at"]
            active_run_id = active_run["id"]
        else:
            run_status = "failed"
            run_count = int(schedule["run_count"])
            completed = now
            schedule_status = ScheduleStatus.FAILED.value
            next_run = schedule["next_run_at"]
            active_run_id = None
        run_result = {"plan_status": plan_status}
        run_hash = sha256_json({
            "schedule_id": schedule_id,
            "scheduled_for": active_run["scheduled_for"],
            "goal_id": active_run["goal_id"],
            "plan_id": active_run["plan_id"],
            "status": run_status,
            "result": run_result,
            "error": None if run_status != "failed" else "Scheduled plan failed",
            "started_at": active_run["started_at"],
            "completed_at": completed,
        })
        self.ledger.db.conn.execute(
            """UPDATE scheduled_job_runs
               SET status = ?, completed_at = ?, updated_at = ?, result_json = ?,
                   error = ?, run_hash = ? WHERE id = ?""",
            (
                run_status,
                completed,
                now,
                stable_json(run_result),
                None if run_status != "failed" else "Scheduled plan failed",
                run_hash,
                active_run["id"],
            ),
        )
        self.ledger.db.conn.execute(
            """UPDATE scheduled_jobs
               SET status = ?, next_run_at = ?, run_count = ?, active_run_id = ?,
                   lease_owner = NULL, lease_expires_at = NULL, last_error = ?, updated_at = ?
               WHERE id = ?""",
            (
                schedule_status,
                next_run,
                run_count,
                active_run_id,
                None if run_status != "failed" else "Scheduled plan failed",
                now,
                schedule_id,
            ),
        )
        self.ledger._event(
            "scheduled_job",
            schedule_id,
            "SCHEDULE_RUN_UPDATED",
            {"run_id": active_run["id"], "run_status": run_status, "plan_status": plan_status},
            worker_id,
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()
        return {"schedule": self.get(schedule_id), "run": self.get_run(active_run["id"]), "plan": plan}

    def tick(
        self,
        worker_id: str,
        *,
        max_jobs: int = 10,
        now: str | datetime | None = None,
    ) -> list[dict[str, Any]]:
        self.recover_expired_leases(now=now)
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _ in range(max(1, max_jobs)):
            claimed = self.claim_due(worker_id, now=now, exclude_ids=seen)
            if claimed is None:
                break
            seen.add(claimed["id"])
            results.append(self.run_claimed(claimed["id"], worker_id))
        return results

    def resume(self, schedule_id: str, worker_id: str, *, lease_seconds: int = 300) -> dict[str, Any]:
        schedule = self._row(schedule_id)
        if schedule["status"] not in {ScheduleStatus.BLOCKED.value, ScheduleStatus.ACTIVE.value}:
            raise ValueError("Schedule is not resumable")
        current = datetime.now(timezone.utc)
        self.ledger.db.conn.execute(
            """UPDATE scheduled_jobs
               SET lease_owner = ?, lease_expires_at = ?, updated_at = ? WHERE id = ?""",
            (worker_id, (current + timedelta(seconds=lease_seconds)).isoformat(), current.isoformat(), schedule_id),
        )
        self.ledger.db.conn.commit()
        return self.run_claimed(schedule_id, worker_id)

    def recover_expired_leases(self, *, now: str | datetime | None = None) -> int:
        current = parse_time(now or datetime.now(timezone.utc)).isoformat()
        cursor = self.ledger.db.conn.execute(
            """UPDATE scheduled_jobs
               SET lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
               WHERE lease_expires_at IS NOT NULL AND lease_expires_at < ?""",
            (current, current),
        )
        self.ledger.db.conn.commit()
        return cursor.rowcount

    def pause(self, schedule_id: str) -> dict[str, Any]:
        self._set_status(schedule_id, ScheduleStatus.PAUSED)
        return self.get(schedule_id)

    def activate(self, schedule_id: str) -> dict[str, Any]:
        row = self._row(schedule_id)
        if row["status"] not in {ScheduleStatus.PAUSED.value, ScheduleStatus.FAILED.value}:
            raise ValueError("Only paused or failed schedules can be activated")
        self._set_status(schedule_id, ScheduleStatus.ACTIVE)
        return self.get(schedule_id)

    def cancel(self, schedule_id: str) -> dict[str, Any]:
        self._set_status(schedule_id, ScheduleStatus.CANCELLED)
        return self.get(schedule_id)

    def get(self, schedule_id: str) -> dict[str, Any]:
        row = self._row(schedule_id)
        result = dict(row)
        result["runs"] = [self.get_run(item["id"]) for item in self.ledger.db.conn.execute(
            "SELECT id FROM scheduled_job_runs WHERE schedule_id = ? ORDER BY created_at", (schedule_id,)
        ).fetchall()]
        return result

    def list(self) -> list[dict[str, Any]]:
        return [self.get(row["id"]) for row in self.ledger.db.conn.execute(
            "SELECT id FROM scheduled_jobs ORDER BY created_at DESC"
        ).fetchall()]

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM scheduled_job_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown scheduled run: {run_id}")
        result = dict(row)
        result["result"] = json.loads(result.pop("result_json"))
        result["integrity_valid"] = self.verify_run(run_id) if result.get("run_hash") else None
        return result

    def verify_run(self, run_id: str) -> bool:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM scheduled_job_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown scheduled run: {run_id}")
        if not row["run_hash"]:
            return False
        payload = {
            "schedule_id": row["schedule_id"],
            "scheduled_for": row["scheduled_for"],
            "goal_id": row["goal_id"],
            "plan_id": row["plan_id"],
            "status": row["status"],
            "result": json.loads(row["result_json"]),
            "error": row["error"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        }
        return sha256_json(payload) == row["run_hash"]

    def verify_schedule(self, schedule_id: str) -> bool:
        row = self._row(schedule_id)
        payload = {
            "name": row["name"],
            "kind": row["schedule_kind"],
            "goal_utterance": row["goal_utterance"],
            "workspace": row["workspace"],
            "autonomy_mode": row["autonomy_mode"],
            "next_run_at": row["next_run_at"] if int(row["run_count"]) == 0 else self._initial_next_run(schedule_id),
            "interval_seconds": row["interval_seconds"],
            "max_runs": row["max_runs"],
        }
        return sha256_json(payload) == row["schedule_hash"]

    def _initial_next_run(self, schedule_id: str) -> str:
        event = self.ledger.db.conn.execute(
            """SELECT payload_json FROM events
               WHERE entity_type = 'scheduled_job' AND entity_id = ? AND event_type = 'SCHEDULE_CREATED'
               ORDER BY id LIMIT 1""",
            (schedule_id,),
        ).fetchone()
        if event is None:
            raise ValueError("Schedule creation event is missing")
        return json.loads(event["payload_json"])["next_run_at"]

    def _set_status(self, schedule_id: str, status: ScheduleStatus) -> None:
        self._row(schedule_id)
        self.ledger.db.conn.execute(
            """UPDATE scheduled_jobs SET status = ?, lease_owner = NULL,
               lease_expires_at = NULL, updated_at = ? WHERE id = ?""",
            (status.value, utcnow(), schedule_id),
        )
        self.ledger.db.conn.commit()

    def _row(self, schedule_id: str):
        row = self.ledger.db.conn.execute(
            "SELECT * FROM scheduled_jobs WHERE id = ?", (schedule_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown schedule: {schedule_id}")
        return row
