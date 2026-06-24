from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .agent_os import (
    AutonomyMode,
    ComputerGoalSpec,
    ComputerGoalStatus,
    ComputerGoalType,
    ComputerPlanSpec,
    ComputerStepSpec,
    ComputerStepStatus,
    ComputerAgentRuntime,
    SkillContract,
)
from .models import ActorRole, RiskLevel

if TYPE_CHECKING:  # pragma: no cover
    from .integrations import CapabilityProvider
    from .ledger import EpistemicLedger


ADAPTIVE_API_VERSION = "1.5"
MAX_DESKTOP_ELEMENTS = 5000
MAX_SCREENSHOT_BYTES = 2 * 1024 * 1024
MAX_TEXT_FIELD_CHARS = 10_000


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


class DesktopActionKind(StrEnum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    TYPE_TEXT = "type_text"
    HOTKEY = "hotkey"
    WAIT = "wait"


class AdaptiveWorkflowStatus(StrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    REJECTED = "rejected"
    STALE = "stale"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class DesktopElement:
    element_id: str
    role: str
    name: str
    automation_id: str | None
    bounds: tuple[int, int, int, int]
    visible: bool = True
    enabled: bool = True
    value: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stable_key(self) -> str:
        if self.automation_id:
            return f"automation:{self.automation_id}"
        x, y, w, h = self.bounds
        return f"semantic:{self.role.casefold()}:{self.name.casefold()}:{x}:{y}:{w}:{h}"

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "role": self.role,
            "name": self.name,
            "automation_id": self.automation_id,
            "bounds": {
                "x": self.bounds[0],
                "y": self.bounds[1],
                "width": self.bounds[2],
                "height": self.bounds[3],
            },
            "visible": self.visible,
            "enabled": self.enabled,
            "value": self.value,
            "metadata": self.metadata,
            "stable_key": self.stable_key,
        }


@dataclass(frozen=True, slots=True)
class DesktopSelector:
    element_id: str | None = None
    automation_id: str | None = None
    role: str | None = None
    name: str | None = None
    index: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DesktopSelector":
        if not isinstance(value, dict):
            raise TypeError("Desktop selector must be an object")
        selector = cls(
            element_id=_clean_optional(value.get("element_id")),
            automation_id=_clean_optional(value.get("automation_id")),
            role=_clean_optional(value.get("role")),
            name=_clean_optional(value.get("name")),
            index=int(value["index"]) if value.get("index") is not None else None,
        )
        if not any((selector.element_id, selector.automation_id, selector.role, selector.name)):
            raise ValueError("Desktop selector requires element_id, automation_id, role, or name")
        if selector.index is not None and selector.index < 0:
            raise ValueError("Desktop selector index cannot be negative")
        return selector

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DesktopExpectation:
    required_text: tuple[str, ...] = ()
    forbidden_text: tuple[str, ...] = ()
    active_app: str | None = None
    window_title_contains: str | None = None
    element_present: DesktopSelector | None = None
    element_absent: DesktopSelector | None = None
    value_selector: DesktopSelector | None = None
    value_equals: str | None = None
    screenshot_changed: bool | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "DesktopExpectation":
        raw = value or {}
        if not isinstance(raw, dict):
            raise TypeError("Desktop expectation must be an object")
        result = cls(
            required_text=tuple(_bounded_strings(raw.get("required_text", []), 100, 500)),
            forbidden_text=tuple(_bounded_strings(raw.get("forbidden_text", []), 100, 500)),
            active_app=_clean_optional(raw.get("active_app"), 300),
            window_title_contains=_clean_optional(raw.get("window_title_contains"), 500),
            element_present=DesktopSelector.from_dict(raw["element_present"])
            if raw.get("element_present") is not None
            else None,
            element_absent=DesktopSelector.from_dict(raw["element_absent"])
            if raw.get("element_absent") is not None
            else None,
            value_selector=DesktopSelector.from_dict(raw["value_selector"])
            if raw.get("value_selector") is not None
            else None,
            value_equals=str(raw["value_equals"])
            if raw.get("value_equals") is not None
            else None,
            screenshot_changed=bool(raw["screenshot_changed"])
            if raw.get("screenshot_changed") is not None
            else None,
        )
        if result.value_equals is not None and result.value_selector is None:
            raise ValueError("value_equals requires value_selector")
        if not any(
            (
                result.required_text,
                result.forbidden_text,
                result.active_app,
                result.window_title_contains,
                result.element_present,
                result.element_absent,
                result.value_selector,
                result.screenshot_changed is not None,
            )
        ):
            raise ValueError("A desktop action needs at least one postcondition")
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "required_text": list(self.required_text),
            "forbidden_text": list(self.forbidden_text),
            "active_app": self.active_app,
            "window_title_contains": self.window_title_contains,
            "element_present": self.element_present.as_dict() if self.element_present else None,
            "element_absent": self.element_absent.as_dict() if self.element_absent else None,
            "value_selector": self.value_selector.as_dict() if self.value_selector else None,
            "value_equals": self.value_equals,
            "screenshot_changed": self.screenshot_changed,
        }


@dataclass(frozen=True, slots=True)
class DesktopActionSpec:
    observation_id: str
    kind: DesktopActionKind
    selector: DesktopSelector | None
    text: str | None
    keys: tuple[str, ...]
    expectation: DesktopExpectation
    risk: RiskLevel = RiskLevel.MEDIUM
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DesktopActionSpec":
        if not isinstance(value, dict):
            raise TypeError("Desktop action must be an object")
        kind = DesktopActionKind(value["kind"])
        selector = (
            DesktopSelector.from_dict(value["selector"])
            if value.get("selector") is not None
            else None
        )
        text = str(value["text"]) if value.get("text") is not None else None
        keys = tuple(_bounded_strings(value.get("keys", []), 12, 50))
        if kind in {DesktopActionKind.CLICK, DesktopActionKind.DOUBLE_CLICK, DesktopActionKind.TYPE_TEXT} and selector is None:
            raise ValueError(f"{kind.value} requires a target selector")
        if kind == DesktopActionKind.TYPE_TEXT:
            if text is None:
                raise ValueError("type_text requires text")
            if len(text) > MAX_TEXT_FIELD_CHARS:
                raise ValueError("Desktop text input exceeds 10,000 characters")
        if kind == DesktopActionKind.HOTKEY and not keys:
            raise ValueError("hotkey requires keys")
        if kind == DesktopActionKind.WAIT and (selector is not None or text is not None or keys):
            raise ValueError("wait actions cannot include selector, text, or keys")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Desktop action metadata must be an object")
        return cls(
            observation_id=str(value["observation_id"]),
            kind=kind,
            selector=selector,
            text=text,
            keys=keys,
            expectation=DesktopExpectation.from_dict(value.get("expectation")),
            risk=RiskLevel(value.get("risk", RiskLevel.MEDIUM.value)),
            metadata=metadata,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "kind": self.kind.value,
            "selector": self.selector.as_dict() if self.selector else None,
            "text": self.text,
            "keys": list(self.keys),
            "expectation": self.expectation.as_dict(),
            "risk": self.risk.value,
            "metadata": self.metadata,
        }


def _clean_optional(value: Any, maximum: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > maximum:
        raise ValueError(f"Text field exceeds {maximum} characters")
    return text


def _bounded_strings(value: Any, maximum_items: int, maximum_length: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("Expected a list of strings")
    if len(value) > maximum_items:
        raise ValueError(f"List exceeds {maximum_items} items")
    result: list[str] = []
    for item in value:
        text = str(item)
        if len(text) > maximum_length:
            raise ValueError(f"Text item exceeds {maximum_length} characters")
        result.append(text)
    return result


class AdaptiveSkillRuntime:
    """Verified desktop perception and declarative workflow learning.

    This runtime deliberately does not learn executable Python or shell code.
    A learned skill is a reviewed, hash-bound composition of already registered
    capability contracts. Desktop actions are tied to exact observations and
    postconditions, so stale or ambiguous interfaces fail closed.
    """

    def __init__(
        self,
        ledger: "EpistemicLedger",
        provider: "CapabilityProvider | None" = None,
        workspace: str | Path | None = None,
    ) -> None:
        self.ledger = ledger
        self.provider = provider
        self.workspace = Path(workspace or (ledger.db.path.parent / "adaptive_workspace")).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.artifact_root = self.workspace / "desktop_artifacts"
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def bind_provider(self, provider: "CapabilityProvider") -> None:
        self.provider = provider

    def status(self) -> dict[str, Any]:
        observations = self.ledger.db.conn.execute(
            "SELECT COUNT(*) AS n FROM desktop_observations"
        ).fetchone()["n"]
        workflows = self.ledger.db.conn.execute(
            "SELECT status, COUNT(*) AS n FROM adaptive_workflows GROUP BY status"
        ).fetchall()
        return {
            "api_version": ADAPTIVE_API_VERSION,
            "provider": getattr(self.provider, "name", None),
            "workspace": str(self.workspace),
            "desktop_observations": int(observations),
            "workflows": {row["status"]: int(row["n"]) for row in workflows},
        }

    # ------------------------------------------------------------------
    # Desktop perception
    # ------------------------------------------------------------------
    def capture_desktop(
        self,
        *,
        label: str = "desktop observation",
        include_screenshot: bool = True,
        raw_response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if raw_response is None:
            if self.provider is None:
                raise RuntimeError("No desktop capability provider is bound")
            request = {
                "include_accessibility_tree": True,
                "include_screenshot": bool(include_screenshot),
                "max_elements": MAX_DESKTOP_ELEMENTS,
            }
            response = self.provider.invoke(
                "desktop.observe",
                request,
                idempotency_key=f"desktop-observe:{sha256_json(request)}:{uuid.uuid4().hex}",
            )
        else:
            response = dict(raw_response)
        normalized = self._normalize_observation_response(response, label)
        observation_id = new_id("dob")
        created = utcnow()
        screenshot_path: str | None = None
        screenshot_hash: str | None = normalized.pop("_screenshot_hash")
        screenshot_bytes: bytes | None = normalized.pop("_screenshot_bytes")
        if screenshot_bytes is not None:
            target = self.artifact_root / f"{observation_id}.png"
            target.write_bytes(screenshot_bytes)
            screenshot_path = str(target.relative_to(self.workspace))
        payload = {
            "api_version": ADAPTIVE_API_VERSION,
            "label": label,
            "provider": normalized["provider"],
            "active_app": normalized["active_app"],
            "window_title": normalized["window_title"],
            "screen": normalized["screen"],
            "elements": normalized["elements"],
            "accessibility_fingerprint": normalized["accessibility_fingerprint"],
            "visual_fingerprint": screenshot_hash,
            "state_fingerprint": normalized["state_fingerprint"],
            "provider_receipt": normalized.get("provider_receipt"),
        }
        observation_hash = sha256_json(payload)
        self.ledger.db.conn.execute(
            """INSERT INTO desktop_observations
               (id, label, provider, active_app, window_title, screen_json, elements_json,
                accessibility_fingerprint, visual_fingerprint, state_fingerprint,
                screenshot_path, screenshot_hash, payload_json, observation_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                observation_id,
                label,
                normalized["provider"],
                normalized["active_app"],
                normalized["window_title"],
                stable_json(normalized["screen"]),
                stable_json(normalized["elements"]),
                normalized["accessibility_fingerprint"],
                screenshot_hash,
                normalized["state_fingerprint"],
                screenshot_path,
                screenshot_hash,
                stable_json(payload),
                observation_hash,
                created,
            ),
        )
        self.ledger._event(
            "desktop_observation",
            observation_id,
            "DESKTOP_OBSERVATION_CAPTURED",
            {
                "observation_hash": observation_hash,
                "active_app": normalized["active_app"],
                "element_count": len(normalized["elements"]),
                "has_screenshot": screenshot_hash is not None,
            },
            "adaptive_runtime",
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()
        return self.get_observation(observation_id)

    def _normalize_observation_response(
        self, response: dict[str, Any], label: str
    ) -> dict[str, Any]:
        if not isinstance(response, dict):
            raise TypeError("Desktop provider response must be an object")
        if response.get("ok") is not True:
            raise RuntimeError(str(response.get("error") or "Desktop observation failed"))
        active_app = _clean_optional(response.get("active_app"), 300) or "unknown"
        window_title = _clean_optional(response.get("window_title"), 500) or ""
        screen = response.get("screen") or {}
        if not isinstance(screen, dict):
            raise TypeError("screen must be an object")
        width = int(screen.get("width", 0))
        height = int(screen.get("height", 0))
        scale = float(screen.get("scale", 1.0))
        if not (1 <= width <= 32768 and 1 <= height <= 32768 and 0.1 <= scale <= 16.0):
            raise ValueError("Desktop screen dimensions or scale are invalid")
        screen_out = {"width": width, "height": height, "scale": scale}
        raw_elements = response.get("elements", [])
        if not isinstance(raw_elements, list):
            raise TypeError("elements must be a list")
        if len(raw_elements) > MAX_DESKTOP_ELEMENTS:
            raise ValueError(f"Desktop observation exceeds {MAX_DESKTOP_ELEMENTS} elements")
        elements: list[DesktopElement] = []
        seen_ids: set[str] = set()
        for position, raw in enumerate(raw_elements):
            element = self._normalize_element(raw, position, width, height)
            if element.element_id in seen_ids:
                raise ValueError(f"Duplicate desktop element_id: {element.element_id}")
            seen_ids.add(element.element_id)
            elements.append(element)
        element_payload = [element.as_dict() for element in elements]
        accessibility_fingerprint = sha256_json(
            [
                {
                    "stable_key": item["stable_key"],
                    "role": item["role"],
                    "name": item["name"],
                    "value": item["value"],
                    "visible": item["visible"],
                    "enabled": item["enabled"],
                }
                for item in element_payload
            ]
        )
        state_fingerprint = sha256_json(
            {
                "active_app": active_app,
                "window_title": window_title,
                "screen": screen_out,
                "accessibility_fingerprint": accessibility_fingerprint,
            }
        )
        screenshot_bytes: bytes | None = None
        screenshot_hash: str | None = None
        encoded = response.get("screenshot_png_base64")
        if encoded is not None:
            if not isinstance(encoded, str):
                raise TypeError("screenshot_png_base64 must be a string")
            try:
                screenshot_bytes = base64.b64decode(encoded, validate=True)
            except Exception as exc:
                raise ValueError("Desktop screenshot is not valid base64") from exc
            if len(screenshot_bytes) > MAX_SCREENSHOT_BYTES:
                raise ValueError("Desktop screenshot exceeds 2 MiB")
            if not screenshot_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Desktop screenshot must be a PNG")
            screenshot_hash = hashlib.sha256(screenshot_bytes).hexdigest()
        elif response.get("screenshot_sha256") is not None:
            screenshot_hash = str(response["screenshot_sha256"]).lower()
            if not re.fullmatch(r"[0-9a-f]{64}", screenshot_hash):
                raise ValueError("screenshot_sha256 must be a lowercase SHA-256 digest")
        return {
            "provider": _clean_optional(response.get("provider"), 300)
            or getattr(self.provider, "name", "fixture"),
            "label": label,
            "active_app": active_app,
            "window_title": window_title,
            "screen": screen_out,
            "elements": element_payload,
            "accessibility_fingerprint": accessibility_fingerprint,
            "state_fingerprint": state_fingerprint,
            "provider_receipt": response.get("receipt"),
            "_screenshot_bytes": screenshot_bytes,
            "_screenshot_hash": screenshot_hash,
        }

    @staticmethod
    def _normalize_element(raw: Any, position: int, width: int, height: int) -> DesktopElement:
        if not isinstance(raw, dict):
            raise TypeError("Each desktop element must be an object")
        element_id = _clean_optional(raw.get("element_id"), 300) or f"element-{position}"
        role = _clean_optional(raw.get("role"), 200) or "unknown"
        name = _clean_optional(raw.get("name"), 1000) or ""
        automation_id = _clean_optional(raw.get("automation_id"), 500)
        bounds = raw.get("bounds") or {}
        if not isinstance(bounds, dict):
            raise TypeError("Desktop element bounds must be an object")
        x = int(bounds.get("x", 0))
        y = int(bounds.get("y", 0))
        w = int(bounds.get("width", 0))
        h = int(bounds.get("height", 0))
        if w < 0 or h < 0:
            raise ValueError("Desktop element width and height cannot be negative")
        if x < -width or y < -height or x > width * 2 or y > height * 2:
            raise ValueError("Desktop element bounds are implausibly outside the screen")
        value = raw.get("value")
        if value is not None:
            value = str(value)
            if len(value) > MAX_TEXT_FIELD_CHARS:
                value = value[:MAX_TEXT_FIELD_CHARS]
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Desktop element metadata must be an object")
        return DesktopElement(
            element_id=element_id,
            role=role,
            name=name,
            automation_id=automation_id,
            bounds=(x, y, w, h),
            visible=bool(raw.get("visible", True)),
            enabled=bool(raw.get("enabled", True)),
            value=value,
            metadata=metadata,
        )

    def get_observation(self, observation_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM desktop_observations WHERE id = ?", (observation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown desktop observation: {observation_id}")
        result = dict(row)
        result["screen"] = json.loads(result.pop("screen_json"))
        result["elements"] = json.loads(result.pop("elements_json"))
        result["payload"] = json.loads(result.pop("payload_json"))
        result["integrity_valid"] = self.verify_observation(observation_id)
        return result

    def list_observations(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT id FROM desktop_observations ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 1000)),)
        ).fetchall()
        return [self.get_observation(row["id"]) for row in rows]

    def verify_observation(self, observation_id: str) -> bool:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM desktop_observations WHERE id = ?", (observation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown desktop observation: {observation_id}")
        payload = json.loads(row["payload_json"])
        if sha256_json(payload) != row["observation_hash"]:
            return False
        screenshot_path = row["screenshot_path"]
        if screenshot_path:
            path = (self.workspace / screenshot_path).resolve()
            try:
                path.relative_to(self.workspace)
            except ValueError:
                return False
            if not path.is_file() or path.is_symlink():
                return False
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["screenshot_hash"]:
                return False
        return True

    def compare_observations(self, before_id: str, after_id: str) -> dict[str, Any]:
        before = self.get_observation(before_id)
        after = self.get_observation(after_id)
        if not before["integrity_valid"] or not after["integrity_valid"]:
            raise ValueError("Cannot compare a tampered desktop observation")
        before_map = {item["stable_key"]: item for item in before["elements"]}
        after_map = {item["stable_key"]: item for item in after["elements"]}
        added = [after_map[key] for key in sorted(after_map.keys() - before_map.keys())]
        removed = [before_map[key] for key in sorted(before_map.keys() - after_map.keys())]
        changed = []
        for key in sorted(before_map.keys() & after_map.keys()):
            old = before_map[key]
            new = after_map[key]
            fields = {}
            for field_name in ("name", "value", "visible", "enabled", "bounds"):
                if old.get(field_name) != new.get(field_name):
                    fields[field_name] = {"before": old.get(field_name), "after": new.get(field_name)}
            if fields:
                changed.append({"stable_key": key, "changes": fields})
        denominator = max(1, len(before_map | after_map))
        drift_score = min(1.0, (len(added) + len(removed) + len(changed)) / denominator)
        payload = {
            "before_id": before_id,
            "after_id": after_id,
            "application_changed": before["active_app"] != after["active_app"],
            "window_title_changed": before["window_title"] != after["window_title"],
            "visual_changed": before["visual_fingerprint"] != after["visual_fingerprint"],
            "added": added,
            "removed": removed,
            "changed": changed,
            "drift_score": drift_score,
        }
        diff_id = new_id("ddf")
        diff_hash = sha256_json(payload)
        self.ledger.db.conn.execute(
            """INSERT INTO desktop_observation_diffs
               (id, before_id, after_id, diff_json, drift_score, diff_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (diff_id, before_id, after_id, stable_json(payload), drift_score, diff_hash, utcnow()),
        )
        self.ledger.db.conn.commit()
        return {"id": diff_id, "diff_hash": diff_hash, **payload}

    # ------------------------------------------------------------------
    # Desktop actions
    # ------------------------------------------------------------------
    def propose_action(self, spec: DesktopActionSpec | dict[str, Any]) -> dict[str, Any]:
        action = spec if isinstance(spec, DesktopActionSpec) else DesktopActionSpec.from_dict(spec)
        observation = self.get_observation(action.observation_id)
        if not observation["integrity_valid"]:
            raise ValueError("Desktop observation integrity verification failed")
        target = self._resolve_selector(observation, action.selector) if action.selector else None
        action_payload = action.as_dict()
        action_payload["observation_hash"] = observation["observation_hash"]
        action_payload["target"] = target
        action_hash = sha256_json(action_payload)
        action_id = new_id("dac")
        now = utcnow()
        self.ledger.db.conn.execute(
            """INSERT INTO desktop_actions
               (id, observation_id, action_kind, selector_json, target_json, action_json,
                action_hash, risk, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, ?)""",
            (
                action_id,
                action.observation_id,
                action.kind.value,
                stable_json(action.selector.as_dict() if action.selector else {}),
                stable_json(target or {}),
                stable_json(action_payload),
                action_hash,
                action.risk.value,
                now,
                now,
            ),
        )
        self.ledger._event(
            "desktop_action",
            action_id,
            "DESKTOP_ACTION_PROPOSED",
            {"action_hash": action_hash, "kind": action.kind.value, "observation_id": action.observation_id},
            "adaptive_runtime",
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()
        return self.get_action(action_id)

    def request_action_approval(self, action_id: str) -> dict[str, Any]:
        action = self._action_row(action_id)
        self._require_action_integrity(action)
        if action["status"] not in {"proposed", "rejected"}:
            raise ValueError("Only proposed or rejected actions can request approval")
        approval_id = new_id("daa")
        self.ledger.db.conn.execute(
            """INSERT INTO desktop_action_approvals
               (id, action_id, action_hash, status, created_at)
               VALUES (?, ?, ?, 'pending', ?)""",
            (approval_id, action_id, action["action_hash"], utcnow()),
        )
        self.ledger.db.conn.execute(
            "UPDATE desktop_actions SET status = 'pending_approval', updated_at = ? WHERE id = ?",
            (utcnow(), action_id),
        )
        self.ledger.db.conn.commit()
        return self.get_action_approval(approval_id)

    def decide_action_approval(
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
            "SELECT * FROM desktop_action_approvals WHERE id = ?", (approval_id,)
        ).fetchone()
        if row is None or row["status"] != "pending":
            raise ValueError("Desktop action approval not found or already decided")
        action = self._action_row(row["action_id"])
        self._require_action_integrity(action)
        if row["action_hash"] != action["action_hash"]:
            raise ValueError("Approval no longer matches the exact desktop action")
        status = "approved" if approved else "rejected"
        now = utcnow()
        self.ledger.db.conn.execute(
            """UPDATE desktop_action_approvals
               SET status = ?, reviewer = ?, rationale = ?, decided_at = ? WHERE id = ?""",
            (status, reviewer, rationale, now, approval_id),
        )
        self.ledger.db.conn.execute(
            "UPDATE desktop_actions SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, action["id"]),
        )
        self.ledger._event(
            "desktop_action",
            action["id"],
            "DESKTOP_ACTION_APPROVED" if approved else "DESKTOP_ACTION_REJECTED",
            {"approval_id": approval_id, "action_hash": action["action_hash"], "rationale": rationale},
            reviewer,
            ActorRole.HUMAN,
        )
        self.ledger.db.conn.commit()
        return self.get_action_approval(approval_id)

    def execute_action(self, action_id: str) -> dict[str, Any]:
        if self.provider is None:
            raise RuntimeError("No desktop capability provider is bound")
        action = self._action_row(action_id)
        self._require_action_integrity(action)
        if action["status"] == "succeeded":
            receipt = self.ledger.db.conn.execute(
                "SELECT id FROM desktop_action_receipts WHERE action_id = ? ORDER BY created_at DESC LIMIT 1",
                (action_id,),
            ).fetchone()
            return self.get_action_receipt(receipt["id"]) if receipt else self.get_action(action_id)
        approval = self.ledger.db.conn.execute(
            """SELECT * FROM desktop_action_approvals
               WHERE action_id = ? AND action_hash = ? AND status = 'approved' AND consumed_at IS NULL
               ORDER BY created_at DESC LIMIT 1""",
            (action_id, action["action_hash"]),
        ).fetchone()
        if approval is None:
            raise PermissionError("The exact desktop action has not been approved")
        source_observation = self.get_observation(action["observation_id"])
        current = self.capture_desktop(label="pre-action freshness check", include_screenshot=False)
        if current["state_fingerprint"] != source_observation["state_fingerprint"]:
            self.ledger.db.conn.execute(
                "UPDATE desktop_actions SET status = 'stale', updated_at = ? WHERE id = ?",
                (utcnow(), action_id),
            )
            self.ledger.db.conn.commit()
            raise RuntimeError("Desktop state changed after the action was proposed; re-observe and re-approve")
        payload = json.loads(action["action_json"])
        request_payload = {
            "kind": payload["kind"],
            "target": payload.get("target"),
            "text": payload.get("text"),
            "keys": payload.get("keys", []),
            "observation_hash": source_observation["observation_hash"],
            "action_hash": action["action_hash"],
        }
        started = utcnow()
        response: dict[str, Any] = {}
        error: str | None = None
        checks: list[dict[str, Any]] = []
        post_observation: dict[str, Any] | None = None
        try:
            response = self.provider.invoke(
                "desktop.perform_action",
                request_payload,
                idempotency_key=f"desktop-action:{action_id}:{action['action_hash']}",
            )
            if response.get("ok") is not True:
                raise RuntimeError(str(response.get("error") or "Desktop provider rejected the action"))
            raw_post = response.get("post_observation")
            post_observation = self.capture_desktop(
                label=f"post-action {action_id}",
                include_screenshot=True,
                raw_response=raw_post if isinstance(raw_post, dict) else None,
            )
            expectation = DesktopExpectation.from_dict(payload.get("expectation"))
            checks = self._verify_expectation(source_observation, post_observation, expectation)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        ok = error is None and bool(checks) and all(item["ok"] for item in checks)
        completed = utcnow()
        receipt_payload = {
            "action_id": action_id,
            "action_hash": action["action_hash"],
            "provider": self.provider.name,
            "ok": ok,
            "provider_response": response,
            "post_observation_id": post_observation["id"] if post_observation else None,
            "checks": checks,
            "error": error,
            "started_at": started,
            "completed_at": completed,
        }
        receipt_hash = sha256_json(receipt_payload)
        receipt_id = new_id("dar")
        self.ledger.db.conn.execute(
            """INSERT INTO desktop_action_receipts
               (id, action_id, provider, response_json, post_observation_id, checks_json,
                ok, error, receipt_hash, started_at, completed_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                receipt_id,
                action_id,
                self.provider.name,
                stable_json(response),
                post_observation["id"] if post_observation else None,
                stable_json(checks),
                int(ok),
                error,
                receipt_hash,
                started,
                completed,
                completed,
            ),
        )
        self.ledger.db.conn.execute(
            "UPDATE desktop_action_approvals SET consumed_at = ? WHERE id = ?",
            (completed, approval["id"]),
        )
        self.ledger.db.conn.execute(
            "UPDATE desktop_actions SET status = ?, updated_at = ? WHERE id = ?",
            ("succeeded" if ok else "failed", completed, action_id),
        )
        self.ledger._event(
            "desktop_action",
            action_id,
            "DESKTOP_ACTION_SUCCEEDED" if ok else "DESKTOP_ACTION_FAILED",
            {"receipt_id": receipt_id, "receipt_hash": receipt_hash, "error": error},
            "adaptive_runtime",
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()
        return self.get_action_receipt(receipt_id)

    def _verify_expectation(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        expectation: DesktopExpectation,
    ) -> list[dict[str, Any]]:
        text_parts = [after["window_title"], after["active_app"]]
        for item in after["elements"]:
            text_parts.extend([str(item.get("name") or ""), str(item.get("value") or "")])
        corpus = "\n".join(text_parts).casefold()
        checks: list[dict[str, Any]] = []

        def add(name: str, ok: bool, detail: str) -> None:
            checks.append({"name": name, "ok": bool(ok), "detail": detail})

        for text in expectation.required_text:
            add(f"required_text:{text}", text.casefold() in corpus, "required text search")
        for text in expectation.forbidden_text:
            add(f"forbidden_text:{text}", text.casefold() not in corpus, "forbidden text search")
        if expectation.active_app:
            add("active_app", after["active_app"].casefold() == expectation.active_app.casefold(), after["active_app"])
        if expectation.window_title_contains:
            add(
                "window_title_contains",
                expectation.window_title_contains.casefold() in after["window_title"].casefold(),
                after["window_title"],
            )
        if expectation.element_present:
            try:
                self._resolve_selector(after, expectation.element_present)
                present = True
            except (KeyError, ValueError):
                present = False
            add("element_present", present, stable_json(expectation.element_present.as_dict()))
        if expectation.element_absent:
            try:
                self._resolve_selector(after, expectation.element_absent)
                absent = False
            except (KeyError, ValueError):
                absent = True
            add("element_absent", absent, stable_json(expectation.element_absent.as_dict()))
        if expectation.value_selector:
            try:
                item = self._resolve_selector(after, expectation.value_selector)
                actual = item.get("value")
            except (KeyError, ValueError):
                actual = None
            add("value_equals", actual == expectation.value_equals, f"actual={actual!r}")
        if expectation.screenshot_changed is not None:
            changed = before.get("visual_fingerprint") != after.get("visual_fingerprint")
            add("screenshot_changed", changed == expectation.screenshot_changed, f"changed={changed}")
        return checks

    @staticmethod
    def _resolve_selector(
        observation: dict[str, Any], selector: DesktopSelector | None
    ) -> dict[str, Any]:
        if selector is None:
            raise ValueError("A selector is required")
        matches: list[dict[str, Any]] = []
        for item in observation["elements"]:
            if selector.element_id and item.get("element_id") != selector.element_id:
                continue
            if selector.automation_id and item.get("automation_id") != selector.automation_id:
                continue
            if selector.role and str(item.get("role", "")).casefold() != selector.role.casefold():
                continue
            if selector.name and str(item.get("name", "")).casefold() != selector.name.casefold():
                continue
            matches.append(item)
        if selector.index is not None:
            if selector.index >= len(matches):
                raise KeyError("Desktop selector index does not exist")
            return matches[selector.index]
        if not matches:
            raise KeyError("Desktop selector matched no element")
        if len(matches) != 1:
            raise ValueError(f"Desktop selector is ambiguous and matched {len(matches)} elements")
        return matches[0]

    def get_action(self, action_id: str) -> dict[str, Any]:
        row = self._action_row(action_id)
        result = dict(row)
        result["selector"] = json.loads(result.pop("selector_json"))
        result["target"] = json.loads(result.pop("target_json"))
        result["action"] = json.loads(result.pop("action_json"))
        result["integrity_valid"] = sha256_json(result["action"]) == result["action_hash"]
        result["approvals"] = [
            dict(item)
            for item in self.ledger.db.conn.execute(
                "SELECT * FROM desktop_action_approvals WHERE action_id = ? ORDER BY created_at",
                (action_id,),
            ).fetchall()
        ]
        result["receipts"] = [
            self.get_action_receipt(item["id"])
            for item in self.ledger.db.conn.execute(
                "SELECT id FROM desktop_action_receipts WHERE action_id = ? ORDER BY created_at",
                (action_id,),
            ).fetchall()
        ]
        return result

    def get_action_approval(self, approval_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM desktop_action_approvals WHERE id = ?", (approval_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown desktop action approval: {approval_id}")
        return dict(row)

    def get_action_receipt(self, receipt_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM desktop_action_receipts WHERE id = ?", (receipt_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown desktop action receipt: {receipt_id}")
        result = dict(row)
        result["response"] = json.loads(result.pop("response_json"))
        result["checks"] = json.loads(result.pop("checks_json"))
        result["integrity_valid"] = self.verify_action_receipt(receipt_id)
        return result

    def verify_action_receipt(self, receipt_id: str) -> bool:
        row = self.ledger.db.conn.execute(
            """SELECT r.*, a.action_hash
               FROM desktop_action_receipts r JOIN desktop_actions a ON a.id = r.action_id
               WHERE r.id = ?""",
            (receipt_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown desktop action receipt: {receipt_id}")
        payload = {
            "action_id": row["action_id"],
            "action_hash": row["action_hash"],
            "provider": row["provider"],
            "ok": bool(row["ok"]),
            "provider_response": json.loads(row["response_json"]),
            "post_observation_id": row["post_observation_id"],
            "checks": json.loads(row["checks_json"]),
            "error": row["error"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        }
        return sha256_json(payload) == row["receipt_hash"]

    def _action_row(self, action_id: str):
        row = self.ledger.db.conn.execute(
            "SELECT * FROM desktop_actions WHERE id = ?", (action_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown desktop action: {action_id}")
        return row

    @staticmethod
    def _require_action_integrity(action) -> None:
        payload = json.loads(action["action_json"])
        if sha256_json(payload) != action["action_hash"]:
            raise ValueError("Desktop action integrity verification failed")

    # ------------------------------------------------------------------
    # Adaptive workflow learning
    # ------------------------------------------------------------------
    def learn_workflow_from_plan(
        self,
        plan_id: str,
        *,
        name: str,
        description: str,
        parameter_map: dict[str, str] | None = None,
        required_observation_id: str | None = None,
        actor: str = "user",
    ) -> dict[str, Any]:
        name = name.strip()
        description = description.strip()
        if not name or not description:
            raise ValueError("Workflow name and description are required")
        agent = ComputerAgentRuntime(self.ledger)
        plan = agent.get_plan(plan_id)
        if plan["status"] != "succeeded":
            raise ValueError("Only a fully succeeded plan can become a workflow")
        if not agent.verify_plan(plan_id):
            raise ValueError("Source computer plan integrity verification failed")
        parameter_map = parameter_map or {}
        if not isinstance(parameter_map, dict):
            raise TypeError("parameter_map must be an object")
        parameters: dict[str, dict[str, Any]] = {}
        steps: list[dict[str, Any]] = []
        contract_hashes: dict[str, str] = {}
        for step in plan["steps"]:
            if step["status"] != ComputerStepStatus.SUCCEEDED.value:
                raise ValueError("Every source step must have succeeded")
            successful_receipts = [item for item in step["receipts"] if item["ok"]]
            if not successful_receipts:
                raise ValueError(f"Step {step['step_key']} has no successful receipt")
            if not all(agent.verify_receipt(item["id"]) for item in successful_receipts):
                raise ValueError(f"Step {step['step_key']} has a tampered receipt")
            contract = agent.registry.contract(step["skill_name"])
            contract_hashes[contract.name] = sha256_json(contract.as_dict())
            args = json.loads(json.dumps(step["args"]))
            for path, parameter_name in parameter_map.items():
                step_key, separator, arg_name = path.partition(".")
                if not separator or step_key != step["step_key"]:
                    continue
                if arg_name not in args:
                    raise ValueError(f"Parameter mapping refers to missing argument: {path}")
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", parameter_name):
                    raise ValueError(f"Invalid workflow parameter name: {parameter_name}")
                original = args[arg_name]
                declared = parameters.get(parameter_name)
                inferred = {"type": _json_type(original), "source": path}
                if declared and declared["type"] != inferred["type"]:
                    raise ValueError(f"Workflow parameter {parameter_name} is used with incompatible types")
                parameters[parameter_name] = inferred
                args[arg_name] = {"$param": parameter_name}
            steps.append(
                {
                    "key": step["step_key"],
                    "skill": step["skill_name"],
                    "args": args,
                    "depends_on": step["depends_on"],
                    "obligations": step["obligations"],
                    "required_claims": step["required_claims"],
                    "risk": step["risk"],
                    "max_attempts": step["max_attempts"],
                }
            )
        required_state_fingerprint = None
        if required_observation_id:
            observation = self.get_observation(required_observation_id)
            if not observation["integrity_valid"]:
                raise ValueError("Required desktop observation is tampered")
            required_state_fingerprint = observation["state_fingerprint"]
        definition = {
            "api_version": ADAPTIVE_API_VERSION,
            "kind": "verified_plan_template",
            "name": name,
            "description": description,
            "parameters": parameters,
            "steps": steps,
            "source_plan_id": plan_id,
            "source_plan_hash": plan["plan_hash"],
            "skill_contract_hashes": contract_hashes,
            "required_state_fingerprint": required_state_fingerprint,
        }
        workflow_id = new_id("awf")
        workflow_hash = sha256_json(definition)
        now = utcnow()
        self.ledger.db.conn.execute(
            """INSERT INTO adaptive_workflows
               (id, name, description, status, source_plan_id, definition_json,
                workflow_hash, required_state_fingerprint, created_by, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                workflow_id,
                name,
                description,
                AdaptiveWorkflowStatus.PROPOSED.value,
                plan_id,
                stable_json(definition),
                workflow_hash,
                required_state_fingerprint,
                actor,
                now,
                now,
            ),
        )
        self.ledger._event(
            "adaptive_workflow",
            workflow_id,
            "ADAPTIVE_WORKFLOW_PROPOSED",
            {"workflow_hash": workflow_hash, "source_plan_id": plan_id, "parameters": sorted(parameters)},
            actor,
            ActorRole.HUMAN,
        )
        self.ledger.db.conn.commit()
        return self.get_workflow(workflow_id)

    def review_workflow(
        self,
        workflow_id: str,
        *,
        approved: bool,
        reviewer: str,
        rationale: str,
    ) -> dict[str, Any]:
        if not reviewer.strip() or not rationale.strip():
            raise ValueError("Reviewer and rationale are required")
        workflow = self._workflow_row(workflow_id)
        self._require_workflow_integrity(workflow)
        if workflow["status"] not in {
            AdaptiveWorkflowStatus.PROPOSED.value,
            AdaptiveWorkflowStatus.STALE.value,
        }:
            raise ValueError("Workflow is not awaiting review")
        status = AdaptiveWorkflowStatus.ACTIVE.value if approved else AdaptiveWorkflowStatus.REJECTED.value
        review_id = new_id("awr")
        now = utcnow()
        self.ledger.db.conn.execute(
            """INSERT INTO adaptive_workflow_reviews
               (id, workflow_id, workflow_hash, decision, reviewer, rationale, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (review_id, workflow_id, workflow["workflow_hash"], status, reviewer, rationale, now),
        )
        self.ledger.db.conn.execute(
            "UPDATE adaptive_workflows SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, workflow_id),
        )
        self.ledger._event(
            "adaptive_workflow",
            workflow_id,
            "ADAPTIVE_WORKFLOW_ACTIVATED" if approved else "ADAPTIVE_WORKFLOW_REJECTED",
            {"review_id": review_id, "workflow_hash": workflow["workflow_hash"], "rationale": rationale},
            reviewer,
            ActorRole.HUMAN,
        )
        self.ledger.db.conn.commit()
        return self.get_workflow(workflow_id)

    def instantiate_workflow(
        self,
        workflow_id: str,
        parameters: dict[str, Any],
        *,
        workspace: str | Path | None = None,
        autonomy_mode: AutonomyMode | str = AutonomyMode.VERIFIED,
        current_observation_id: str | None = None,
        actor: str = "user",
    ) -> dict[str, Any]:
        workflow = self._workflow_row(workflow_id)
        self._require_workflow_integrity(workflow)
        if workflow["status"] != AdaptiveWorkflowStatus.ACTIVE.value:
            raise PermissionError("Only active reviewed workflows can be instantiated")
        definition = json.loads(workflow["definition_json"])
        runtime = ComputerAgentRuntime(self.ledger, workspace) if workspace else self.ledger.agent
        for skill_name, expected_hash in definition["skill_contract_hashes"].items():
            if not runtime.registry.has_contract(skill_name):
                self._mark_workflow_stale(workflow_id, f"Missing skill contract: {skill_name}")
                raise RuntimeError(f"Workflow is stale because skill {skill_name} is unavailable")
            current_hash = sha256_json(runtime.registry.contract(skill_name).as_dict())
            if current_hash != expected_hash:
                self._mark_workflow_stale(workflow_id, f"Skill contract changed: {skill_name}")
                raise RuntimeError(f"Workflow is stale because skill {skill_name} changed")
        required_state = definition.get("required_state_fingerprint")
        if required_state:
            if not current_observation_id:
                raise ValueError("Workflow requires a current desktop observation")
            current_observation = self.get_observation(current_observation_id)
            if current_observation["state_fingerprint"] != required_state:
                raise RuntimeError("Current desktop state does not match the workflow's reviewed state")
        declared = definition.get("parameters", {})
        if set(parameters) != set(declared):
            missing = sorted(set(declared) - set(parameters))
            extra = sorted(set(parameters) - set(declared))
            raise ValueError(f"Workflow parameters mismatch; missing={missing}, extra={extra}")
        for name, spec in declared.items():
            if _json_type(parameters[name]) != spec["type"]:
                raise TypeError(f"Workflow parameter {name} must be {spec['type']}")
        materialized_steps: list[ComputerStepSpec] = []
        for raw_step in definition["steps"]:
            materialized_steps.append(
                ComputerStepSpec(
                    key=raw_step["key"],
                    skill=raw_step["skill"],
                    args=_materialize(raw_step["args"], parameters),
                    depends_on=list(raw_step.get("depends_on", [])),
                    obligations=list(raw_step.get("obligations", [])),
                    required_claims=list(raw_step.get("required_claims", [])),
                    risk=RiskLevel(raw_step["risk"]),
                    max_attempts=int(raw_step.get("max_attempts", 1)),
                )
            )
        goal_spec = ComputerGoalSpec(
            utterance=f"Run reviewed workflow: {workflow['name']}",
            goal_type=ComputerGoalType.LEARNED_WORKFLOW,
            parameters={"workflow_id": workflow_id, "parameters": parameters},
            constraints={"workflow_hash": workflow["workflow_hash"], "workspace_only": True},
            success_conditions=[{"type": "all_workflow_steps_verified"}],
            parser_trace=["adaptive_workflow_instantiation"],
        )
        plan_spec = ComputerPlanSpec(
            goal_type=ComputerGoalType.LEARNED_WORKFLOW,
            summary=f"Reviewed workflow: {workflow['name']}",
            steps=materialized_steps,
            rationale=[
                "Materialized from a human-reviewed successful plan",
                f"Workflow hash: {workflow['workflow_hash']}",
            ],
        )
        goal_id, plan_id = self._persist_materialized_plan(
            runtime, goal_spec, plan_spec, AutonomyMode(autonomy_mode), actor
        )
        instance_id = new_id("awi")
        binding = {
            "workflow_id": workflow_id,
            "workflow_hash": workflow["workflow_hash"],
            "parameters": parameters,
            "goal_id": goal_id,
            "plan_id": plan_id,
            "workspace": str(runtime.boundary.root),
            "autonomy_mode": AutonomyMode(autonomy_mode).value,
        }
        binding_hash = sha256_json(binding)
        self.ledger.db.conn.execute(
            """INSERT INTO adaptive_workflow_instances
               (id, workflow_id, workflow_hash, parameters_json, goal_id, plan_id,
                binding_hash, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'created', ?, ?)""",
            (
                instance_id,
                workflow_id,
                workflow["workflow_hash"],
                stable_json(parameters),
                goal_id,
                plan_id,
                binding_hash,
                utcnow(),
                utcnow(),
            ),
        )
        self.ledger.db.conn.commit()
        return {
            "id": instance_id,
            "workflow_id": workflow_id,
            "workflow_hash": workflow["workflow_hash"],
            "parameters": parameters,
            "goal_id": goal_id,
            "plan_id": plan_id,
            "binding_hash": binding_hash,
            "plan": runtime.get_plan(plan_id),
        }

    def compose_workflows(
        self,
        workflow_ids: list[str],
        *,
        name: str,
        description: str,
        actor: str = "user",
    ) -> dict[str, Any]:
        if len(workflow_ids) < 2:
            raise ValueError("Composition requires at least two active workflows")
        definitions = []
        for workflow_id in workflow_ids:
            row = self._workflow_row(workflow_id)
            self._require_workflow_integrity(row)
            if row["status"] != AdaptiveWorkflowStatus.ACTIVE.value:
                raise PermissionError(f"Workflow {workflow_id} is not active")
            definitions.append((workflow_id, json.loads(row["definition_json"]), row["workflow_hash"]))
        parameters: dict[str, Any] = {}
        steps: list[dict[str, Any]] = []
        skill_contract_hashes: dict[str, str] = {}
        previous_tail: str | None = None
        for position, (workflow_id, definition, _) in enumerate(definitions, start=1):
            prefix = f"w{position}_"
            parameter_rename: dict[str, str] = {}
            for parameter_name, parameter_spec in definition.get("parameters", {}).items():
                new_name = f"w{position}_{parameter_name}"
                parameter_rename[parameter_name] = new_name
                parameters[new_name] = {**parameter_spec, "source_workflow_id": workflow_id}
            local_keys = {step["key"]: f"{prefix}{step['key']}" for step in definition["steps"]}
            for idx, step in enumerate(definition["steps"]):
                args = _rename_parameters(step["args"], parameter_rename)
                dependencies = [local_keys[key] for key in step.get("depends_on", [])]
                if idx == 0 and previous_tail:
                    dependencies.append(previous_tail)
                new_step = {
                    **step,
                    "key": local_keys[step["key"]],
                    "args": args,
                    "depends_on": dependencies,
                }
                steps.append(new_step)
            previous_tail = local_keys[definition["steps"][-1]["key"]] if definition["steps"] else previous_tail
            skill_contract_hashes.update(definition.get("skill_contract_hashes", {}))
        definition = {
            "api_version": ADAPTIVE_API_VERSION,
            "kind": "composed_workflow",
            "name": name.strip(),
            "description": description.strip(),
            "parameters": parameters,
            "steps": steps,
            "source_workflow_ids": workflow_ids,
            "source_workflow_hashes": {workflow_id: hash_value for workflow_id, _, hash_value in definitions},
            "skill_contract_hashes": skill_contract_hashes,
            "required_state_fingerprint": None,
        }
        if not definition["name"] or not definition["description"]:
            raise ValueError("Workflow name and description are required")
        workflow_id = new_id("awf")
        workflow_hash = sha256_json(definition)
        now = utcnow()
        self.ledger.db.conn.execute(
            """INSERT INTO adaptive_workflows
               (id, name, description, status, source_plan_id, definition_json,
                workflow_hash, required_state_fingerprint, created_by, created_at, updated_at)
               VALUES (?, ?, ?, ?, NULL, ?, ?, NULL, ?, ?, ?)""",
            (
                workflow_id,
                definition["name"],
                definition["description"],
                AdaptiveWorkflowStatus.PROPOSED.value,
                stable_json(definition),
                workflow_hash,
                actor,
                now,
                now,
            ),
        )
        self.ledger.db.conn.commit()
        return self.get_workflow(workflow_id)

    def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        row = self._workflow_row(workflow_id)
        result = dict(row)
        result["definition"] = json.loads(result.pop("definition_json"))
        result["integrity_valid"] = sha256_json(result["definition"]) == result["workflow_hash"]
        result["reviews"] = [
            dict(item)
            for item in self.ledger.db.conn.execute(
                "SELECT * FROM adaptive_workflow_reviews WHERE workflow_id = ? ORDER BY created_at",
                (workflow_id,),
            ).fetchall()
        ]
        result["instances"] = [
            dict(item)
            for item in self.ledger.db.conn.execute(
                "SELECT * FROM adaptive_workflow_instances WHERE workflow_id = ? ORDER BY created_at DESC",
                (workflow_id,),
            ).fetchall()
        ]
        for item in result["instances"]:
            item["parameters"] = json.loads(item.pop("parameters_json"))
        return result

    def list_workflows(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            AdaptiveWorkflowStatus(status)
            rows = self.ledger.db.conn.execute(
                "SELECT id FROM adaptive_workflows WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self.ledger.db.conn.execute(
                "SELECT id FROM adaptive_workflows ORDER BY created_at DESC"
            ).fetchall()
        return [self.get_workflow(row["id"]) for row in rows]

    def verify_workflow(self, workflow_id: str) -> bool:
        row = self._workflow_row(workflow_id)
        return sha256_json(json.loads(row["definition_json"])) == row["workflow_hash"]

    def _mark_workflow_stale(self, workflow_id: str, reason: str) -> None:
        self.ledger.db.conn.execute(
            "UPDATE adaptive_workflows SET status = ?, updated_at = ? WHERE id = ?",
            (AdaptiveWorkflowStatus.STALE.value, utcnow(), workflow_id),
        )
        self.ledger._event(
            "adaptive_workflow",
            workflow_id,
            "ADAPTIVE_WORKFLOW_STALE",
            {"reason": reason},
            "adaptive_runtime",
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()

    def _workflow_row(self, workflow_id: str):
        row = self.ledger.db.conn.execute(
            "SELECT * FROM adaptive_workflows WHERE id = ?", (workflow_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown adaptive workflow: {workflow_id}")
        return row

    @staticmethod
    def _require_workflow_integrity(workflow) -> None:
        if sha256_json(json.loads(workflow["definition_json"])) != workflow["workflow_hash"]:
            raise ValueError("Adaptive workflow integrity verification failed")

    def _persist_materialized_plan(
        self,
        runtime: ComputerAgentRuntime,
        goal_spec: ComputerGoalSpec,
        plan_spec: ComputerPlanSpec,
        mode: AutonomyMode,
        actor: str,
    ) -> tuple[str, str]:
        goal_id = new_id("cgl")
        plan_id = new_id("cpl")
        now = utcnow()
        goal_payload = goal_spec.as_dict()
        plan_payload = plan_spec.as_dict()
        plan_hash = sha256_json(plan_payload)
        self.ledger.db.conn.execute(
            """INSERT INTO computer_goals
               (id, utterance, goal_type, structured_json, workspace, autonomy_mode,
                status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                goal_id,
                goal_spec.utterance,
                goal_spec.goal_type.value,
                stable_json(goal_payload),
                str(runtime.boundary.root),
                mode.value,
                ComputerGoalStatus.PLANNED.value,
                now,
                now,
            ),
        )
        self.ledger.db.conn.execute(
            """INSERT INTO computer_plans
               (id, goal_id, revision, status, plan_json, plan_hash, rationale_json, created_at, updated_at)
               VALUES (?, ?, 1, 'pending', ?, ?, ?, ?, ?)""",
            (plan_id, goal_id, stable_json(plan_payload), plan_hash, stable_json(plan_spec.rationale), now, now),
        )
        for sequence, step in enumerate(plan_spec.steps):
            contract = runtime.registry.contract(step.skill)
            risk = step.risk or contract.risk
            args_hash = sha256_json(step.args)
            self.ledger.db.conn.execute(
                """INSERT INTO computer_steps
                   (id, plan_id, step_key, sequence, skill_name, args_json, args_hash,
                    depends_on_json, obligations_json, required_claims_json, risk,
                    status, attempts, max_attempts, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
                (
                    new_id("cst"),
                    plan_id,
                    step.key,
                    sequence,
                    step.skill,
                    stable_json(step.args),
                    args_hash,
                    stable_json(step.depends_on),
                    stable_json(step.obligations),
                    stable_json(step.required_claims),
                    risk.value,
                    ComputerStepStatus.PENDING.value,
                    max(1, step.max_attempts),
                    now,
                    now,
                ),
            )
        self.ledger._event(
            "adaptive_workflow_instance",
            plan_id,
            "ADAPTIVE_WORKFLOW_INSTANTIATED",
            {"goal_id": goal_id, "plan_id": plan_id, "plan_hash": plan_hash},
            actor,
            ActorRole.HUMAN,
        )
        return goal_id, plan_id


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise TypeError(f"Unsupported workflow parameter type: {type(value).__name__}")


def _materialize(value: Any, parameters: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$param"}:
            name = value["$param"]
            if name not in parameters:
                raise KeyError(f"Missing workflow parameter: {name}")
            return json.loads(json.dumps(parameters[name]))
        return {key: _materialize(item, parameters) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize(item, parameters) for item in value]
    return value


def _rename_parameters(value: Any, rename: dict[str, str]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$param"}:
            return {"$param": rename.get(value["$param"], value["$param"])}
        return {key: _rename_parameters(item, rename) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename_parameters(item, rename) for item in value]
    return value
