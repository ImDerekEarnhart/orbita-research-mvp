from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

from .execution import ContainerExecutionSpec
from .models import ActorRole, RiskLevel, SupportState

if TYPE_CHECKING:  # pragma: no cover
    from .ledger import EpistemicLedger
    from .execution import OCIEngine


AGENT_OS_API_VERSION = "1.4"

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)



class AutonomyMode(StrEnum):
    OBSERVE = "observe"
    PROPOSE = "propose"
    VERIFIED = "verified"
    GOVERNED = "governed"


class ComputerGoalType(StrEnum):
    INSPECT_WORKSPACE = "inspect_workspace"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    COPY_FILE = "copy_file"
    BACKUP_TREE = "backup_tree"
    HASH_FILE = "hash_file"
    LIST_FILES = "list_files"
    GIT_STATUS = "git_status"
    GIT_DIFF = "git_diff"
    PARSE_PYTHON = "parse_python"
    RUN_CONTAINER_TASK = "run_container_task"
    REPAIR_CODE = "repair_code"
    BROWSE_VERIFY = "browse_verify"
    DRAFT_EMAIL = "draft_email"
    DRAFT_CALENDAR = "draft_calendar"
    LAUNCH_WINDOWS_APP = "launch_windows_app"
    LEARNED_WORKFLOW = "learned_workflow"
    UNKNOWN = "unknown"


class ComputerGoalStatus(StrEnum):
    PROPOSED = "proposed"
    NEEDS_CLARIFICATION = "needs_clarification"
    PLANNED = "planned"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_INPUT = "waiting_input"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ComputerStepStatus(StrEnum):
    PENDING = "pending"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_INPUT = "waiting_input"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass(slots=True)
class ComputerGoalSpec:
    utterance: str
    goal_type: ComputerGoalType
    parameters: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    success_conditions: list[dict[str, Any]] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    parser_trace: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.goal_type != ComputerGoalType.UNKNOWN and not self.ambiguities

    def as_dict(self) -> dict[str, Any]:
        return {
            "utterance": self.utterance,
            "goal_type": self.goal_type.value,
            "parameters": self.parameters,
            "constraints": self.constraints,
            "success_conditions": self.success_conditions,
            "ambiguities": self.ambiguities,
            "parser_trace": self.parser_trace,
            "ready": self.ready,
        }


@dataclass(frozen=True, slots=True)
class SkillContract:
    name: str
    description: str
    risk: RiskLevel
    mutates: bool
    reversible: bool
    required_args: tuple[str, ...] = ()
    default_obligations: tuple[dict[str, Any], ...] = ()
    external_side_effect: bool = False
    requires_container: bool = False

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["risk"] = self.risk.value
        value["required_args"] = list(self.required_args)
        value["default_obligations"] = list(self.default_obligations)
        return value


@dataclass(slots=True)
class SkillResult:
    ok: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    waiting_input: bool = False
    waiting_approval: bool = False


@dataclass(slots=True)
class ComputerStepSpec:
    key: str
    skill: str
    args: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)
    obligations: list[dict[str, Any]] = field(default_factory=list)
    required_claims: list[str] = field(default_factory=list)
    risk: RiskLevel | None = None
    max_attempts: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "skill": self.skill,
            "args": self.args,
            "depends_on": self.depends_on,
            "obligations": self.obligations,
            "required_claims": self.required_claims,
            "risk": self.risk.value if self.risk else None,
            "max_attempts": self.max_attempts,
        }


@dataclass(slots=True)
class ComputerPlanSpec:
    goal_type: ComputerGoalType
    summary: str
    steps: list[ComputerStepSpec]
    rationale: list[str] = field(default_factory=list)
    configuration_needed: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal_type": self.goal_type.value,
            "summary": self.summary,
            "steps": [step.as_dict() for step in self.steps],
            "rationale": self.rationale,
            "configuration_needed": self.configuration_needed,
        }


class ExternalSkillAdapter(Protocol):
    def __call__(self, args: dict[str, Any]) -> SkillResult: ...


class WorkspaceBoundary:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str | Path, *, must_exist: bool = False) -> Path:
        raw = Path(str(relative))
        if raw.is_absolute():
            raise PermissionError("Absolute paths are not permitted by this workspace capability")
        candidate = (self.root / raw).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"Path escapes the configured workspace: {relative}") from exc
        if must_exist and not candidate.exists():
            raise FileNotFoundError(str(relative))
        return candidate

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()


class ComputerGoalCompiler:
    """Controlled language compiler for computer goals.

    It intentionally refuses vague goals instead of inventing dangerous details.
    Open-ended coding goals compile to an inspect/test/patch/retest workflow but
    remain blocked until an immutable container test specification and patch
    proposal provider are supplied.
    """

    @staticmethod
    def _clean(value: str) -> str:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            return value[1:-1]
        return value

    def compile(self, utterance: str) -> ComputerGoalSpec:
        text = " ".join(utterance.strip().split())
        low = text.casefold().rstrip(".")
        if not text:
            return ComputerGoalSpec(utterance, ComputerGoalType.UNKNOWN, ambiguities=["goal is empty"])

        for pattern in (
            r"(?:write|save)\s+(.+?)\s+to\s+(?:file\s+)?(.+)",
            r"create\s+(?:a\s+)?file\s+(.+?)\s+with\s+(?:the\s+)?(?:text|content)\s+(.+)",
        ):
            match = re.fullmatch(pattern, text, flags=re.I)
            if match:
                if pattern.startswith("create"):
                    path, content = match.group(1), match.group(2)
                else:
                    content, path = match.group(1), match.group(2)
                path, content = self._clean(path), self._clean(content)
                return ComputerGoalSpec(
                    utterance,
                    ComputerGoalType.WRITE_FILE,
                    {"path": path, "text": content},
                    {"workspace_only": True, "overwrite": False},
                    [{"type": "file_exists", "path": path}, {"type": "content_matches", "path": path}],
                    parser_trace=["write_file_pattern"],
                )

        match = re.fullmatch(r"(?:read|show|open)\s+(?:the\s+)?(?:file\s+)?(.+)", text, flags=re.I)
        if match and not match.group(1).casefold().startswith(("http://", "https://", "app ")):
            path = self._clean(match.group(1))
            return ComputerGoalSpec(
                utterance,
                ComputerGoalType.READ_FILE,
                {"path": path},
                {"workspace_only": True},
                [{"type": "output_key", "key": "text"}],
                parser_trace=["read_file_pattern"],
            )

        match = re.fullmatch(r"copy\s+(.+?)\s+to\s+(.+)", text, flags=re.I)
        if match:
            source, destination = map(self._clean, match.groups())
            return ComputerGoalSpec(
                utterance,
                ComputerGoalType.COPY_FILE,
                {"source": source, "destination": destination},
                {"workspace_only": True, "overwrite": False},
                [{"type": "files_equal", "source": source, "destination": destination}],
                parser_trace=["copy_file_pattern"],
            )

        match = re.fullmatch(r"backup\s+(.+?)\s+to\s+(.+)", text, flags=re.I)
        if match:
            source, destination = map(self._clean, match.groups())
            return ComputerGoalSpec(
                utterance,
                ComputerGoalType.BACKUP_TREE,
                {"source": source, "destination": destination},
                {"workspace_only": True, "overwrite": False, "max_files": 1000, "max_bytes": 100_000_000},
                [{"type": "backup_manifest_verified"}],
                parser_trace=["backup_tree_pattern"],
            )

        match = re.fullmatch(r"(?:hash|checksum)\s+(?:the\s+)?(?:file\s+)?(.+)", text, flags=re.I)
        if match:
            path = self._clean(match.group(1))
            return ComputerGoalSpec(
                utterance,
                ComputerGoalType.HASH_FILE,
                {"path": path},
                {"workspace_only": True},
                [{"type": "output_key", "key": "sha256"}],
                parser_trace=["hash_file_pattern"],
            )

        match = re.fullmatch(r"(?:list|show)\s+(?:the\s+)?files(?:\s+in\s+(.+))?", text, flags=re.I)
        if match:
            path = self._clean(match.group(1) or ".")
            return ComputerGoalSpec(
                utterance,
                ComputerGoalType.LIST_FILES,
                {"path": path},
                {"workspace_only": True},
                [{"type": "output_key", "key": "entries"}],
                parser_trace=["list_files_pattern"],
            )

        if re.fullmatch(r"(?:inspect|scan|summarize)\s+(?:the\s+)?(?:workspace|repository|project)", text, flags=re.I):
            return ComputerGoalSpec(
                utterance,
                ComputerGoalType.INSPECT_WORKSPACE,
                {},
                {"read_only": True},
                [{"type": "output_key", "key": "file_count"}],
                parser_trace=["inspect_workspace_pattern"],
            )

        if re.fullmatch(r"(?:show|check|get)\s+(?:the\s+)?git\s+status", text, flags=re.I):
            return ComputerGoalSpec(
                utterance,
                ComputerGoalType.GIT_STATUS,
                {},
                {"read_only": True},
                [{"type": "output_key", "key": "status"}],
                parser_trace=["git_status_pattern"],
            )

        if re.fullmatch(r"(?:show|check|get)\s+(?:the\s+)?git\s+diff", text, flags=re.I):
            return ComputerGoalSpec(
                utterance,
                ComputerGoalType.GIT_DIFF,
                {},
                {"read_only": True},
                [{"type": "output_key", "key": "diff"}],
                parser_trace=["git_diff_pattern"],
            )

        match = re.fullmatch(r"(?:check|parse|validate)\s+(?:the\s+)?python\s+(?:file\s+)?(.+)", text, flags=re.I)
        if match:
            path = self._clean(match.group(1))
            return ComputerGoalSpec(
                utterance,
                ComputerGoalType.PARSE_PYTHON,
                {"path": path},
                {"read_only": True},
                [{"type": "output_value", "key": "valid", "equals": True}],
                parser_trace=["parse_python_pattern"],
            )

        match = re.fullmatch(r"run\s+(.+?)\s+using\s+(?:the\s+)?execution\s+spec\s+(.+)", text, flags=re.I)
        if match:
            name, spec_path = map(self._clean, match.groups())
            return ComputerGoalSpec(
                utterance,
                ComputerGoalType.RUN_CONTAINER_TASK,
                {"name": name, "spec_path": spec_path},
                {"container_required": True, "human_approval": True},
                [{"type": "container_receipt_valid"}],
                parser_trace=["container_task_pattern"],
            )

        match = re.fullmatch(
            r'(?:open|visit|browse to)\s+(https?://\S+)\s+and\s+verify\s+(?:it\s+)?contains\s+["\'](.+?)["\']',
            text,
            flags=re.I,
        )
        if match:
            url, required = match.groups()
            return ComputerGoalSpec(
                utterance,
                ComputerGoalType.BROWSE_VERIFY,
                {
                    "url": url,
                    "verification": {
                        "expected_url_prefix": url,
                        "required_text": [required],
                        "forbidden_text": [],
                    },
                },
                {"external_provider_required": True, "human_approval": True},
                [{"type": "output_value", "key": "verified", "equals": True}],
                parser_trace=["browse_verify_pattern"],
            )

        match = re.fullmatch(
            r'draft\s+(?:an\s+)?email\s+to\s+(\S+@\S+)\s+subject\s+["\'](.+?)["\']\s+body\s+["\'](.+)["\']',
            text,
            flags=re.I,
        )
        if match:
            recipient, subject, body = match.groups()
            return ComputerGoalSpec(
                utterance,
                ComputerGoalType.DRAFT_EMAIL,
                {"to": recipient, "subject": subject, "body": body},
                {"draft_only": True, "no_send": True},
                [{"type": "output_key", "key": "draft_id"}],
                parser_trace=["draft_email_pattern"],
            )

        match = re.fullmatch(
            r'draft\s+(?:a\s+)?calendar\s+event\s+["\'](.+?)["\']\s+from\s+["\'](.+?)["\']\s+to\s+["\'](.+?)["\']\s+timezone\s+["\'](.+?)["\']',
            text,
            flags=re.I,
        )
        if match:
            title, start, end, timezone_name = match.groups()
            return ComputerGoalSpec(
                utterance,
                ComputerGoalType.DRAFT_CALENDAR,
                {"title": title, "start": start, "end": end, "timezone": timezone_name},
                {"draft_only": True, "no_calendar_change": True},
                [{"type": "output_key", "key": "draft_id"}],
                parser_trace=["draft_calendar_pattern"],
            )

        match = re.fullmatch(r'(?:launch|open|start)\s+(?:the\s+)?app\s+([a-zA-Z0-9._-]+)', text, flags=re.I)
        if match:
            app_id = match.group(1).casefold()
            return ComputerGoalSpec(
                utterance,
                ComputerGoalType.LAUNCH_WINDOWS_APP,
                {"app_id": app_id, "arguments": []},
                {"allowlisted_app_only": True, "human_approval": True},
                [{"type": "output_value", "key": "verified", "equals": True}],
                parser_trace=["launch_windows_app_pattern"],
            )

        if any(phrase in low for phrase in ("fix failing tests", "repair the code", "debug the project", "fix the code")):
            ambiguities: list[str] = []
            spec_match = re.search(r"execution\s+spec\s+([^,]+)$", text, flags=re.I)
            spec_path = self._clean(spec_match.group(1)) if spec_match else None
            if not spec_path:
                ambiguities.append("an immutable container execution spec for the test command is required")
            return ComputerGoalSpec(
                utterance,
                ComputerGoalType.REPAIR_CODE,
                {"test_spec_path": spec_path, "patch_provider": None},
                {
                    "branch_required": True,
                    "tests_required": True,
                    "no_push_without_approval": True,
                    "container_required": True,
                },
                [
                    {"type": "tests_reproduced"},
                    {"type": "patch_diff_scoped"},
                    {"type": "tests_pass"},
                ],
                ambiguities=ambiguities,
                parser_trace=["code_repair_workflow"],
            )

        return ComputerGoalSpec(
            utterance,
            ComputerGoalType.UNKNOWN,
            ambiguities=["the request does not match a registered goal pattern"],
            parser_trace=["no_safe_goal_pattern"],
        )


class ComputerPlanner:
    def __init__(self, registry: "ComputerSkillRegistry"):
        self.registry = registry

    def plan(self, goal: ComputerGoalSpec) -> ComputerPlanSpec:
        p = goal.parameters
        t = goal.goal_type
        if t == ComputerGoalType.WRITE_FILE:
            steps = [ComputerStepSpec("write", "file.write_text", {"path": p["path"], "text": p["text"], "overwrite": False})]
        elif t == ComputerGoalType.READ_FILE:
            steps = [ComputerStepSpec("read", "file.read_text", {"path": p["path"]})]
        elif t == ComputerGoalType.COPY_FILE:
            steps = [
                ComputerStepSpec("hash_source", "file.hash", {"path": p["source"]}),
                ComputerStepSpec(
                    "copy",
                    "file.copy",
                    {"source": p["source"], "destination": p["destination"], "overwrite": False},
                    depends_on=["hash_source"],
                ),
            ]
        elif t == ComputerGoalType.BACKUP_TREE:
            steps = [
                ComputerStepSpec("inspect_source", "workspace.inspect", {"path": p["source"]}),
                ComputerStepSpec(
                    "backup",
                    "backup.copy_tree",
                    {
                        "source": p["source"],
                        "destination": p["destination"],
                        "overwrite": False,
                        "max_files": int(goal.constraints.get("max_files", 1000)),
                        "max_bytes": int(goal.constraints.get("max_bytes", 100_000_000)),
                    },
                    depends_on=["inspect_source"],
                ),
            ]
        elif t == ComputerGoalType.HASH_FILE:
            steps = [ComputerStepSpec("hash", "file.hash", {"path": p["path"]})]
        elif t == ComputerGoalType.LIST_FILES:
            steps = [ComputerStepSpec("list", "file.list", {"path": p["path"]})]
        elif t == ComputerGoalType.INSPECT_WORKSPACE:
            steps = [ComputerStepSpec("inspect", "workspace.inspect", {"path": "."})]
        elif t == ComputerGoalType.GIT_STATUS:
            steps = [ComputerStepSpec("git_status", "git.status", {"path": "."})]
        elif t == ComputerGoalType.GIT_DIFF:
            steps = [ComputerStepSpec("git_diff", "git.diff", {"path": "."})]
        elif t == ComputerGoalType.PARSE_PYTHON:
            steps = [ComputerStepSpec("parse", "code.parse_python", {"path": p["path"]})]
        elif t == ComputerGoalType.RUN_CONTAINER_TASK:
            steps = [
                ComputerStepSpec("inspect", "workspace.inspect", {"path": "."}),
                ComputerStepSpec(
                    "container_task",
                    "container.prepare_execution",
                    {"spec_path": p["spec_path"]},
                    depends_on=["inspect"],
                    risk=RiskLevel.HIGH,
                ),
            ]
        elif t == ComputerGoalType.BROWSE_VERIFY:
            steps = [
                ComputerStepSpec(
                    "browse",
                    "external.browser.navigate_verified",
                    {"url": p["url"], "verification": p["verification"]},
                    risk=RiskLevel.MEDIUM,
                )
            ]
        elif t == ComputerGoalType.DRAFT_EMAIL:
            steps = [
                ComputerStepSpec(
                    "create_email_draft",
                    "integration.email.create_draft",
                    {"to": p["to"], "subject": p["subject"], "body": p["body"]},
                    risk=RiskLevel.LOW,
                )
            ]
        elif t == ComputerGoalType.DRAFT_CALENDAR:
            steps = [
                ComputerStepSpec(
                    "create_calendar_draft",
                    "integration.calendar.create_draft",
                    {
                        "title": p["title"],
                        "start": p["start"],
                        "end": p["end"],
                        "timezone": p["timezone"],
                    },
                    risk=RiskLevel.LOW,
                )
            ]
        elif t == ComputerGoalType.LAUNCH_WINDOWS_APP:
            steps = [
                ComputerStepSpec(
                    "launch_app",
                    "external.windows.launch_app",
                    {"app_id": p["app_id"], "arguments": p.get("arguments", [])},
                    risk=RiskLevel.HIGH,
                )
            ]
        elif t == ComputerGoalType.REPAIR_CODE:
            steps = [
                ComputerStepSpec("inspect", "workspace.inspect", {"path": "."}),
                ComputerStepSpec("git_status", "git.status", {"path": "."}, depends_on=["inspect"]),
                ComputerStepSpec("detect_project", "code.detect_project", {"path": "."}, depends_on=["inspect"]),
                ComputerStepSpec(
                    "start_native_coding_loop",
                    "coding.start_repair",
                    {
                        "repository": ".",
                        "goal": goal.utterance,
                        "test_spec_path": p.get("test_spec_path"),
                        "allowed_paths": ["."],
                        "max_candidates": 4,
                    },
                    depends_on=["git_status", "detect_project"],
                    risk=RiskLevel.MEDIUM,
                ),
            ]
        else:
            return ComputerPlanSpec(t, "No executable plan", [], ["Goal is not safely understood"], ["clarify goal"])

        for step in steps:
            contract = self.registry.contract(step.skill)
            if step.risk is None:
                step.risk = contract.risk
            if not step.obligations:
                step.obligations = [dict(item) for item in contract.default_obligations]
        configuration_needed = list(goal.ambiguities)
        for step in steps:
            if step.skill.startswith("external.") and not self.registry.has_handler(step.skill):
                configuration_needed.append(f"skill adapter required: {step.skill}")
            if step.skill == "container.prepare_execution" and not step.args.get("spec_path"):
                configuration_needed.append("container execution spec path is missing")
        return ComputerPlanSpec(
            t,
            f"Execute {t.value} through registered capability contracts",
            steps,
            [
                "Goal was compiled into a typed task",
                "Each step names a capability, dependencies, risk, and proof obligations",
                "Mutating or external steps remain approval-bound",
            ],
            list(dict.fromkeys(configuration_needed)),
        )


class ComputerSkillRegistry:
    def __init__(self, ledger: "EpistemicLedger", boundary: WorkspaceBoundary):
        self.ledger = ledger
        self.boundary = boundary
        self._contracts: dict[str, SkillContract] = {}
        self._handlers: dict[str, Callable[[dict[str, Any]], SkillResult]] = {}
        self._register_builtins()

    def register(self, contract: SkillContract, handler: Callable[[dict[str, Any]], SkillResult] | None = None) -> None:
        if contract.name in self._contracts:
            raise ValueError(f"Skill already registered: {contract.name}")
        self._contracts[contract.name] = contract
        if handler is not None:
            self._handlers[contract.name] = handler

    def register_external(self, contract: SkillContract, adapter: ExternalSkillAdapter) -> None:
        if not contract.name.startswith("external."):
            raise ValueError("External skills must use the external. namespace")
        self.register(contract, adapter)

    def contract(self, name: str) -> SkillContract:
        if name not in self._contracts:
            raise KeyError(f"Unknown skill: {name}")
        return self._contracts[name]

    def has_handler(self, name: str) -> bool:
        return name in self._handlers

    def has_contract(self, name: str) -> bool:
        return name in self._contracts

    def bind_handler(self, name: str, handler: Callable[[dict[str, Any]], SkillResult]) -> None:
        if name not in self._contracts:
            raise KeyError(f"Unknown skill contract: {name}")
        self._handlers[name] = handler

    def list_contracts(self) -> list[dict[str, Any]]:
        return [self._contracts[name].as_dict() | {"available": self.has_handler(name)} for name in sorted(self._contracts)]

    def invoke(self, name: str, args: dict[str, Any]) -> SkillResult:
        contract = self.contract(name)
        missing = [key for key in contract.required_args if args.get(key) is None]
        if missing:
            return SkillResult(False, error=f"Missing required arguments: {', '.join(missing)}", waiting_input=True)
        handler = self._handlers.get(name)
        if handler is None:
            return SkillResult(False, error=f"No adapter is installed for {name}", waiting_input=True)
        return handler(dict(args))

    def _register_builtins(self) -> None:
        self.register(
            SkillContract("workspace.inspect", "Inventory a confined workspace", RiskLevel.LOW, False, True, default_obligations=({"type": "output_key", "key": "file_count"},)),
            self._workspace_inspect,
        )
        self.register(
            SkillContract("file.read_text", "Read UTF-8 text from the workspace", RiskLevel.LOW, False, True, ("path",), ({"type": "output_key", "key": "text"},)),
            self._file_read,
        )
        self.register(
            SkillContract("file.write_text", "Create a UTF-8 text file inside the workspace", RiskLevel.LOW, True, True, ("path", "text"), ({"type": "path_exists_from_arg", "arg": "path"}, {"type": "output_key", "key": "sha256"})),
            self._file_write,
        )
        self.register(
            SkillContract("file.copy", "Copy one file inside the workspace", RiskLevel.MEDIUM, True, True, ("source", "destination"), ({"type": "path_exists_from_arg", "arg": "destination"}, {"type": "output_values_equal", "left": "source_sha256", "right": "destination_sha256"})),
            self._file_copy,
        )
        self.register(
            SkillContract("file.hash", "Hash one workspace file", RiskLevel.LOW, False, True, ("path",), ({"type": "output_key", "key": "sha256"},)),
            self._file_hash,
        )
        self.register(
            SkillContract("file.list", "List a workspace directory", RiskLevel.LOW, False, True, ("path",), ({"type": "output_key", "key": "entries"},)),
            self._file_list,
        )
        self.register(
            SkillContract("backup.copy_tree", "Copy a bounded directory tree inside the workspace", RiskLevel.MEDIUM, True, True, ("source", "destination"), ({"type": "output_value", "key": "verified", "equals": True},)),
            self._backup_tree,
        )
        self.register(
            SkillContract("git.status", "Read repository status without a shell", RiskLevel.LOW, False, True, default_obligations=({"type": "output_key", "key": "status"},)),
            self._git_status,
        )
        self.register(
            SkillContract("git.diff", "Read repository diff without a shell", RiskLevel.LOW, False, True, default_obligations=({"type": "output_key", "key": "diff"},)),
            self._git_diff,
        )
        self.register(
            SkillContract("code.parse_python", "Parse Python source without executing it", RiskLevel.LOW, False, True, ("path",), ({"type": "output_value", "key": "valid", "equals": True},)),
            self._parse_python,
        )
        self.register(
            SkillContract("code.detect_project", "Detect project metadata and test configuration", RiskLevel.LOW, False, True, default_obligations=({"type": "output_key", "key": "project_type"},)),
            self._detect_project,
        )
        self.register(
            SkillContract("container.prepare_execution", "Stage an immutable container execution manifest", RiskLevel.HIGH, True, False, ("spec_path",), ({"type": "output_key", "key": "execution_run_id"},), requires_container=True),
            self._prepare_execution,
        )
        self.register(
            SkillContract(
                "coding.start_repair",
                "Create a durable native coding session; candidate generation remains a replaceable provider",
                RiskLevel.MEDIUM,
                True,
                True,
                ("repository", "goal", "test_spec_path"),
                ({"type": "output_key", "key": "coding_session_id"},),
            ),
            self._start_coding_repair,
        )
        self.register(
            SkillContract("external.propose_patch", "Request a patch from a replaceable proposal provider", RiskLevel.MEDIUM, False, True, ("goal",), ({"type": "output_key", "key": "patch"},)),
        )
        self.register(
            SkillContract("external.apply_patch", "Apply an externally proposed patch under a verifier", RiskLevel.HIGH, True, True, ("source_step",), ({"type": "output_key", "key": "changed_files"},)),
        )

    def _workspace_inspect(self, args: dict[str, Any]) -> SkillResult:
        root = self.boundary.resolve(args.get("path", "."), must_exist=True)
        if not root.is_dir():
            return SkillResult(False, error="Inspection target is not a directory")
        files: list[dict[str, Any]] = []
        total = 0
        truncated = False
        for path in sorted(root.rglob("*")):
            if ".git" in path.parts or path.is_symlink() or not path.is_file():
                continue
            size = path.stat().st_size
            total += size
            if len(files) < 500:
                files.append({"path": self.boundary.relative(path), "size": size})
            else:
                truncated = True
        return SkillResult(True, {"path": self.boundary.relative(root), "file_count": len(files), "total_bytes": total, "files": files, "truncated": truncated})

    def _file_read(self, args: dict[str, Any]) -> SkillResult:
        path = self.boundary.resolve(args["path"], must_exist=True)
        if not path.is_file() or path.is_symlink():
            return SkillResult(False, error="Target is not a regular file")
        data = path.read_bytes()
        if len(data) > 2 * 1024 * 1024:
            return SkillResult(False, error="Text read exceeds 2 MiB")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return SkillResult(False, error="File is not UTF-8 text")
        return SkillResult(True, {"path": self.boundary.relative(path), "text": text, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})

    def _file_write(self, args: dict[str, Any]) -> SkillResult:
        path = self.boundary.resolve(args["path"])
        if path.exists() and not bool(args.get("overwrite", False)):
            return SkillResult(False, error="Destination already exists and overwrite is false")
        path.parent.mkdir(parents=True, exist_ok=True)
        text = str(args["text"])
        data = text.encode("utf-8")
        if len(data) > 2 * 1024 * 1024:
            return SkillResult(False, error="Write exceeds 2 MiB")
        path.write_bytes(data)
        return SkillResult(True, {"path": self.boundary.relative(path), "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})

    def _file_copy(self, args: dict[str, Any]) -> SkillResult:
        source = self.boundary.resolve(args["source"], must_exist=True)
        destination = self.boundary.resolve(args["destination"])
        if not source.is_file() or source.is_symlink():
            return SkillResult(False, error="Source is not a regular file")
        if destination.exists() and not bool(args.get("overwrite", False)):
            return SkillResult(False, error="Destination exists and overwrite is false")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        destination_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
        return SkillResult(True, {"source": self.boundary.relative(source), "destination": self.boundary.relative(destination), "source_sha256": source_hash, "destination_sha256": destination_hash})

    def _file_hash(self, args: dict[str, Any]) -> SkillResult:
        path = self.boundary.resolve(args["path"], must_exist=True)
        if not path.is_file() or path.is_symlink():
            return SkillResult(False, error="Target is not a regular file")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return SkillResult(True, {"path": self.boundary.relative(path), "sha256": digest, "size": path.stat().st_size})

    def _file_list(self, args: dict[str, Any]) -> SkillResult:
        path = self.boundary.resolve(args["path"], must_exist=True)
        if not path.is_dir():
            return SkillResult(False, error="Target is not a directory")
        entries = []
        for child in sorted(path.iterdir(), key=lambda p: p.name.casefold())[:1000]:
            if child.is_symlink():
                kind = "symlink"
            elif child.is_dir():
                kind = "directory"
            else:
                kind = "file"
            entries.append({"name": child.name, "kind": kind, "size": child.stat().st_size if child.is_file() else None})
        return SkillResult(True, {"path": self.boundary.relative(path), "entries": entries})

    def _backup_tree(self, args: dict[str, Any]) -> SkillResult:
        source = self.boundary.resolve(args["source"], must_exist=True)
        destination = self.boundary.resolve(args["destination"])
        if not source.is_dir() or source.is_symlink():
            return SkillResult(False, error="Backup source must be a regular directory")
        if destination.exists() and not bool(args.get("overwrite", False)):
            return SkillResult(False, error="Backup destination exists and overwrite is false")
        max_files = int(args.get("max_files", 1000))
        max_bytes = int(args.get("max_bytes", 100_000_000))
        manifest: list[dict[str, Any]] = []
        total = 0
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                return SkillResult(False, error=f"Symlink not allowed in backup source: {self.boundary.relative(path)}")
            if not path.is_file():
                continue
            total += path.stat().st_size
            if len(manifest) + 1 > max_files or total > max_bytes:
                return SkillResult(False, error="Backup exceeds configured bounds")
            rel = path.relative_to(source)
            manifest.append({"path": rel.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size})
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        for item in manifest:
            src = source / item["path"]
            dst = destination / item["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        verified = all(hashlib.sha256((destination / item["path"]).read_bytes()).hexdigest() == item["sha256"] for item in manifest)
        return SkillResult(True, {"source": self.boundary.relative(source), "destination": self.boundary.relative(destination), "file_count": len(manifest), "total_bytes": total, "manifest": manifest, "verified": verified})

    def _run_git(self, args: list[str], path: Path) -> SkillResult:
        if shutil.which("git") is None:
            return SkillResult(False, error="git is not installed")
        proc = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=20, shell=False)
        if proc.returncode != 0:
            return SkillResult(False, outputs={"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}, error=proc.stderr.strip() or "git command failed")
        return SkillResult(True, {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})

    def _git_status(self, args: dict[str, Any]) -> SkillResult:
        path = self.boundary.resolve(args.get("path", "."), must_exist=True)
        result = self._run_git(["status", "--porcelain=v1", "--branch"], path)
        if result.ok:
            result.outputs["status"] = result.outputs.pop("stdout")
        return result

    def _git_diff(self, args: dict[str, Any]) -> SkillResult:
        path = self.boundary.resolve(args.get("path", "."), must_exist=True)
        result = self._run_git(["diff", "--no-ext-diff", "--"], path)
        if result.ok:
            result.outputs["diff"] = result.outputs.pop("stdout")
        return result

    def _parse_python(self, args: dict[str, Any]) -> SkillResult:
        path = self.boundary.resolve(args["path"], must_exist=True)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            return SkillResult(False, {"valid": False}, f"{type(exc).__name__}: {exc}")
        nodes = sum(1 for _ in ast.walk(tree))
        return SkillResult(True, {"path": self.boundary.relative(path), "valid": True, "ast_nodes": nodes})

    def _detect_project(self, args: dict[str, Any]) -> SkillResult:
        path = self.boundary.resolve(args.get("path", "."), must_exist=True)
        markers = {
            "python": ["pyproject.toml", "setup.py", "requirements.txt"],
            "node": ["package.json"],
            "rust": ["Cargo.toml"],
            "go": ["go.mod"],
        }
        found = {kind: [name for name in names if (path / name).exists()] for kind, names in markers.items()}
        kinds = [kind for kind, names in found.items() if names]
        project_type = kinds[0] if len(kinds) == 1 else ("mixed" if kinds else "unknown")
        test_hints = []
        if "python" in kinds:
            test_hints.extend(["python -m pytest -q", "python -m unittest"])
        if "node" in kinds:
            test_hints.append("npm test")
        return SkillResult(True, {"project_type": project_type, "markers": found, "test_command_hints": test_hints})

    def _start_coding_repair(self, args: dict[str, Any]) -> SkillResult:
        from .coding import CodingRuntime, CodingTestSpec

        spec_path = self.boundary.resolve(args["test_spec_path"], must_exist=True)
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return SkillResult(False, error="Coding test spec must be a JSON object")
        runtime = CodingRuntime(self.ledger, self.boundary.root)
        session = runtime.start_session(
            args.get("repository", "."),
            str(args["goal"]),
            test_spec=CodingTestSpec.from_dict(raw),
            allowed_paths=list(args.get("allowed_paths", ["."])),
            max_candidates=int(args.get("max_candidates", 4)),
            actor="computer_agent",
        )
        return SkillResult(
            True,
            {
                "coding_session_id": session["id"],
                "session_hash": session["session_hash"],
                "next_action": "supply patch candidates through a registered provider or coding-add-candidate",
            },
        )

    def _prepare_execution(self, args: dict[str, Any]) -> SkillResult:
        spec_path = self.boundary.resolve(args["spec_path"], must_exist=True)
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        spec = ContainerExecutionSpec.from_dict(raw, base_dir=spec_path.parent)
        submitted = self.ledger.executions.submit(spec, actor="computer_agent", actor_role=ActorRole.TOOL)
        return SkillResult(
            True,
            {
                "execution_run_id": submitted["id"],
                "execution_status": submitted["status"],
                "manifest_hash": submitted["manifest_hash"],
                "requires_human_approval": True,
            },
            waiting_approval=True,
        )


class ComputerAgentRuntime:
    """Durable, capability-based computer agent core.

    The runtime understands a bounded goal language, compiles goals into typed
    plans, executes only registered skills, binds approvals to exact arguments,
    verifies proof obligations, records receipts, and resumes without rerunning
    succeeded steps.
    """

    def __init__(self, ledger: "EpistemicLedger", workspace: str | Path | None = None):
        self.ledger = ledger
        self.boundary = WorkspaceBoundary(workspace or (ledger.db.path.parent / "computer_workspace"))
        from .support import SupportEngine

        self.support = SupportEngine(ledger)
        self.compiler = ComputerGoalCompiler()
        self.registry = ComputerSkillRegistry(ledger, self.boundary)
        self.planner = ComputerPlanner(self.registry)

    def compile_goal(self, utterance: str) -> dict[str, Any]:
        return self.compiler.compile(utterance).as_dict()

    def create_goal(
        self,
        utterance: str,
        *,
        autonomy_mode: AutonomyMode | str = AutonomyMode.VERIFIED,
        actor: str = "user",
    ) -> dict[str, Any]:
        mode = AutonomyMode(autonomy_mode)
        spec = self.compiler.compile(utterance)
        goal_id = new_id("cgl")
        now = utcnow()
        status = ComputerGoalStatus.PROPOSED if spec.ready else ComputerGoalStatus.NEEDS_CLARIFICATION
        self.ledger.db.conn.execute(
            """INSERT INTO computer_goals
               (id, utterance, goal_type, structured_json, workspace, autonomy_mode,
                status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (goal_id, utterance, spec.goal_type.value, stable_json(spec.as_dict()), str(self.boundary.root), mode.value, status.value, now, now),
        )
        self.ledger._event(
            "computer_goal",
            goal_id,
            "COMPUTER_GOAL_COMPILED",
            {"goal": spec.as_dict(), "autonomy_mode": mode.value, "workspace": str(self.boundary.root)},
            actor,
            ActorRole.HUMAN,
        )
        self.ledger.db.conn.commit()
        return self.get_goal(goal_id)

    def plan_goal(self, goal_id: str) -> dict[str, Any]:
        goal_row = self._goal_row(goal_id)
        spec = self._spec_from_goal(goal_row)
        plan = self.planner.plan(spec)
        revision_row = self.ledger.db.conn.execute(
            "SELECT COALESCE(MAX(revision), 0) AS revision FROM computer_plans WHERE goal_id = ?",
            (goal_id,),
        ).fetchone()
        revision = int(revision_row["revision"]) + 1
        plan_id = new_id("cpl")
        now = utcnow()
        payload = plan.as_dict()
        plan_hash = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
        status = "needs_configuration" if plan.configuration_needed else "pending"
        self.ledger.db.conn.execute(
            """INSERT INTO computer_plans
               (id, goal_id, revision, status, plan_json, plan_hash, rationale_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (plan_id, goal_id, revision, status, stable_json(payload), plan_hash, stable_json(plan.rationale), now, now),
        )
        for sequence, step in enumerate(plan.steps):
            contract = self.registry.contract(step.skill)
            risk = step.risk or contract.risk
            args_hash = hashlib.sha256(stable_json(step.args).encode("utf-8")).hexdigest()
            self.ledger.db.conn.execute(
                """INSERT INTO computer_steps
                   (id, plan_id, step_key, sequence, skill_name, args_json, args_hash,
                    depends_on_json, obligations_json, required_claims_json, risk,
                    status, attempts, max_attempts, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
                (
                    new_id("cst"), plan_id, step.key, sequence, step.skill,
                    stable_json(step.args), args_hash, stable_json(step.depends_on),
                    stable_json(step.obligations), stable_json(step.required_claims),
                    risk.value, ComputerStepStatus.PENDING.value, max(1, step.max_attempts), now, now,
                ),
            )
        goal_status = ComputerGoalStatus.NEEDS_CLARIFICATION.value if plan.configuration_needed else ComputerGoalStatus.PLANNED.value
        self.ledger.db.conn.execute(
            "UPDATE computer_goals SET status = ?, updated_at = ? WHERE id = ?",
            (goal_status, now, goal_id),
        )
        self.ledger._event(
            "computer_plan",
            plan_id,
            "COMPUTER_PLAN_CREATED",
            {"goal_id": goal_id, "revision": revision, "plan_hash": plan_hash, "configuration_needed": plan.configuration_needed},
            "computer_agent",
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()
        return self.get_plan(plan_id)

    def request_approval(self, step_id: str) -> dict[str, Any]:
        step = self._step_row(step_id)
        if step["status"] == ComputerStepStatus.SUCCEEDED.value:
            raise ValueError("Succeeded steps cannot be approved again")
        approval_id = new_id("cap")
        self.ledger.db.conn.execute(
            """INSERT INTO computer_approvals
               (id, step_id, args_hash, status, created_at)
               VALUES (?, ?, ?, 'pending', ?)""",
            (approval_id, step_id, step["args_hash"], utcnow()),
        )
        self._set_step_status(step_id, ComputerStepStatus.WAITING_APPROVAL)
        self.ledger.db.conn.commit()
        return self.get_approval(approval_id)

    def approve(self, approval_id: str, *, reviewer: str, rationale: str) -> dict[str, Any]:
        if not reviewer.strip() or not rationale.strip():
            raise ValueError("Reviewer and rationale are required")
        row = self.ledger.db.conn.execute("SELECT * FROM computer_approvals WHERE id = ?", (approval_id,)).fetchone()
        if row is None or row["status"] != "pending":
            raise ValueError("Approval not found or already decided")
        step = self._step_row(row["step_id"])
        if row["args_hash"] != step["args_hash"]:
            raise ValueError("Approval no longer matches the exact step arguments")
        now = utcnow()
        self.ledger.db.conn.execute(
            "UPDATE computer_approvals SET status = 'approved', reviewer = ?, rationale = ?, decided_at = ? WHERE id = ?",
            (reviewer, rationale, now, approval_id),
        )
        self.ledger.db.conn.execute(
            "UPDATE computer_steps SET status = ?, error = NULL, updated_at = ? WHERE id = ?",
            (ComputerStepStatus.PENDING.value, now, step["id"]),
        )
        self.ledger._event(
            "computer_step", step["id"], "COMPUTER_STEP_APPROVED",
            {"approval_id": approval_id, "args_hash": step["args_hash"], "rationale": rationale}, reviewer, ActorRole.HUMAN,
        )
        self.ledger.db.conn.commit()
        return self.get_approval(approval_id)

    def run_until_blocked(self, plan_id: str, *, max_steps: int = 100) -> dict[str, Any]:
        plan = self._plan_row(plan_id)
        if not self.verify_plan(plan_id):
            raise ValueError("Computer plan integrity verification failed")
        if plan["status"] == "needs_configuration":
            return self.get_plan(plan_id)
        self.ledger.db.conn.execute("UPDATE computer_plans SET status = 'running', updated_at = ? WHERE id = ?", (utcnow(), plan_id))
        self.ledger.db.conn.execute("UPDATE computer_goals SET status = ?, updated_at = ? WHERE id = ?", (ComputerGoalStatus.RUNNING.value, utcnow(), plan["goal_id"]))
        self.ledger.db.conn.commit()
        count = 0
        while count < max_steps:
            count += 1
            runnable = self._next_runnable_step(plan_id)
            if runnable is None:
                self._refresh_plan_status(plan_id)
                break
            outcome = self.execute_step(runnable["id"])
            if outcome["status"] in {ComputerStepStatus.WAITING_APPROVAL.value, ComputerStepStatus.WAITING_INPUT.value, ComputerStepStatus.FAILED.value, ComputerStepStatus.BLOCKED.value}:
                self._refresh_plan_status(plan_id)
                break
        return self.get_plan(plan_id)

    def execute_step(self, step_id: str) -> dict[str, Any]:
        step = self._step_row(step_id)
        if not self.verify_plan(step["plan_id"]):
            raise ValueError("Computer plan integrity verification failed")
        plan = self._plan_row(step["plan_id"])
        goal = self._goal_row(plan["goal_id"])
        mode = AutonomyMode(goal["autonomy_mode"])
        contract = self.registry.contract(step["skill_name"])
        args = json.loads(step["args_json"])
        obligations = json.loads(step["obligations_json"])
        required_claims = json.loads(step["required_claims_json"])

        if step["status"] == ComputerStepStatus.SUCCEEDED.value:
            return self.get_step(step_id)
        for dependency in json.loads(step["depends_on_json"]):
            dep = self.ledger.db.conn.execute("SELECT status FROM computer_steps WHERE plan_id = ? AND step_key = ?", (step["plan_id"], dependency)).fetchone()
            if dep is None or dep["status"] != ComputerStepStatus.SUCCEEDED.value:
                self._set_step_status(step_id, ComputerStepStatus.BLOCKED, f"Dependency {dependency} has not succeeded")
                return self.get_step(step_id)
        for claim_id in required_claims:
            state = self.support.evaluate(claim_id).state
            if state not in {SupportState.SUPPORTED, SupportState.CHALLENGED}:
                self._set_step_status(step_id, ComputerStepStatus.BLOCKED, f"Required claim {claim_id} is {state.value}")
                return self.get_step(step_id)

        if mode == AutonomyMode.OBSERVE and contract.mutates:
            self._set_step_status(step_id, ComputerStepStatus.BLOCKED, "Observe mode forbids mutation")
            return self.get_step(step_id)
        if mode == AutonomyMode.PROPOSE:
            self._set_step_status(step_id, ComputerStepStatus.WAITING_APPROVAL, "Propose mode never executes steps")
            return self.get_step(step_id)

        approval_required = contract.risk in {RiskLevel.MEDIUM, RiskLevel.HIGH} or contract.external_side_effect
        if approval_required:
            approval = self.ledger.db.conn.execute(
                """SELECT * FROM computer_approvals
                   WHERE step_id = ? AND args_hash = ? AND status = 'approved' AND consumed_at IS NULL
                   ORDER BY created_at DESC LIMIT 1""",
                (step_id, step["args_hash"]),
            ).fetchone()
            if approval is None:
                existing = self.ledger.db.conn.execute(
                    "SELECT id FROM computer_approvals WHERE step_id = ? AND args_hash = ? AND status = 'pending'",
                    (step_id, step["args_hash"]),
                ).fetchone()
                if existing is None:
                    self.request_approval(step_id)
                else:
                    self._set_step_status(step_id, ComputerStepStatus.WAITING_APPROVAL)
                return self.get_step(step_id)
            self.ledger.db.conn.execute("UPDATE computer_approvals SET consumed_at = ? WHERE id = ?", (utcnow(), approval["id"]))

        self._set_step_status(step_id, ComputerStepStatus.RUNNING)
        started = utcnow()
        try:
            result = self.registry.invoke(step["skill_name"], args)
        except Exception as exc:  # defensive capability boundary
            result = SkillResult(False, error=f"{type(exc).__name__}: {exc}")
        checks = [self._verify_obligation(obligation, args, result.outputs) for obligation in obligations]
        checks_ok = bool(obligations) and all(item["ok"] for item in checks)
        ok = result.ok and checks_ok and not result.waiting_input
        completed = utcnow()
        receipt_payload = {
            "step_id": step_id,
            "skill": step["skill_name"],
            "args_hash": step["args_hash"],
            "ok": ok,
            "outputs": result.outputs,
            "checks": checks,
            "error": result.error,
            "started_at": started,
            "completed_at": completed,
        }
        receipt_hash = hashlib.sha256(stable_json(receipt_payload).encode("utf-8")).hexdigest()
        self.ledger.db.conn.execute(
            """INSERT INTO computer_receipts
               (id, step_id, ok, outputs_json, checks_json, error, receipt_hash,
                started_at, completed_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (new_id("crc"), step_id, int(ok), stable_json(result.outputs), stable_json(checks), result.error, receipt_hash, started, completed, completed),
        )
        attempts = int(step["attempts"]) + 1
        if result.waiting_input:
            status = ComputerStepStatus.WAITING_INPUT
        elif result.waiting_approval:
            status = ComputerStepStatus.WAITING_APPROVAL
        elif ok:
            status = ComputerStepStatus.SUCCEEDED
        elif attempts < int(step["max_attempts"]):
            status = ComputerStepStatus.PENDING
        else:
            status = ComputerStepStatus.FAILED
        self.ledger.db.conn.execute(
            "UPDATE computer_steps SET status = ?, attempts = ?, error = ?, updated_at = ? WHERE id = ?",
            (status.value, attempts, result.error if not ok else None, completed, step_id),
        )
        self.ledger._event(
            "computer_step", step_id, "COMPUTER_STEP_EXECUTED",
            {"skill": step["skill_name"], "ok": ok, "status": status.value, "receipt_hash": receipt_hash},
            "computer_agent", ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()
        return self.get_step(step_id)

    def continue_container_step(
        self,
        step_id: str,
        *,
        reviewer: str,
        rationale: str,
        engine: "OCIEngine | None" = None,
    ) -> dict[str, Any]:
        receipt = self.ledger.db.conn.execute(
            "SELECT * FROM computer_receipts WHERE step_id = ? ORDER BY created_at DESC LIMIT 1",
            (step_id,),
        ).fetchone()
        if receipt is None:
            raise ValueError("The step has not prepared an execution run")
        outputs = json.loads(receipt["outputs_json"])
        run_id = outputs.get("execution_run_id")
        if not run_id:
            raise ValueError("The latest step receipt has no execution run")
        run = self.ledger.executions.get(run_id)
        if run["status"] == "waiting_approval":
            self.ledger.executions.approve(run_id, reviewer=reviewer, rationale=rationale)
        run = self.ledger.executions.execute(run_id, engine=engine)
        if run["status"] == "succeeded" and self.ledger.executions.verify_receipt(run_id):
            self._set_step_status(step_id, ComputerStepStatus.SUCCEEDED)
        else:
            self._set_step_status(step_id, ComputerStepStatus.FAILED, run.get("error") or "Container execution failed verification")
        self._refresh_plan_status(self._step_row(step_id)["plan_id"])
        return {"step": self.get_step(step_id), "execution": run}

    def recover_interrupted(self) -> dict[str, int]:
        """Return stale RUNNING steps to PENDING after a process or machine restart.

        Succeeded steps and consumed approvals remain untouched. The next run therefore
        resumes from the durable cursor instead of replaying completed side effects.
        """
        now = utcnow()
        rows = self.ledger.db.conn.execute(
            "SELECT id, plan_id FROM computer_steps WHERE status = ?",
            (ComputerStepStatus.RUNNING.value,),
        ).fetchall()
        plan_ids = {row["plan_id"] for row in rows}
        for row in rows:
            self.ledger.db.conn.execute(
                "UPDATE computer_steps SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (
                    ComputerStepStatus.PENDING.value,
                    "Recovered after interrupted execution; proof obligations will be rechecked",
                    now,
                    row["id"],
                ),
            )
            self.ledger._event(
                "computer_step",
                row["id"],
                "COMPUTER_STEP_RECOVERED",
                {"previous_status": ComputerStepStatus.RUNNING.value},
                "computer_agent",
                ActorRole.TOOL,
            )
        for plan_id in plan_ids:
            self.ledger.db.conn.execute(
                "UPDATE computer_plans SET status = 'running', updated_at = ? WHERE id = ?",
                (now, plan_id),
            )
        self.ledger.db.conn.commit()
        return {"recovered_steps": len(rows), "affected_plans": len(plan_ids)}

    def register_external_skill(self, contract: SkillContract, adapter: ExternalSkillAdapter) -> None:
        if self.registry.has_contract(contract.name):
            self.registry.bind_handler(contract.name, adapter)
        else:
            self.registry.register_external(contract, adapter)

    def get_goal(self, goal_id: str) -> dict[str, Any]:
        row = self._goal_row(goal_id)
        result = dict(row)
        result["structured"] = json.loads(result.pop("structured_json"))
        result["plans"] = [item["id"] for item in self.ledger.db.conn.execute("SELECT id FROM computer_plans WHERE goal_id = ? ORDER BY revision", (goal_id,)).fetchall()]
        return result

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        row = self._plan_row(plan_id)
        result = dict(row)
        result["plan"] = json.loads(result.pop("plan_json"))
        result["rationale"] = json.loads(result.pop("rationale_json"))
        result["steps"] = [self._step_dict(item) for item in self.ledger.db.conn.execute("SELECT * FROM computer_steps WHERE plan_id = ? ORDER BY sequence", (plan_id,)).fetchall()]
        return result

    def get_step(self, step_id: str) -> dict[str, Any]:
        return self._step_dict(self._step_row(step_id))

    def get_approval(self, approval_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute("SELECT * FROM computer_approvals WHERE id = ?", (approval_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown computer approval: {approval_id}")
        return dict(row)

    def list_goals(self) -> list[dict[str, Any]]:
        return [self.get_goal(row["id"]) for row in self.ledger.db.conn.execute("SELECT id FROM computer_goals ORDER BY created_at DESC").fetchall()]

    def verify_plan(self, plan_id: str) -> bool:
        row = self._plan_row(plan_id)
        payload = json.loads(row["plan_json"])
        if hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest() != row["plan_hash"]:
            return False
        steps = self.ledger.db.conn.execute(
            "SELECT args_json, args_hash FROM computer_steps WHERE plan_id = ?", (plan_id,)
        ).fetchall()
        return all(
            hashlib.sha256(stable_json(json.loads(step["args_json"])).encode("utf-8")).hexdigest()
            == step["args_hash"]
            for step in steps
        )

    def verify_receipt(self, receipt_id: str) -> bool:
        row = self.ledger.db.conn.execute(
            """SELECT r.*, s.skill_name, s.args_hash
               FROM computer_receipts r JOIN computer_steps s ON s.id = r.step_id
               WHERE r.id = ?""",
            (receipt_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown computer receipt: {receipt_id}")
        payload = {
            "step_id": row["step_id"],
            "skill": row["skill_name"],
            "args_hash": row["args_hash"],
            "ok": bool(row["ok"]),
            "outputs": json.loads(row["outputs_json"]),
            "checks": json.loads(row["checks_json"]),
            "error": row["error"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        }
        return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest() == row["receipt_hash"]

    def machine_state(self, *, goal_id: str | None = None) -> dict[str, Any]:
        result = self.registry.invoke("workspace.inspect", {"path": "."})
        state = {
            "api_version": AGENT_OS_API_VERSION,
            "workspace": str(self.boundary.root),
            "skills": self.registry.list_contracts(),
            "inventory": result.outputs if result.ok else {"error": result.error},
            "container_runtime": self.ledger.executions.runtime_status(),
        }
        digest = hashlib.sha256(stable_json(state).encode("utf-8")).hexdigest()
        snapshot_id = new_id("css")
        self.ledger.db.conn.execute(
            "INSERT INTO computer_state_snapshots (id, goal_id, workspace, state_json, state_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (snapshot_id, goal_id, str(self.boundary.root), stable_json(state), digest, utcnow()),
        )
        self.ledger.db.conn.commit()
        return {"id": snapshot_id, "state_hash": digest, **state}

    def _verify_obligation(self, obligation: dict[str, Any], args: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
        kind = obligation.get("type")
        ok = False
        detail = ""
        try:
            if kind == "output_key":
                key = str(obligation["key"])
                ok = key in outputs
                detail = f"output key {key!r} {'exists' if ok else 'is missing'}"
            elif kind == "output_value":
                key = str(obligation["key"])
                expected = obligation.get("equals")
                ok = outputs.get(key) == expected
                detail = f"{key}={outputs.get(key)!r}; expected {expected!r}"
            elif kind == "path_exists_from_arg":
                path = self.boundary.resolve(args[str(obligation["arg"])])
                ok = path.exists() and not path.is_symlink()
                detail = f"{self.boundary.relative(path)} {'exists' if ok else 'is missing'}"
            elif kind == "output_values_equal":
                left, right = str(obligation["left"]), str(obligation["right"])
                ok = outputs.get(left) == outputs.get(right) and outputs.get(left) is not None
                detail = f"{left} and {right} {'match' if ok else 'do not match'}"
            else:
                detail = f"Unknown obligation type: {kind}"
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            ok = False
        return {"obligation": obligation, "ok": ok, "detail": detail}

    def _next_runnable_step(self, plan_id: str):
        rows = self.ledger.db.conn.execute("SELECT * FROM computer_steps WHERE plan_id = ? ORDER BY sequence", (plan_id,)).fetchall()
        statuses = {row["step_key"]: row["status"] for row in rows}
        for row in rows:
            if row["status"] != ComputerStepStatus.PENDING.value:
                continue
            deps = json.loads(row["depends_on_json"])
            if all(statuses.get(dep) == ComputerStepStatus.SUCCEEDED.value for dep in deps):
                return row
        return None

    def _refresh_plan_status(self, plan_id: str) -> None:
        plan = self._plan_row(plan_id)
        rows = self.ledger.db.conn.execute("SELECT status FROM computer_steps WHERE plan_id = ?", (plan_id,)).fetchall()
        statuses = [row["status"] for row in rows]
        if rows and all(s == ComputerStepStatus.SUCCEEDED.value for s in statuses):
            plan_status, goal_status = "succeeded", ComputerGoalStatus.SUCCEEDED.value
        elif any(s == ComputerStepStatus.FAILED.value for s in statuses):
            plan_status, goal_status = "failed", ComputerGoalStatus.FAILED.value
        elif any(s == ComputerStepStatus.WAITING_INPUT.value for s in statuses):
            plan_status, goal_status = "waiting_input", ComputerGoalStatus.WAITING_INPUT.value
        elif any(s == ComputerStepStatus.WAITING_APPROVAL.value for s in statuses):
            plan_status, goal_status = "waiting_approval", ComputerGoalStatus.WAITING_APPROVAL.value
        elif any(s == ComputerStepStatus.BLOCKED.value for s in statuses):
            plan_status, goal_status = "blocked", ComputerGoalStatus.FAILED.value
        else:
            plan_status, goal_status = "running", ComputerGoalStatus.RUNNING.value
        now = utcnow()
        self.ledger.db.conn.execute("UPDATE computer_plans SET status = ?, updated_at = ? WHERE id = ?", (plan_status, now, plan_id))
        self.ledger.db.conn.execute("UPDATE computer_goals SET status = ?, updated_at = ? WHERE id = ?", (goal_status, now, plan["goal_id"]))
        self.ledger.db.conn.commit()

    def _set_step_status(self, step_id: str, status: ComputerStepStatus, error: str | None = None) -> None:
        self.ledger.db.conn.execute("UPDATE computer_steps SET status = ?, error = ?, updated_at = ? WHERE id = ?", (status.value, error, utcnow(), step_id))
        self.ledger.db.conn.commit()

    def _goal_row(self, goal_id: str):
        row = self.ledger.db.conn.execute("SELECT * FROM computer_goals WHERE id = ?", (goal_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown computer goal: {goal_id}")
        return row

    def _plan_row(self, plan_id: str):
        row = self.ledger.db.conn.execute("SELECT * FROM computer_plans WHERE id = ?", (plan_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown computer plan: {plan_id}")
        return row

    def _step_row(self, step_id: str):
        row = self.ledger.db.conn.execute("SELECT * FROM computer_steps WHERE id = ?", (step_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown computer step: {step_id}")
        return row

    def _spec_from_goal(self, row) -> ComputerGoalSpec:
        raw = json.loads(row["structured_json"])
        return ComputerGoalSpec(
            utterance=raw["utterance"],
            goal_type=ComputerGoalType(raw["goal_type"]),
            parameters=dict(raw.get("parameters", {})),
            constraints=dict(raw.get("constraints", {})),
            success_conditions=list(raw.get("success_conditions", [])),
            ambiguities=list(raw.get("ambiguities", [])),
            parser_trace=list(raw.get("parser_trace", [])),
        )

    def _step_dict(self, row) -> dict[str, Any]:
        result = dict(row)
        for field_name in ("args_json", "depends_on_json", "obligations_json", "required_claims_json"):
            result[field_name.removesuffix("_json")] = json.loads(result.pop(field_name))
        receipts = self.ledger.db.conn.execute("SELECT * FROM computer_receipts WHERE step_id = ? ORDER BY created_at", (row["id"],)).fetchall()
        result["receipts"] = []
        for receipt in receipts:
            item = dict(receipt)
            item["outputs"] = json.loads(item.pop("outputs_json"))
            item["checks"] = json.loads(item.pop("checks_json"))
            result["receipts"].append(item)
        approvals = self.ledger.db.conn.execute("SELECT * FROM computer_approvals WHERE step_id = ? ORDER BY created_at", (row["id"],)).fetchall()
        result["approvals"] = [dict(item) for item in approvals]
        return result
