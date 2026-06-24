from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol

from jsonschema import Draft202012Validator

from .analysis import MetricCondition
from .models import ActorRole, AnalysisOutcome, EvidenceKind, Stance

if TYPE_CHECKING:  # pragma: no cover
    from .ledger import EpistemicLedger


EXECUTION_API_VERSION = "1"
_DIGEST_IMAGE = re.compile(r"^[A-Za-z0-9._/:+-]+@sha256:[0-9a-f]{64}$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SECRET_ENV = re.compile(r"(?:SECRET|TOKEN|PASSWORD|PASSWD|API_KEY|PRIVATE_KEY|CREDENTIAL)", re.IGNORECASE)
_MAX_INLINE_FILE_BYTES = 2 * 1024 * 1024
_DEFAULT_STDIO_LIMIT = 1024 * 1024


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _safe_relative(raw: str) -> str:
    value = raw.replace("\\", "/").strip()
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"Unsafe relative path: {raw!r}")
    if any(part in {"", "/"} for part in path.parts):
        raise ValueError(f"Unsafe relative path: {raw!r}")
    return str(path)


def _extract_json_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split(".") if path else []:
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(path)
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        else:
            raise KeyError(path)
    return current


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    timeout_seconds: int = 120
    memory_mb: int = 512
    cpus: float = 1.0
    pids: int = 128
    tmpfs_mb: int = 64
    stdout_bytes: int = _DEFAULT_STDIO_LIMIT
    stderr_bytes: int = _DEFAULT_STDIO_LIMIT

    def __post_init__(self) -> None:
        if not 1 <= int(self.timeout_seconds) <= 3600:
            raise ValueError("timeout_seconds must be between 1 and 3600")
        if not 64 <= int(self.memory_mb) <= 32768:
            raise ValueError("memory_mb must be between 64 and 32768")
        if not 0.1 <= float(self.cpus) <= 16:
            raise ValueError("cpus must be between 0.1 and 16")
        if not 16 <= int(self.pids) <= 1024:
            raise ValueError("pids must be between 16 and 1024")
        if not 8 <= int(self.tmpfs_mb) <= 1024:
            raise ValueError("tmpfs_mb must be between 8 and 1024")
        if not 1024 <= int(self.stdout_bytes) <= 16 * 1024 * 1024:
            raise ValueError("stdout_bytes must be between 1 KiB and 16 MiB")
        if not 1024 <= int(self.stderr_bytes) <= 16 * 1024 * 1024:
            raise ValueError("stderr_bytes must be between 1 KiB and 16 MiB")

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "ResourceLimits":
        return cls(**dict(value or {}))


@dataclass(frozen=True, slots=True)
class StagedFile:
    target: str
    source: str | Path | None = None
    text: str | None = None
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _safe_relative(self.target))
        if (self.source is None) == (self.text is None):
            raise ValueError("Exactly one of source or text is required for a staged file")
        if self.text is not None and len(self.text.encode("utf-8")) > _MAX_INLINE_FILE_BYTES:
            raise ValueError("Inline staged file exceeds the 2 MiB limit")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StagedFile":
        return cls(
            target=str(value["target"]),
            source=value.get("source"),
            text=value.get("text"),
            media_type=str(value.get("media_type", "application/octet-stream")),
        )


@dataclass(frozen=True, slots=True)
class OutputObligation:
    path: str
    required: bool = True
    media_type: str = "application/octet-stream"
    max_bytes: int = 5 * 1024 * 1024
    sha256: str | None = None
    json_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _safe_relative(self.path))
        if not 1 <= int(self.max_bytes) <= 100 * 1024 * 1024:
            raise ValueError("Output max_bytes must be between 1 byte and 100 MiB")
        if self.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("Output sha256 must be a lowercase 64-character hex digest")
        if self.json_schema is not None:
            Draft202012Validator.check_schema(self.json_schema)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OutputObligation":
        return cls(
            path=str(value["path"]),
            required=bool(value.get("required", True)),
            media_type=str(value.get("media_type", "application/octet-stream")),
            max_bytes=int(value.get("max_bytes", 5 * 1024 * 1024)),
            sha256=value.get("sha256"),
            json_schema=value.get("json_schema"),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionClaimTest:
    claim_id: str
    output_path: str
    metric_path: str
    support_condition: MetricCondition | dict[str, Any]
    refute_condition: MetricCondition | dict[str, Any] | None = None
    confidence: float = 1.0
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("claim_id is required")
        object.__setattr__(self, "output_path", _safe_relative(self.output_path))
        if not self.metric_path:
            raise ValueError("metric_path is required")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "support_condition", MetricCondition.from_value(self.support_condition))
        if self.refute_condition is not None:
            object.__setattr__(
                self, "refute_condition", MetricCondition.from_value(self.refute_condition)
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExecutionClaimTest":
        return cls(
            claim_id=str(value["claim_id"]),
            output_path=str(value["output_path"]),
            metric_path=str(value["metric_path"]),
            support_condition=value["support_condition"],
            refute_condition=value.get("refute_condition"),
            confidence=float(value.get("confidence", 1.0)),
            rationale=str(value.get("rationale", "")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "output_path": self.output_path,
            "metric_path": self.metric_path,
            "support_condition": asdict(self.support_condition),
            "refute_condition": asdict(self.refute_condition) if self.refute_condition else None,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class ContainerExecutionSpec:
    name: str
    image: str
    command: tuple[str, ...]
    code_files: tuple[StagedFile, ...]
    input_files: tuple[StagedFile, ...] = ()
    outputs: tuple[OutputObligation, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    required_claims: tuple[str, ...] = ()
    claim_tests: tuple[ExecutionClaimTest, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    network: bool = False
    user: str = "65532:65532"
    allow_unlisted_outputs: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Execution name is required")
        if not _DIGEST_IMAGE.fullmatch(self.image):
            raise ValueError("Container image must be pinned as name@sha256:<64 lowercase hex>")
        if not self.command or any(not isinstance(x, str) or not x for x in self.command):
            raise ValueError("command must be a non-empty string array")
        if len(self.command) > 128:
            raise ValueError("command has too many arguments")
        if not self.code_files:
            raise ValueError("At least one code file is required")
        if self.network:
            raise ValueError("Network access is disabled in execution API v1")
        if self.user in {"0", "0:0", "root", "root:root"}:
            raise ValueError("Container must run as a non-root user")
        targets = [item.target for item in (*self.code_files, *self.input_files)]
        if len(targets) != len(set(targets)):
            raise ValueError("Staged file targets must be unique")
        output_paths = [item.path for item in self.outputs]
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("Output obligation paths must be unique")
        for key, value in self.environment.items():
            if not _ENV_NAME.fullmatch(str(key)):
                raise ValueError(f"Unsafe environment variable name: {key!r}")
            if _SECRET_ENV.search(str(key)):
                raise ValueError(
                    f"Secret-like environment variable {key!r} is forbidden because manifests are auditable"
                )
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError(f"Invalid environment value for {key}")

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, base_dir: str | Path | None = None) -> "ContainerExecutionSpec":
        base = Path(base_dir).resolve() if base_dir is not None else None

        def staged(items: list[dict[str, Any]]) -> tuple[StagedFile, ...]:
            result = []
            for raw in items:
                item = dict(raw)
                if item.get("source") is not None and base is not None:
                    source = Path(str(item["source"]))
                    if not source.is_absolute():
                        item["source"] = str((base / source).resolve())
                result.append(StagedFile.from_dict(item))
            return tuple(result)

        return cls(
            name=str(value["name"]),
            image=str(value["image"]),
            command=tuple(str(x) for x in value["command"]),
            code_files=staged(list(value.get("code_files", []))),
            input_files=staged(list(value.get("input_files", []))),
            outputs=tuple(OutputObligation.from_dict(x) for x in value.get("outputs", [])),
            environment={str(k): str(v) for k, v in dict(value.get("environment", {})).items()},
            limits=ResourceLimits.from_dict(value.get("limits")),
            required_claims=tuple(str(x) for x in value.get("required_claims", [])),
            claim_tests=tuple(ExecutionClaimTest.from_dict(x) for x in value.get("claim_tests", [])),
            metadata=dict(value.get("metadata", {})),
            network=bool(value.get("network", False)),
            user=str(value.get("user", "65532:65532")),
            allow_unlisted_outputs=bool(value.get("allow_unlisted_outputs", False)),
        )


@dataclass(slots=True)
class EngineResult:
    engine: str
    invoked_command: list[str]
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    started_at: str = field(default_factory=_utcnow)
    completed_at: str = field(default_factory=_utcnow)
    error: str | None = None


class OCIEngine(Protocol):
    name: str

    def available(self) -> bool: ...

    def run(self, manifest: dict[str, Any], run_root: Path) -> EngineResult: ...


class CliOCIEngine:
    """Docker/Podman CLI adapter with a fixed, non-shell invocation."""

    def __init__(self, executable: str):
        if executable not in {"docker", "podman"}:
            raise ValueError("OCI executable must be docker or podman")
        self.executable = executable
        self.name = executable

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def build_command(self, manifest: dict[str, Any], run_root: Path) -> tuple[list[str], str]:
        run_name = "orbita-" + manifest["run_id"].replace("_", "-")[:48]
        limits = manifest["limits"]
        code_dir = (run_root / "code").resolve()
        input_dir = (run_root / "input").resolve()
        output_dir = (run_root / "output").resolve()
        command = [
            self.executable,
            "run",
            "--rm",
            "--name",
            run_name,
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            str(limits["pids"]),
            "--memory",
            f"{limits['memory_mb']}m",
            "--cpus",
            str(limits["cpus"]),
            "--user",
            manifest["user"],
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,nodev,size={limits['tmpfs_mb']}m",
            "--mount",
            f"type=bind,src={code_dir},dst=/workspace/code,readonly",
            "--mount",
            f"type=bind,src={input_dir},dst=/workspace/input,readonly",
            "--mount",
            f"type=bind,src={output_dir},dst=/workspace/output",
            "--workdir",
            "/workspace/code",
        ]
        for key, value in sorted(manifest["environment"].items()):
            command.extend(["--env", f"{key}={value}"])
        command.append(manifest["image"])
        command.extend(manifest["command"])
        return command, run_name

    def run(self, manifest: dict[str, Any], run_root: Path) -> EngineResult:
        if not self.available():
            return EngineResult(
                engine=self.name,
                invoked_command=[],
                exit_code=None,
                timed_out=False,
                stdout="",
                stderr="",
                error=f"{self.executable} is not installed or not on PATH",
            )
        command, run_name = self.build_command(manifest, run_root)
        started = _utcnow()
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            shell=False,
            close_fds=True,
        )
        buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
        truncated = {"stdout": False, "stderr": False}

        def drain(name: str, pipe, limit: int) -> None:
            try:
                while True:
                    chunk = pipe.read(65536)
                    if not chunk:
                        break
                    remaining = limit - len(buffers[name])
                    if remaining > 0:
                        buffers[name].extend(chunk[:remaining])
                    if len(chunk) > max(remaining, 0):
                        truncated[name] = True
            finally:
                pipe.close()

        threads = [
            threading.Thread(
                target=drain,
                args=("stdout", process.stdout, int(manifest["limits"]["stdout_bytes"])),
                daemon=True,
            ),
            threading.Thread(
                target=drain,
                args=("stderr", process.stderr, int(manifest["limits"]["stderr_bytes"])),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        timed_out = False
        error = None
        try:
            process.wait(timeout=int(manifest["limits"]["timeout_seconds"]))
        except subprocess.TimeoutExpired:
            timed_out = True
            error = "Container execution exceeded the declared timeout"
            subprocess.run(
                [self.executable, "kill", run_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
            process.kill()
            process.wait(timeout=15)
        finally:
            for thread in threads:
                thread.join(timeout=15)
        completed = _utcnow()
        return EngineResult(
            engine=self.name,
            invoked_command=command,
            exit_code=process.returncode,
            timed_out=timed_out,
            stdout=bytes(buffers["stdout"]).decode("utf-8", errors="replace"),
            stderr=bytes(buffers["stderr"]).decode("utf-8", errors="replace"),
            stdout_truncated=truncated["stdout"],
            stderr_truncated=truncated["stderr"],
            started_at=started,
            completed_at=completed,
            error=error,
        )


class ContainerExecutionRuntime:
    """Manifest-bound OCI execution with human approval and durable receipts."""

    def __init__(self, ledger: "EpistemicLedger", workspace: str | Path | None = None):
        self.ledger = ledger
        self.workspace = (
            Path(workspace).expanduser().resolve()
            if workspace is not None
            else ledger.db.path.parent / "execution_workspace"
        )
        self.workspace.mkdir(parents=True, exist_ok=True)

    def runtime_status(self) -> dict[str, Any]:
        engines = {name: shutil.which(name) for name in ("docker", "podman")}
        return {
            "api_version": EXECUTION_API_VERSION,
            "engines": {key: bool(value) for key, value in engines.items()},
            "engine_paths": engines,
            "network_policy": "disabled",
            "approval_policy": "human approval bound to exact manifest hash",
            "workspace": str(self.workspace),
        }

    def submit(
        self,
        spec: ContainerExecutionSpec | dict[str, Any],
        *,
        actor: str = "user",
        actor_role: ActorRole = ActorRole.HUMAN,
        parent_run_id: str | None = None,
    ) -> dict[str, Any]:
        if isinstance(spec, dict):
            spec = ContainerExecutionSpec.from_dict(spec)
        for claim_id in spec.required_claims:
            self.ledger._require_claim(claim_id)
        for test in spec.claim_tests:
            self.ledger._require_claim(test.claim_id)
            if test.output_path not in {item.path for item in spec.outputs}:
                raise ValueError(
                    f"Claim test output {test.output_path!r} has no declared output obligation"
                )
        run_id = _new_id("xrn")
        run_root = self.workspace / "runs" / run_id
        for child in ("code", "input", "output"):
            (run_root / child).mkdir(parents=True, exist_ok=False)
        # The empty output bind mount must be writable by the declared non-root
        # container UID. It is a dedicated per-run directory with no input data.
        os.chmod(run_root / "output", 0o777)
        code_artifacts = self._stage_files(run_id, run_root / "code", spec.code_files, "code")
        input_artifacts = self._stage_files(run_id, run_root / "input", spec.input_files, "input")
        independence_payload = {
            "image": spec.image,
            "command": list(spec.command),
            "environment": dict(sorted(spec.environment.items())),
            "code_artifacts": [self._artifact_manifest(x) for x in code_artifacts],
            "input_artifacts": [self._artifact_manifest(x) for x in input_artifacts],
            "output_obligations": [item.as_dict() for item in spec.outputs],
            "claim_tests": [item.as_dict() for item in spec.claim_tests],
        }
        epistemic_independence_hash = _sha256_bytes(
            _stable_json(independence_payload).encode("utf-8")
        )
        manifest = {
            "schema_version": EXECUTION_API_VERSION,
            "run_id": run_id,
            "name": spec.name,
            "image": spec.image,
            "image_digest": spec.image.rsplit("@", 1)[1],
            "command": list(spec.command),
            "environment": dict(sorted(spec.environment.items())),
            "limits": asdict(spec.limits),
            "security": {
                "network": "none",
                "read_only_root": True,
                "capabilities": "drop-all",
                "no_new_privileges": True,
                "non_root_user": spec.user,
                "input_mount": "read-only",
                "code_mount": "read-only",
                "output_mount": "read-write",
                "shell_interpretation": False,
            },
            "user": spec.user,
            "code_artifacts": [self._artifact_manifest(x) for x in code_artifacts],
            "input_artifacts": [self._artifact_manifest(x) for x in input_artifacts],
            "output_obligations": [item.as_dict() for item in spec.outputs],
            "allow_unlisted_outputs": spec.allow_unlisted_outputs,
            "required_claims": list(spec.required_claims),
            "claim_tests": [item.as_dict() for item in spec.claim_tests],
            "metadata": dict(spec.metadata),
            "parent_run_id": parent_run_id,
            "epistemic_independence_hash": epistemic_independence_hash,
        }
        manifest_hash = _sha256_bytes(_stable_json(manifest).encode("utf-8"))
        approval_id = _new_id("xap")
        self.ledger.db.conn.execute(
            """INSERT INTO execution_runs
               (id, parent_run_id, name, status, image_ref, image_digest, command_json,
                manifest_json, manifest_hash, run_root, required_claims_json,
                output_obligations_json, claim_tests_json, metadata_json, created_at)
               VALUES (?, ?, ?, 'waiting_approval', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                parent_run_id,
                spec.name,
                spec.image,
                manifest["image_digest"],
                _stable_json(manifest["command"]),
                _stable_json(manifest),
                manifest_hash,
                str(run_root),
                _stable_json(manifest["required_claims"]),
                _stable_json(manifest["output_obligations"]),
                _stable_json(manifest["claim_tests"]),
                _stable_json(manifest["metadata"]),
                _utcnow(),
            ),
        )
        for artifact in (*code_artifacts, *input_artifacts):
            self._insert_artifact(**artifact)
        self.ledger.db.conn.execute(
            """INSERT INTO execution_approvals
               (id, run_id, manifest_hash, status, created_at)
               VALUES (?, ?, ?, 'pending', ?)""",
            (approval_id, run_id, manifest_hash, _utcnow()),
        )
        self.ledger._event(
            "execution_run",
            run_id,
            "EXECUTION_SUBMITTED",
            {
                "manifest_hash": manifest_hash,
                "image": spec.image,
                "approval_id": approval_id,
                "security": manifest["security"],
            },
            actor,
            actor_role,
        )
        self.ledger.db.conn.commit()
        return self.get(run_id)

    def approve(
        self,
        run_id: str,
        *,
        reviewer: str,
        rationale: str,
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> dict[str, Any]:
        if actor_role != ActorRole.HUMAN:
            raise PermissionError("Container execution approval requires a human actor")
        if not reviewer.strip() or not rationale.strip():
            raise ValueError("Reviewer and rationale are required")
        row = self._run_row(run_id)
        if row["status"] != "waiting_approval":
            raise ValueError(f"Run is not awaiting approval: {row['status']}")
        approval = self.ledger.db.conn.execute(
            """SELECT * FROM execution_approvals
               WHERE run_id = ? AND manifest_hash = ? AND status = 'pending'
               ORDER BY created_at DESC LIMIT 1""",
            (run_id, row["manifest_hash"]),
        ).fetchone()
        if approval is None:
            raise ValueError("No pending approval matches the current manifest")
        now = _utcnow()
        self.ledger.db.conn.execute(
            """UPDATE execution_approvals
               SET status = 'approved', reviewer = ?, rationale = ?, decided_at = ?
               WHERE id = ? AND status = 'pending'""",
            (reviewer, rationale, now, approval["id"]),
        )
        self.ledger.db.conn.execute(
            "UPDATE execution_runs SET status = 'approved' WHERE id = ?",
            (run_id,),
        )
        self.ledger._event(
            "execution_run",
            run_id,
            "EXECUTION_APPROVED",
            {"approval_id": approval["id"], "manifest_hash": row["manifest_hash"], "rationale": rationale},
            reviewer,
            actor_role,
        )
        self.ledger.db.conn.commit()
        return self.get(run_id)

    def reject(
        self,
        run_id: str,
        *,
        reviewer: str,
        rationale: str,
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> dict[str, Any]:
        if actor_role != ActorRole.HUMAN:
            raise PermissionError("Container execution rejection requires a human actor")
        row = self._run_row(run_id)
        approval = self.ledger.db.conn.execute(
            "SELECT * FROM execution_approvals WHERE run_id = ? AND status = 'pending' ORDER BY created_at DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row["status"] != "waiting_approval" or approval is None:
            raise ValueError("Run is not awaiting a decision")
        now = _utcnow()
        self.ledger.db.conn.execute(
            "UPDATE execution_approvals SET status = 'rejected', reviewer = ?, rationale = ?, decided_at = ? WHERE id = ?",
            (reviewer, rationale, now, approval["id"]),
        )
        self.ledger.db.conn.execute("UPDATE execution_runs SET status = 'rejected' WHERE id = ?", (run_id,))
        self.ledger._event(
            "execution_run", run_id, "EXECUTION_REJECTED", {"rationale": rationale}, reviewer, actor_role
        )
        self.ledger.db.conn.commit()
        return self.get(run_id)

    def execute(
        self,
        run_id: str,
        *,
        engine: OCIEngine | None = None,
        actor: str = "execution_runtime",
        actor_role: ActorRole = ActorRole.TOOL,
    ) -> dict[str, Any]:
        row = self._run_row(run_id)
        if row["status"] != "approved":
            raise ValueError(f"Run must be approved before execution; current status is {row['status']}")
        manifest = json.loads(row["manifest_json"])
        if _sha256_bytes(_stable_json(manifest).encode("utf-8")) != row["manifest_hash"]:
            raise ValueError("Stored execution manifest failed its integrity check")
        approval = self.ledger.db.conn.execute(
            """SELECT * FROM execution_approvals
               WHERE run_id = ? AND manifest_hash = ? AND status = 'approved' AND consumed_at IS NULL
               ORDER BY created_at DESC LIMIT 1""",
            (run_id, row["manifest_hash"]),
        ).fetchone()
        if approval is None:
            raise ValueError("No unconsumed approval matches this exact manifest")
        self._verify_required_claims(manifest["required_claims"])
        if not self._verify_staged_artifacts(run_id, roles={"code", "input"}):
            raise ValueError("Code or input artifacts changed after approval")
        selected = engine or self._default_engine()
        if not selected.available():
            raise RuntimeError(f"OCI engine {selected.name!r} is unavailable")
        run_root = Path(row["run_root"])
        output_dir = run_root / "output"
        self._clear_output_dir(output_dir)
        now = _utcnow()
        self.ledger.db.conn.execute(
            "UPDATE execution_approvals SET consumed_at = ? WHERE id = ?",
            (now, approval["id"]),
        )
        self.ledger.db.conn.execute(
            "UPDATE execution_runs SET status = 'running', engine_used = ?, started_at = ? WHERE id = ?",
            (selected.name, now, run_id),
        )
        self.ledger._event(
            "execution_run",
            run_id,
            "EXECUTION_STARTED",
            {"engine": selected.name, "manifest_hash": row["manifest_hash"]},
            actor,
            actor_role,
        )
        self.ledger.db.conn.commit()
        try:
            result = selected.run(manifest, run_root)
        except Exception as exc:  # engine boundary must always produce a durable failure
            result = EngineResult(
                engine=selected.name,
                invoked_command=[],
                exit_code=None,
                timed_out=False,
                stdout="",
                stderr="",
                error=f"{type(exc).__name__}: {exc}",
            )
        checks, output_artifacts = self._verify_outputs(run_id, output_dir, manifest)
        exit_ok = result.exit_code == 0 and not result.timed_out and result.error is None
        all_checks = bool(manifest["output_obligations"]) and all(item["ok"] for item in checks)
        ok = exit_ok and all_checks
        if not manifest["output_obligations"]:
            checks.append({"type": "output_obligations_present", "ok": False, "detail": "No outputs were declared"})
        assessments = self._assess_claims(manifest["claim_tests"], output_dir) if ok else []
        comparison = self._compare_parent(row["parent_run_id"], output_artifacts) if row["parent_run_id"] else {}
        completed = result.completed_at or _utcnow()
        receipt_payload = {
            "schema_version": EXECUTION_API_VERSION,
            "run_id": run_id,
            "parent_run_id": row["parent_run_id"],
            "manifest_hash": row["manifest_hash"],
            "engine": result.engine,
            "image": manifest["image"],
            "invoked_command": result.invoked_command,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
            "engine_error": result.error,
            "checks": checks,
            "output_artifacts": [self._artifact_manifest(x) for x in output_artifacts],
            "assessments": assessments,
            "comparison": comparison,
            "ok": ok,
            "started_at": result.started_at,
            "completed_at": completed,
        }
        receipt_hash = _sha256_bytes(_stable_json(receipt_payload).encode("utf-8"))
        receipt_document = {**receipt_payload, "receipt_hash": receipt_hash}
        receipt_path = run_root / "receipt.json"
        receipt_path.write_text(json.dumps(receipt_document, indent=2, ensure_ascii=False), encoding="utf-8")
        receipt_digest, receipt_size = _sha256_file(receipt_path)
        self._insert_artifact(
            run_id,
            "receipt",
            "receipt.json",
            receipt_path,
            receipt_digest,
            receipt_size,
            "application/json",
        )
        evidence_id = None
        if ok:
            evidence_id = self.ledger.add_evidence(
                f"execution://{run_id}",
                f"Verified container execution receipt {receipt_hash}",
                source_kind=EvidenceKind.CODE_EXECUTION_RECEIPT,
                independence_key=f"execution:{manifest['epistemic_independence_hash']}",
                content=receipt_hash,
                metadata={
                    "run_id": run_id,
                    "manifest_hash": row["manifest_hash"],
                    "receipt_hash": receipt_hash,
                    "image": manifest["image"],
                    "run_metadata": manifest.get("metadata", {}),
                },
                actor=actor,
                actor_role=actor_role,
            )
        self._persist_assessments(run_id, assessments, evidence_id, actor, actor_role)
        error = None
        if not ok:
            if result.error:
                error = result.error
            elif result.timed_out:
                error = "Execution timed out"
            elif result.exit_code != 0:
                error = f"Container exited with code {result.exit_code}"
            else:
                error = "One or more output proof obligations failed"
        self.ledger.db.conn.execute(
            """UPDATE execution_runs
               SET status = ?, exit_code = ?, timed_out = ?, stdout = ?, stderr = ?,
                   checks_json = ?, receipt_json = ?, receipt_hash = ?, evidence_id = ?,
                   comparison_json = ?, error = ?, completed_at = ?
               WHERE id = ?""",
            (
                "succeeded" if ok else "failed",
                result.exit_code,
                int(result.timed_out),
                result.stdout,
                result.stderr,
                _stable_json(checks),
                _stable_json(receipt_payload),
                receipt_hash,
                evidence_id,
                _stable_json(comparison),
                error,
                completed,
                run_id,
            ),
        )
        self.ledger._event(
            "execution_run",
            run_id,
            "EXECUTION_FINALIZED",
            {
                "ok": ok,
                "status": "succeeded" if ok else "failed",
                "receipt_hash": receipt_hash,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "checks_passed": sum(1 for item in checks if item["ok"]),
                "checks_total": len(checks),
            },
            actor,
            actor_role,
        )
        self.ledger.db.conn.commit()
        return self.get(run_id)

    def prepare_reproduction(
        self,
        run_id: str,
        *,
        actor: str = "user",
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> dict[str, Any]:
        original = self.get(run_id)
        if original["status"] != "succeeded":
            raise ValueError("Only a successful execution can be reproduced")
        manifest = original["manifest"]
        source_root = Path(original["run_root"])
        spec = ContainerExecutionSpec(
            name=f"Reproduction of {original['name']}",
            image=manifest["image"],
            command=tuple(manifest["command"]),
            code_files=tuple(
                StagedFile(target=item["relative_path"], source=source_root / "code" / item["relative_path"], media_type=item["media_type"])
                for item in manifest["code_artifacts"]
            ),
            input_files=tuple(
                StagedFile(target=item["relative_path"], source=source_root / "input" / item["relative_path"], media_type=item["media_type"])
                for item in manifest["input_artifacts"]
            ),
            outputs=tuple(OutputObligation.from_dict(item) for item in manifest["output_obligations"]),
            environment=dict(manifest["environment"]),
            limits=ResourceLimits.from_dict(manifest["limits"]),
            required_claims=tuple(manifest["required_claims"]),
            claim_tests=tuple(ExecutionClaimTest.from_dict(item) for item in manifest["claim_tests"]),
            metadata={**manifest["metadata"], "reproduction_of": run_id},
            user=manifest["user"],
            allow_unlisted_outputs=bool(manifest["allow_unlisted_outputs"]),
        )
        return self.submit(spec, actor=actor, actor_role=actor_role, parent_run_id=run_id)

    def get(self, run_id: str) -> dict[str, Any]:
        row = self._run_row(run_id)
        result = dict(row)
        for key in (
            "command_json",
            "manifest_json",
            "required_claims_json",
            "output_obligations_json",
            "claim_tests_json",
            "metadata_json",
            "checks_json",
            "receipt_json",
            "comparison_json",
        ):
            result[key.removesuffix("_json")] = json.loads(result.pop(key) or "{}")
        result["manifest_integrity_valid"] = self.verify_manifest(run_id)
        result["artifact_integrity_valid"] = self.verify_artifacts(run_id)
        result["receipt_integrity_valid"] = self.verify_receipt(run_id)
        result["approval"] = self._approval_dict(run_id)
        result["artifacts"] = [
            self._artifact_row(item)
            for item in self.ledger.db.conn.execute(
                "SELECT * FROM execution_artifacts WHERE run_id = ? ORDER BY role, relative_path",
                (run_id,),
            ).fetchall()
        ]
        result["assessments"] = [
            self._assessment_row(item)
            for item in self.ledger.db.conn.execute(
                "SELECT * FROM execution_claim_assessments WHERE run_id = ? ORDER BY position",
                (run_id,),
            ).fetchall()
        ]
        return result

    def list(self) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT id FROM execution_runs ORDER BY created_at DESC"
        ).fetchall()
        return [self.get(row["id"]) for row in rows]

    def verify_manifest(self, run_id: str) -> bool:
        row = self._run_row(run_id)
        manifest = json.loads(row["manifest_json"])
        return _sha256_bytes(_stable_json(manifest).encode("utf-8")) == row["manifest_hash"]

    def verify_artifacts(self, run_id: str) -> bool:
        rows = self.ledger.db.conn.execute(
            "SELECT path, content_hash, size_bytes FROM execution_artifacts WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        if not rows:
            return False
        for row in rows:
            path = Path(row["path"])
            if not path.is_file() or path.is_symlink():
                return False
            digest, size = _sha256_file(path)
            if digest != row["content_hash"] or size != row["size_bytes"]:
                return False
        return True

    def verify_receipt(self, run_id: str) -> bool | None:
        row = self._run_row(run_id)
        if row["receipt_hash"] is None:
            return None
        payload = json.loads(row["receipt_json"])
        return _sha256_bytes(_stable_json(payload).encode("utf-8")) == row["receipt_hash"]

    def _stage_files(
        self,
        run_id: str,
        root: Path,
        files: tuple[StagedFile, ...],
        role: str,
    ) -> list[dict[str, Any]]:
        artifacts = []
        for item in files:
            destination = root / item.target
            destination.parent.mkdir(parents=True, exist_ok=True)
            if item.text is not None:
                destination.write_text(item.text, encoding="utf-8")
            else:
                source = Path(item.source).expanduser().resolve()
                if not source.is_file() or source.is_symlink():
                    raise FileNotFoundError(f"Staged source must be a regular file: {source}")
                shutil.copyfile(source, destination)
            os.chmod(destination, 0o444)
            digest, size = _sha256_file(destination)
            artifact = {
                "id": _new_id("xar"),
                "run_id": run_id,
                "role": role,
                "relative_path": item.target,
                "path": str(destination),
                "content_hash": digest,
                "size_bytes": size,
                "media_type": item.media_type,
            }
            artifacts.append(artifact)
        return artifacts

    def _insert_artifact(
        self,
        run_id: str,
        role: str,
        relative_path: str,
        path: Path | str,
        content_hash: str,
        size_bytes: int,
        media_type: str,
        id: str | None = None,
    ) -> None:
        self.ledger.db.conn.execute(
            """INSERT INTO execution_artifacts
               (id, run_id, role, relative_path, path, content_hash, size_bytes, media_type, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                id or _new_id("xar"),
                run_id,
                role,
                relative_path,
                str(path),
                content_hash,
                size_bytes,
                media_type,
                _utcnow(),
            ),
        )

    @staticmethod
    def _artifact_manifest(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": item["role"],
            "relative_path": item["relative_path"],
            "content_hash": item["content_hash"],
            "size_bytes": item["size_bytes"],
            "media_type": item["media_type"],
        }

    @staticmethod
    def _artifact_row(row) -> dict[str, Any]:
        return dict(row)

    def _approval_dict(self, run_id: str) -> dict[str, Any] | None:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM execution_approvals WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return dict(row) if row else None

    def _run_row(self, run_id: str):
        row = self.ledger.db.conn.execute(
            "SELECT * FROM execution_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown execution run: {run_id}")
        return row

    def _default_engine(self) -> OCIEngine:
        for name in ("docker", "podman"):
            engine = CliOCIEngine(name)
            if engine.available():
                return engine
        return CliOCIEngine("docker")

    def _verify_required_claims(self, claim_ids: list[str]) -> None:
        from .support import SupportEngine

        support = SupportEngine(self.ledger)
        for claim_id in claim_ids:
            state = support.evaluate(claim_id).state.value
            if state != "supported":
                raise ValueError(f"Required claim {claim_id} is {state}; execution requires supported")

    def _verify_staged_artifacts(self, run_id: str, *, roles: set[str]) -> bool:
        rows = self.ledger.db.conn.execute(
            "SELECT path, content_hash, size_bytes FROM execution_artifacts WHERE run_id = ? AND role IN ({})".format(
                ",".join("?" for _ in roles)
            ),
            (run_id, *sorted(roles)),
        ).fetchall()
        for row in rows:
            path = Path(row["path"])
            if not path.is_file() or path.is_symlink():
                return False
            digest, size = _sha256_file(path)
            if digest != row["content_hash"] or size != row["size_bytes"]:
                return False
        return True

    @staticmethod
    def _clear_output_dir(output_dir: Path) -> None:
        for child in output_dir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    def _verify_outputs(
        self, run_id: str, output_dir: Path, manifest: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        checks: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        declared = {item["path"]: item for item in manifest["output_obligations"]}
        actual_files: set[str] = set()
        for path in output_dir.rglob("*"):
            if path.is_symlink():
                checks.append({"type": "no_symlink_outputs", "path": str(path.relative_to(output_dir)), "ok": False, "detail": "Symlink output is forbidden"})
                continue
            if path.is_file():
                actual_files.add(path.relative_to(output_dir).as_posix())
        if not manifest["allow_unlisted_outputs"]:
            extras = sorted(actual_files - set(declared))
            checks.append({"type": "no_unlisted_outputs", "ok": not extras, "detail": f"unexpected={extras}"})
        for relative, obligation in declared.items():
            path = output_dir / relative
            exists = path.is_file() and not path.is_symlink()
            if obligation["required"]:
                checks.append({"type": "file_exists", "path": relative, "ok": exists, "detail": "exists" if exists else "missing"})
            if not exists:
                continue
            digest, size = _sha256_file(path)
            size_ok = size <= int(obligation["max_bytes"])
            checks.append({"type": "max_bytes", "path": relative, "ok": size_ok, "detail": f"size={size}, limit={obligation['max_bytes']}"})
            if obligation.get("sha256"):
                checks.append({"type": "sha256_equals", "path": relative, "ok": digest == obligation["sha256"], "detail": f"actual={digest}"})
            if obligation.get("json_schema") is not None:
                try:
                    parsed = json.loads(path.read_text(encoding="utf-8"))
                    errors = sorted(Draft202012Validator(obligation["json_schema"]).iter_errors(parsed), key=lambda e: list(e.path))
                    checks.append({"type": "json_schema", "path": relative, "ok": not errors, "detail": "; ".join(error.message for error in errors[:10]) or "valid"})
                except Exception as exc:
                    checks.append({"type": "json_schema", "path": relative, "ok": False, "detail": f"{type(exc).__name__}: {exc}"})
            artifact = {
                "id": _new_id("xar"),
                "run_id": run_id,
                "role": "output",
                "relative_path": relative,
                "path": str(path),
                "content_hash": digest,
                "size_bytes": size,
                "media_type": obligation["media_type"],
            }
            self._insert_artifact(**artifact)
            artifacts.append(artifact)
        return checks, artifacts

    def _assess_claims(self, tests: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
        assessments = []
        for position, raw in enumerate(tests):
            test = ExecutionClaimTest.from_dict(raw)
            output = json.loads((output_dir / test.output_path).read_text(encoding="utf-8"))
            observed = _extract_json_path(output, test.metric_path)
            support = test.support_condition.evaluate(observed)
            refute = test.refute_condition.evaluate(observed) if test.refute_condition else False
            if support and refute:
                outcome = AnalysisOutcome.INCONCLUSIVE
                rationale = "Support and refute conditions both matched"
            elif support:
                outcome = AnalysisOutcome.SUPPORT
                rationale = test.rationale or "Predeclared support condition matched"
            elif refute:
                outcome = AnalysisOutcome.REFUTE
                rationale = test.rationale or "Predeclared refute condition matched"
            else:
                outcome = AnalysisOutcome.INCONCLUSIVE
                rationale = test.rationale or "Observed value matched neither region"
            assessments.append({
                "position": position,
                "claim_id": test.claim_id,
                "output_path": test.output_path,
                "metric_path": test.metric_path,
                "metric_value": observed,
                "outcome": outcome.value,
                "support_condition": asdict(test.support_condition),
                "refute_condition": asdict(test.refute_condition) if test.refute_condition else None,
                "confidence": test.confidence,
                "rationale": rationale,
            })
        return assessments

    def _persist_assessments(
        self,
        run_id: str,
        assessments: list[dict[str, Any]],
        evidence_id: str | None,
        actor: str,
        actor_role: ActorRole,
    ) -> None:
        for item in assessments:
            assessment_id = _new_id("xca")
            attestation_id = None
            if evidence_id and item["outcome"] in {"support", "refute"}:
                attestation_id = self.ledger.attest(
                    item["claim_id"],
                    evidence_id,
                    Stance.SUPPORT if item["outcome"] == "support" else Stance.REFUTE,
                    confidence=float(item["confidence"]),
                    actor=actor,
                    actor_role=actor_role,
                )
            self.ledger.db.conn.execute(
                """INSERT INTO execution_claim_assessments
                   (id, run_id, position, claim_id, output_path, metric_path,
                    metric_value_json, outcome, support_condition_json,
                    refute_condition_json, confidence, rationale, evidence_id,
                    attestation_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    assessment_id,
                    run_id,
                    item["position"],
                    item["claim_id"],
                    item["output_path"],
                    item["metric_path"],
                    _stable_json(item["metric_value"]),
                    item["outcome"],
                    _stable_json(item["support_condition"]),
                    _stable_json(item["refute_condition"]),
                    item["confidence"],
                    item["rationale"],
                    evidence_id,
                    attestation_id,
                    _utcnow(),
                ),
            )

    @staticmethod
    def _assessment_row(row) -> dict[str, Any]:
        item = dict(row)
        for key in ("metric_value_json", "support_condition_json", "refute_condition_json"):
            item[key.removesuffix("_json")] = json.loads(item.pop(key))
        return item

    def _compare_parent(self, parent_run_id: str, output_artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        parent = {
            row["relative_path"]: row["content_hash"]
            for row in self.ledger.db.conn.execute(
                "SELECT relative_path, content_hash FROM execution_artifacts WHERE run_id = ? AND role = 'output'",
                (parent_run_id,),
            ).fetchall()
        }
        current = {item["relative_path"]: item["content_hash"] for item in output_artifacts}
        return {
            "parent_run_id": parent_run_id,
            "outputs_match": parent == current,
            "parent_outputs": parent,
            "current_outputs": current,
        }
