from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
import tomllib
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable, Protocol

from .execution import (
    ContainerExecutionSpec,
    OCIEngine,
    OutputObligation,
    ResourceLimits,
    StagedFile,
)
from .models import ActorRole

if TYPE_CHECKING:  # pragma: no cover
    from .ledger import EpistemicLedger


CODING_API_VERSION = "1.3"
_MAX_PATCH_BYTES = 2 * 1024 * 1024
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


@dataclass(frozen=True, slots=True)
class PatchProposal:
    patch: str
    rationale: str
    provider: str = "human"
    expected_effect: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.patch.strip():
            raise ValueError("Patch text is required")
        if len(self.patch.encode("utf-8")) > _MAX_PATCH_BYTES:
            raise ValueError("Patch exceeds the configured maximum size")
        if not self.rationale.strip():
            raise ValueError("Patch rationale is required")
        if not self.provider.strip():
            raise ValueError("Patch provider identity is required")


class PatchProvider(Protocol):
    identity: str

    def propose(self, context: dict[str, Any], *, max_candidates: int) -> list[PatchProposal]: ...


class CallablePatchProvider:
    """Vendor-neutral adapter for Codex, OpenClaw, an LLM, or a human tool.

    The callable receives a bounded repository context and must return explicit
    unified-diff proposals. Its output remains non-authoritative until Orbita
    validates, tests, ranks, and separately approves a candidate.
    """

    def __init__(self, identity: str, fn: Callable[[dict[str, Any], int], list[PatchProposal | dict[str, Any]]]):
        if not identity.strip():
            raise ValueError("Provider identity is required")
        self.identity = identity
        self._fn = fn

    def propose(self, context: dict[str, Any], *, max_candidates: int) -> list[PatchProposal]:
        raw = self._fn(context, max_candidates)
        result: list[PatchProposal] = []
        for item in raw[:max_candidates]:
            if isinstance(item, PatchProposal):
                proposal = item
            else:
                proposal = PatchProposal(
                    patch=str(item["patch"]),
                    rationale=str(item["rationale"]),
                    provider=str(item.get("provider") or self.identity),
                    expected_effect=str(item.get("expected_effect", "")),
                    metadata=dict(item.get("metadata", {})),
                )
            result.append(proposal)
        return result


@dataclass(frozen=True, slots=True)
class CodingTestSpec:
    image: str
    command: tuple[str, ...]
    outputs: tuple[OutputObligation, ...]
    include_globs: tuple[str, ...] = ("**/*",)
    environment: dict[str, str] = field(default_factory=dict)
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    max_files: int = 4000
    max_bytes: int = 100 * 1024 * 1024
    allow_unlisted_outputs: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("A test command is required")
        if not self.outputs:
            raise ValueError("At least one test output obligation is required")
        if not self.include_globs:
            raise ValueError("At least one include glob is required")
        if self.max_files < 1 or self.max_files > 20000:
            raise ValueError("max_files must be between 1 and 20000")
        if self.max_bytes < 1 or self.max_bytes > 2 * 1024 * 1024 * 1024:
            raise ValueError("max_bytes is outside the safe range")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CodingTestSpec":
        return cls(
            image=str(value["image"]),
            command=tuple(str(x) for x in value["command"]),
            outputs=tuple(OutputObligation.from_dict(x) for x in value.get("outputs", [])),
            include_globs=tuple(str(x) for x in value.get("include_globs", ["**/*"])),
            environment={str(k): str(v) for k, v in dict(value.get("environment", {})).items()},
            limits=ResourceLimits.from_dict(value.get("limits")),
            max_files=int(value.get("max_files", 4000)),
            max_bytes=int(value.get("max_bytes", 100 * 1024 * 1024)),
            allow_unlisted_outputs=bool(value.get("allow_unlisted_outputs", False)),
            metadata=dict(value.get("metadata", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "command": list(self.command),
            "outputs": [item.as_dict() for item in self.outputs],
            "include_globs": list(self.include_globs),
            "environment": dict(self.environment),
            "limits": asdict(self.limits),
            "max_files": self.max_files,
            "max_bytes": self.max_bytes,
            "allow_unlisted_outputs": self.allow_unlisted_outputs,
            "metadata": dict(self.metadata),
        }


class CodingRuntime:
    """Governed native coding loop.

    Patch generation is replaceable. Repository mutation, candidate isolation,
    test execution, comparison, promotion, rollback, and receipts are owned by
    Orbita and are never delegated to a proposal model.
    """

    def __init__(self, ledger: "EpistemicLedger", workspace: str | Path | None = None):
        self.ledger = ledger
        self.workspace = Path(workspace or (ledger.db.path.parent / "coding_workspace")).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.worktrees = self.workspace / "worktrees"
        self.worktrees.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Session and repository state
    # ------------------------------------------------------------------
    def start_session(
        self,
        repository: str | Path,
        goal: str,
        *,
        test_spec: CodingTestSpec | dict[str, Any] | None = None,
        allowed_paths: list[str] | tuple[str, ...] = (".",),
        max_candidates: int = 4,
        actor: str = "user",
    ) -> dict[str, Any]:
        if not goal.strip():
            raise ValueError("Coding goal is required")
        repo = self._resolve_repository(repository)
        self._require_git_repository(repo)
        if not (1 <= int(max_candidates) <= 12):
            raise ValueError("max_candidates must be between 1 and 12")
        scopes = self._normalize_scopes(allowed_paths)
        base_commit = self._git(repo, ["rev-parse", "HEAD"])["stdout"].strip()
        branch = self._git(repo, ["branch", "--show-current"])["stdout"].strip()
        status = self._git(repo, ["status", "--porcelain=v1"])["stdout"]
        spec = CodingTestSpec.from_dict(test_spec) if isinstance(test_spec, dict) else test_spec
        session_id = new_id("cds")
        now = utcnow()
        payload = {
            "goal": goal.strip(),
            "repository": str(repo),
            "base_commit": base_commit,
            "base_branch": branch,
            "allowed_paths": scopes,
            "test_spec": spec.as_dict() if spec else None,
            "max_candidates": int(max_candidates),
        }
        session_hash = sha256_text(stable_json(payload))
        self.ledger.db.conn.execute(
            """INSERT INTO coding_sessions
               (id, goal, repository_path, base_commit, base_branch, allowed_paths_json,
                test_spec_json, max_candidates, status, session_hash, initial_status_json,
                selected_candidate_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'created', ?, ?, NULL, ?, ?)""",
            (
                session_id,
                goal.strip(),
                str(repo),
                base_commit,
                branch,
                stable_json(scopes),
                stable_json(spec.as_dict() if spec else {}),
                int(max_candidates),
                session_hash,
                stable_json({"porcelain": status, "clean": not bool(status.strip())}),
                now,
                now,
            ),
        )
        self.ledger._event(
            "coding_session",
            session_id,
            "CODING_SESSION_CREATED",
            {"session_hash": session_hash, "base_commit": base_commit, "allowed_paths": scopes},
            actor,
            ActorRole.HUMAN,
        )
        self.ledger.db.conn.commit()
        return self.get_session(session_id)

    def build_provider_context(
        self,
        session_id: str,
        *,
        max_files: int = 80,
        max_total_bytes: int = 256_000,
        excerpt_bytes: int = 12_000,
    ) -> dict[str, Any]:
        session = self._session_row(session_id)
        repo = Path(session["repository_path"])
        files = self._tracked_files(repo)
        manifest: list[dict[str, Any]] = []
        total = 0
        for rel in files[:max_files]:
            path = repo / rel
            if not path.is_file() or path.is_symlink():
                continue
            digest, size = sha256_file(path)
            item: dict[str, Any] = {"path": rel, "sha256": digest, "size": size}
            if size <= excerpt_bytes and self._looks_textual(path):
                text = path.read_text(encoding="utf-8", errors="replace")
                encoded = text.encode("utf-8")
                if total + len(encoded) <= max_total_bytes:
                    item["content"] = text
                    total += len(encoded)
            manifest.append(item)
        baseline = self._latest_test_for_session(session_id, phase="baseline")
        return {
            "schema_version": CODING_API_VERSION,
            "session_id": session_id,
            "goal": session["goal"],
            "base_commit": session["base_commit"],
            "allowed_paths": json.loads(session["allowed_paths_json"]),
            "repository_manifest": manifest,
            "baseline": baseline,
            "constraints": {
                "output": "unified_diff_only",
                "no_shell_control": True,
                "no_commit": True,
                "no_push": True,
                "max_candidates": int(session["max_candidates"]),
            },
        }

    def request_candidates(self, session_id: str, provider: PatchProvider) -> list[dict[str, Any]]:
        session = self._session_row(session_id)
        existing = self.ledger.db.conn.execute(
            "SELECT COUNT(*) AS n FROM coding_candidates WHERE session_id = ?", (session_id,)
        ).fetchone()["n"]
        remaining = int(session["max_candidates"]) - int(existing)
        if remaining <= 0:
            raise ValueError("The session candidate budget is exhausted")
        context = self.build_provider_context(session_id)
        proposals = provider.propose(context, max_candidates=remaining)
        if not proposals:
            raise ValueError("Patch provider returned no candidates")
        return [self.add_candidate(session_id, item) for item in proposals[:remaining]]

    # ------------------------------------------------------------------
    # Candidate validation and isolated worktrees
    # ------------------------------------------------------------------
    def add_candidate(
        self,
        session_id: str,
        proposal: PatchProposal | dict[str, Any],
        *,
        actor: str = "proposal_provider",
    ) -> dict[str, Any]:
        session = self._session_row(session_id)
        if not isinstance(proposal, PatchProposal):
            proposal = PatchProposal(
                patch=str(proposal["patch"]),
                rationale=str(proposal["rationale"]),
                provider=str(proposal.get("provider", actor)),
                expected_effect=str(proposal.get("expected_effect", "")),
                metadata=dict(proposal.get("metadata", {})),
            )
        count = self.ledger.db.conn.execute(
            "SELECT COUNT(*) AS n FROM coding_candidates WHERE session_id = ?", (session_id,)
        ).fetchone()["n"]
        if int(count) >= int(session["max_candidates"]):
            raise ValueError("The session candidate budget is exhausted")
        changed_paths = self._validate_patch(proposal.patch, json.loads(session["allowed_paths_json"]))
        patch_hash = sha256_text(proposal.patch)
        candidate_id = new_id("cdc")
        now = utcnow()
        try:
            self.ledger.db.conn.execute(
                """INSERT INTO coding_candidates
                   (id, session_id, position, provider, rationale, expected_effect,
                    patch_text, patch_hash, changed_paths_json, status, worktree_path,
                    applied_diff_text, applied_diff_hash, static_checks_json, diff_stats_json,
                    score_json, metadata_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'validated', NULL, '', NULL, '[]', '{}', '{}', ?, ?, ?)""",
                (
                    candidate_id,
                    session_id,
                    int(count),
                    proposal.provider,
                    proposal.rationale,
                    proposal.expected_effect,
                    proposal.patch,
                    patch_hash,
                    stable_json(changed_paths),
                    stable_json(proposal.metadata),
                    now,
                    now,
                ),
            )
        except Exception as exc:
            if "UNIQUE" in str(exc).upper():
                raise ValueError("An identical patch candidate already exists in this session") from exc
            raise
        self.ledger._event(
            "coding_candidate",
            candidate_id,
            "CODING_CANDIDATE_REGISTERED",
            {"session_id": session_id, "patch_hash": patch_hash, "changed_paths": changed_paths, "provider": proposal.provider},
            actor,
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()
        return self.get_candidate(candidate_id)

    def prepare_candidate(self, candidate_id: str) -> dict[str, Any]:
        candidate = self._candidate_row(candidate_id)
        session = self._session_row(candidate["session_id"])
        repo = Path(session["repository_path"])
        worktree = self.worktrees / candidate_id
        if worktree.exists():
            self._remove_worktree(repo, worktree)
        self._git(repo, ["worktree", "add", "--detach", str(worktree), session["base_commit"]], timeout=60)
        try:
            self._git_with_input(worktree, ["apply", "--check", "--recount", "-"], candidate["patch_text"])
            self._git_with_input(
                worktree,
                ["apply", "--whitespace=error-all", "--recount", "-"],
                candidate["patch_text"],
            )
            checks, stats = self._static_checks(
                worktree,
                json.loads(session["allowed_paths_json"]),
                json.loads(candidate["changed_paths_json"]),
            )
            applied_diff = self._canonical_diff(worktree)
            applied_hash = sha256_text(applied_diff)
            status = "prepared" if checks and all(item["ok"] for item in checks) else "static_failed"
            self.ledger.db.conn.execute(
                """UPDATE coding_candidates SET status = ?, worktree_path = ?, applied_diff_text = ?,
                   applied_diff_hash = ?, static_checks_json = ?, diff_stats_json = ?, updated_at = ?
                   WHERE id = ?""",
                (
                    status,
                    str(worktree),
                    applied_diff,
                    applied_hash,
                    stable_json(checks),
                    stable_json(stats),
                    utcnow(),
                    candidate_id,
                ),
            )
            self.ledger._event(
                "coding_candidate",
                candidate_id,
                "CODING_CANDIDATE_PREPARED",
                {"status": status, "applied_diff_hash": applied_hash, "static_checks": checks, "diff_stats": stats},
                "coding_runtime",
                ActorRole.TOOL,
            )
            self.ledger.db.conn.commit()
        except Exception as exc:
            self.ledger.db.conn.execute(
                "UPDATE coding_candidates SET status = 'apply_failed', static_checks_json = ?, updated_at = ? WHERE id = ?",
                (stable_json([{"type": "patch_apply", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}]), utcnow(), candidate_id),
            )
            self.ledger.db.conn.commit()
        return self.get_candidate(candidate_id)

    def prepare_all_candidates(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT id FROM coding_candidates WHERE session_id = ? ORDER BY position", (session_id,)
        ).fetchall()
        return [self.prepare_candidate(row["id"]) for row in rows]

    # ------------------------------------------------------------------
    # Containerized testing and diagnosis
    # ------------------------------------------------------------------
    def submit_baseline_test(self, session_id: str) -> dict[str, Any]:
        session = self._session_row(session_id)
        baseline_root = self.worktrees / f"{session_id}-baseline"
        repo = Path(session["repository_path"])
        if baseline_root.exists():
            self._remove_worktree(repo, baseline_root)
        self._git(repo, ["worktree", "add", "--detach", str(baseline_root), session["base_commit"]], timeout=60)
        return self._submit_test(session_id, None, baseline_root, "baseline")

    def submit_candidate_test(self, candidate_id: str) -> dict[str, Any]:
        candidate = self._candidate_row(candidate_id)
        if candidate["status"] not in {"prepared", "tested_failed", "tested_passed"}:
            raise ValueError("Candidate must pass patch application and static checks before testing")
        return self._submit_test(
            candidate["session_id"], candidate_id, Path(candidate["worktree_path"]), "candidate"
        )

    def finalize_test(self, coding_test_id: str) -> dict[str, Any]:
        row = self._test_row(coding_test_id)
        run = self.ledger.executions.get(row["execution_run_id"])
        diagnostics = self._diagnose_execution(run)
        passed = run["status"] == "succeeded" and run["receipt_integrity_valid"] is True
        now = utcnow()
        self.ledger.db.conn.execute(
            """UPDATE coding_tests SET status = ?, diagnostics_json = ?, receipt_hash = ?, completed_at = ?
               WHERE id = ?""",
            (
                "passed" if passed else "failed",
                stable_json(diagnostics),
                run.get("receipt_hash"),
                now,
                coding_test_id,
            ),
        )
        if row["candidate_id"]:
            candidate = self._candidate_row(row["candidate_id"])
            checks = json.loads(candidate["static_checks_json"])
            stats = json.loads(candidate["diff_stats_json"])
            changed_lines = int(stats.get("insertions", 0)) + int(stats.get("deletions", 0))
            static_ok = bool(checks) and all(item["ok"] for item in checks)
            baseline = self._latest_test_for_session(row["session_id"], phase="baseline")
            baseline_failed = bool(baseline and baseline.get("status") == "failed")
            total = (100.0 if passed else 0.0) + (20.0 if static_ok else 0.0)
            if passed and baseline_failed:
                total += 20.0
            total -= min(20.0, changed_lines / 10.0)
            score = {
                "total": round(total, 3),
                "tests_passed": passed,
                "static_checks_passed": static_ok,
                "baseline_failed": baseline_failed,
                "changed_lines": changed_lines,
                "penalty": round(min(20.0, changed_lines / 10.0), 3),
            }
            self.ledger.db.conn.execute(
                "UPDATE coding_candidates SET status = ?, score_json = ?, updated_at = ? WHERE id = ?",
                ("tested_passed" if passed else "tested_failed", stable_json(score), now, row["candidate_id"]),
            )
        self.ledger._event(
            "coding_test",
            coding_test_id,
            "CODING_TEST_FINALIZED",
            {"execution_run_id": row["execution_run_id"], "passed": passed, "diagnostics": diagnostics},
            "coding_runtime",
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()
        return self.get_test(coding_test_id)

    def rank_candidates(self, session_id: str) -> list[dict[str, Any]]:
        candidates = self.list_candidates(session_id)
        for item in candidates:
            score = item.get("score", {})
            item["rank_key"] = [
                1 if score.get("tests_passed") else 0,
                1 if score.get("static_checks_passed") else 0,
                float(score.get("total", -10_000)),
                -int(score.get("changed_lines", 10**9)),
            ]
        candidates.sort(key=lambda item: tuple(item["rank_key"]), reverse=True)
        for index, item in enumerate(candidates, start=1):
            item["rank"] = index
        return candidates

    def select_candidate(self, session_id: str, candidate_id: str | None = None) -> dict[str, Any]:
        ranked = self.rank_candidates(session_id)
        if candidate_id is None:
            passing = [item for item in ranked if item.get("score", {}).get("tests_passed")]
            if not passing:
                raise ValueError("No candidate has a verified passing test run")
            selected = passing[0]
        else:
            selected = next((item for item in ranked if item["id"] == candidate_id), None)
            if selected is None:
                raise KeyError(f"Unknown candidate in session: {candidate_id}")
            if not selected.get("score", {}).get("tests_passed"):
                raise ValueError("Only a verified passing candidate can be selected")
        self.ledger.db.conn.execute(
            "UPDATE coding_candidates SET status = CASE WHEN id = ? THEN 'selected' WHEN status = 'selected' THEN 'tested_passed' ELSE status END, updated_at = ? WHERE session_id = ?",
            (selected["id"], utcnow(), session_id),
        )
        self.ledger.db.conn.execute(
            "UPDATE coding_sessions SET selected_candidate_id = ?, status = 'candidate_selected', updated_at = ? WHERE id = ?",
            (selected["id"], utcnow(), session_id),
        )
        self.ledger.db.conn.commit()
        return self.get_candidate(selected["id"])

    # ------------------------------------------------------------------
    # Exact approval, promotion, and rollback
    # ------------------------------------------------------------------
    def request_promotion(self, candidate_id: str) -> dict[str, Any]:
        candidate = self._candidate_row(candidate_id)
        session = self._session_row(candidate["session_id"])
        if session["selected_candidate_id"] != candidate_id or candidate["status"] != "selected":
            raise ValueError("Candidate must be selected before promotion")
        latest = self._latest_test_for_candidate(candidate_id)
        if not latest or latest["status"] != "passed":
            raise ValueError("Candidate lacks a verified passing test")
        repo = Path(session["repository_path"])
        current_head = self._git(repo, ["rev-parse", "HEAD"])["stdout"].strip()
        current_status = self._git(repo, ["status", "--porcelain=v1"])["stdout"]
        if current_head != session["base_commit"]:
            raise ValueError("Repository HEAD changed since the session began")
        if current_status.strip():
            raise ValueError("Repository working tree must be clean before promotion")
        payload = {
            "action": "promote",
            "session_id": session["id"],
            "candidate_id": candidate_id,
            "base_commit": session["base_commit"],
            "patch_hash": candidate["patch_hash"],
            "applied_diff_hash": candidate["applied_diff_hash"],
            "test_receipt_hash": latest.get("receipt_hash"),
            "allowed_paths": json.loads(session["allowed_paths_json"]),
        }
        binding_hash = sha256_text(stable_json(payload))
        approval_id = new_id("cda")
        self.ledger.db.conn.execute(
            """INSERT INTO coding_approvals
               (id, session_id, candidate_id, action, binding_json, binding_hash, status,
                reviewer, rationale, created_at, decided_at, consumed_at)
               VALUES (?, ?, ?, 'promote', ?, ?, 'pending', NULL, NULL, ?, NULL, NULL)""",
            (approval_id, session["id"], candidate_id, stable_json(payload), binding_hash, utcnow()),
        )
        self.ledger.db.conn.commit()
        return self.get_approval(approval_id)

    def request_rollback(self, promotion_id: str) -> dict[str, Any]:
        promotion = self._promotion_row(promotion_id)
        if promotion["status"] != "promoted":
            raise ValueError("Only an active promotion can be rolled back")
        payload = {
            "action": "rollback",
            "promotion_id": promotion_id,
            "session_id": promotion["session_id"],
            "candidate_id": promotion["candidate_id"],
            "post_diff_hash": promotion["post_diff_hash"],
            "patch_hash": promotion["patch_hash"],
        }
        approval_id = new_id("cda")
        self.ledger.db.conn.execute(
            """INSERT INTO coding_approvals
               (id, session_id, candidate_id, action, binding_json, binding_hash, status,
                reviewer, rationale, created_at, decided_at, consumed_at)
               VALUES (?, ?, ?, 'rollback', ?, ?, 'pending', NULL, NULL, ?, NULL, NULL)""",
            (
                approval_id,
                promotion["session_id"],
                promotion["candidate_id"],
                stable_json(payload),
                sha256_text(stable_json(payload)),
                utcnow(),
            ),
        )
        self.ledger.db.conn.commit()
        return self.get_approval(approval_id)

    def approve(self, approval_id: str, *, reviewer: str, rationale: str) -> dict[str, Any]:
        if not reviewer.strip() or not rationale.strip():
            raise ValueError("Reviewer and rationale are required")
        row = self._approval_row(approval_id)
        if row["status"] != "pending":
            raise ValueError("Approval was already decided")
        if sha256_text(row["binding_json"]) != row["binding_hash"]:
            raise ValueError("Approval binding has been tampered with")
        now = utcnow()
        self.ledger.db.conn.execute(
            "UPDATE coding_approvals SET status = 'approved', reviewer = ?, rationale = ?, decided_at = ? WHERE id = ?",
            (reviewer, rationale, now, approval_id),
        )
        self.ledger.db.conn.commit()
        return self.get_approval(approval_id)

    def promote(self, approval_id: str) -> dict[str, Any]:
        approval = self._approval_row(approval_id)
        if approval["action"] != "promote" or approval["status"] != "approved" or approval["consumed_at"]:
            raise ValueError("A fresh approved promotion authorization is required")
        binding = json.loads(approval["binding_json"])
        candidate = self._candidate_row(binding["candidate_id"])
        session = self._session_row(binding["session_id"])
        latest = self._latest_test_for_candidate(candidate["id"])
        expected = {
            "action": "promote",
            "session_id": session["id"],
            "candidate_id": candidate["id"],
            "base_commit": session["base_commit"],
            "patch_hash": candidate["patch_hash"],
            "applied_diff_hash": candidate["applied_diff_hash"],
            "test_receipt_hash": latest.get("receipt_hash") if latest else None,
            "allowed_paths": json.loads(session["allowed_paths_json"]),
        }
        if sha256_text(stable_json(expected)) != approval["binding_hash"]:
            raise ValueError("Candidate or test state changed after approval")
        repo = Path(session["repository_path"])
        head = self._git(repo, ["rev-parse", "HEAD"])["stdout"].strip()
        if head != session["base_commit"]:
            raise ValueError("Repository HEAD changed after approval")
        if self._git(repo, ["status", "--porcelain=v1"])["stdout"].strip():
            raise ValueError("Repository working tree changed after approval")
        patch = candidate["applied_diff_text"]
        self._git_with_input(repo, ["apply", "--check", "--recount", "-"], patch)
        self._git_with_input(repo, ["apply", "--whitespace=error-all", "--recount", "-"], patch)
        post_diff = self._canonical_diff(repo)
        post_hash = sha256_text(post_diff)
        if post_hash != candidate["applied_diff_hash"]:
            self._git_with_input(repo, ["apply", "-R", "--recount", "-"], patch)
            raise RuntimeError("Promoted diff did not match the tested candidate; changes were rolled back")
        promotion_id = new_id("cdp")
        payload = {
            "promotion_id": promotion_id,
            "session_id": session["id"],
            "candidate_id": candidate["id"],
            "base_commit": session["base_commit"],
            "patch_hash": candidate["patch_hash"],
            "post_diff_hash": post_hash,
            "approval_id": approval_id,
        }
        receipt_hash = sha256_text(stable_json(payload))
        now = utcnow()
        self.ledger.db.conn.execute(
            """INSERT INTO coding_promotions
               (id, session_id, candidate_id, approval_id, repository_path, base_commit,
                patch_text, patch_hash, post_diff_hash, status, receipt_hash, created_at,
                rolled_back_at, rollback_receipt_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'promoted', ?, ?, NULL, NULL)""",
            (
                promotion_id,
                session["id"],
                candidate["id"],
                approval_id,
                str(repo),
                session["base_commit"],
                patch,
                candidate["patch_hash"],
                post_hash,
                receipt_hash,
                now,
            ),
        )
        self.ledger.db.conn.execute(
            "UPDATE coding_approvals SET consumed_at = ? WHERE id = ?", (now, approval_id)
        )
        self.ledger.db.conn.execute(
            "UPDATE coding_sessions SET status = 'promoted', updated_at = ? WHERE id = ?",
            (now, session["id"]),
        )
        self.ledger._event(
            "coding_promotion",
            promotion_id,
            "CODING_CANDIDATE_PROMOTED",
            {"receipt_hash": receipt_hash, "post_diff_hash": post_hash, "candidate_id": candidate["id"]},
            approval["reviewer"],
            ActorRole.HUMAN,
        )
        self.ledger.db.conn.commit()
        return self.get_promotion(promotion_id)

    def rollback(self, approval_id: str) -> dict[str, Any]:
        approval = self._approval_row(approval_id)
        if approval["action"] != "rollback" or approval["status"] != "approved" or approval["consumed_at"]:
            raise ValueError("A fresh approved rollback authorization is required")
        binding = json.loads(approval["binding_json"])
        promotion = self._promotion_row(binding["promotion_id"])
        repo = Path(promotion["repository_path"])
        current_diff = self._canonical_diff(repo)
        if sha256_text(current_diff) != promotion["post_diff_hash"]:
            raise ValueError("Repository changed after promotion; automatic rollback is unsafe")
        self._git_with_input(repo, ["apply", "-R", "--check", "--recount", "-"], promotion["patch_text"])
        self._git_with_input(repo, ["apply", "-R", "--recount", "-"], promotion["patch_text"])
        if self._git(repo, ["status", "--porcelain=v1"])["stdout"].strip():
            raise RuntimeError("Rollback did not restore the clean base state")
        now = utcnow()
        payload = {
            "promotion_id": promotion["id"],
            "approval_id": approval_id,
            "restored_commit": promotion["base_commit"],
            "clean": True,
        }
        rollback_hash = sha256_text(stable_json(payload))
        self.ledger.db.conn.execute(
            "UPDATE coding_promotions SET status = 'rolled_back', rolled_back_at = ?, rollback_receipt_hash = ? WHERE id = ?",
            (now, rollback_hash, promotion["id"]),
        )
        self.ledger.db.conn.execute(
            "UPDATE coding_approvals SET consumed_at = ? WHERE id = ?", (now, approval_id)
        )
        self.ledger.db.conn.execute(
            "UPDATE coding_sessions SET status = 'rolled_back', updated_at = ? WHERE id = ?",
            (now, promotion["session_id"]),
        )
        self.ledger.db.conn.commit()
        return self.get_promotion(promotion["id"])

    # ------------------------------------------------------------------
    # Read and verify APIs
    # ------------------------------------------------------------------
    def get_session(self, session_id: str) -> dict[str, Any]:
        row = self._session_row(session_id)
        result = dict(row)
        for key in ("allowed_paths_json", "test_spec_json", "initial_status_json"):
            result[key.removesuffix("_json")] = json.loads(result.pop(key))
        result["session_integrity_valid"] = self.verify_session(session_id)
        result["candidates"] = self.list_candidates(session_id)
        result["tests"] = self.list_tests(session_id)
        return result

    def list_sessions(self) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT id FROM coding_sessions ORDER BY created_at DESC"
        ).fetchall()
        return [self.get_session(row["id"]) for row in rows]

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        row = self._candidate_row(candidate_id)
        result = dict(row)
        for key in ("changed_paths_json", "static_checks_json", "diff_stats_json", "score_json", "metadata_json"):
            result[key.removesuffix("_json")] = json.loads(result.pop(key))
        result["patch_integrity_valid"] = sha256_text(result["patch_text"]) == result["patch_hash"]
        result["applied_diff_integrity_valid"] = (
            None if not result["applied_diff_hash"] else sha256_text(result["applied_diff_text"]) == result["applied_diff_hash"]
        )
        result["tests"] = [item for item in self.list_tests(result["session_id"]) if item["candidate_id"] == candidate_id]
        return result

    def list_candidates(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT id FROM coding_candidates WHERE session_id = ? ORDER BY position", (session_id,)
        ).fetchall()
        return [self.get_candidate(row["id"]) for row in rows]

    def get_test(self, coding_test_id: str) -> dict[str, Any]:
        row = self._test_row(coding_test_id)
        result = dict(row)
        result["diagnostics"] = json.loads(result.pop("diagnostics_json"))
        result["execution"] = self.ledger.executions.get(result["execution_run_id"])
        return result

    def list_tests(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT id FROM coding_tests WHERE session_id = ? ORDER BY created_at", (session_id,)
        ).fetchall()
        return [self.get_test(row["id"]) for row in rows]

    def get_approval(self, approval_id: str) -> dict[str, Any]:
        row = self._approval_row(approval_id)
        result = dict(row)
        result["binding"] = json.loads(result.pop("binding_json"))
        result["integrity_valid"] = sha256_text(stable_json(result["binding"])) == result["binding_hash"]
        return result

    def get_promotion(self, promotion_id: str) -> dict[str, Any]:
        row = self._promotion_row(promotion_id)
        result = dict(row)
        payload = {
            "promotion_id": result["id"],
            "session_id": result["session_id"],
            "candidate_id": result["candidate_id"],
            "base_commit": result["base_commit"],
            "patch_hash": result["patch_hash"],
            "post_diff_hash": result["post_diff_hash"],
            "approval_id": result["approval_id"],
        }
        result["receipt_integrity_valid"] = sha256_text(stable_json(payload)) == result["receipt_hash"]
        return result

    def verify_session(self, session_id: str) -> bool:
        row = self._session_row(session_id)
        payload = {
            "goal": row["goal"],
            "repository": row["repository_path"],
            "base_commit": row["base_commit"],
            "base_branch": row["base_branch"],
            "allowed_paths": json.loads(row["allowed_paths_json"]),
            "test_spec": json.loads(row["test_spec_json"]) or None,
            "max_candidates": int(row["max_candidates"]),
        }
        return sha256_text(stable_json(payload)) == row["session_hash"]

    def verify_candidate(self, candidate_id: str) -> bool:
        row = self._candidate_row(candidate_id)
        patch_ok = sha256_text(row["patch_text"]) == row["patch_hash"]
        diff_ok = not row["applied_diff_hash"] or sha256_text(row["applied_diff_text"]) == row["applied_diff_hash"]
        return patch_ok and diff_ok

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _submit_test(self, session_id: str, candidate_id: str | None, root: Path, phase: str) -> dict[str, Any]:
        session = self._session_row(session_id)
        raw_spec = json.loads(session["test_spec_json"])
        if not raw_spec:
            raise ValueError("The coding session has no container test specification")
        spec = CodingTestSpec.from_dict(raw_spec)
        staged = self._stage_repository(root, spec)
        execution_spec = ContainerExecutionSpec(
            name=f"Coding {phase}: {session['goal'][:80]}",
            image=spec.image,
            command=spec.command,
            code_files=tuple(staged),
            outputs=spec.outputs,
            environment=spec.environment,
            limits=spec.limits,
            metadata={
                **spec.metadata,
                "coding_session_id": session_id,
                "coding_candidate_id": candidate_id,
                "coding_phase": phase,
                "base_commit": session["base_commit"],
            },
            allow_unlisted_outputs=spec.allow_unlisted_outputs,
        )
        run = self.ledger.executions.submit(execution_spec, actor="coding_runtime", actor_role=ActorRole.TOOL)
        test_id = new_id("cdt")
        self.ledger.db.conn.execute(
            """INSERT INTO coding_tests
               (id, session_id, candidate_id, phase, execution_run_id, status,
                diagnostics_json, receipt_hash, created_at, completed_at)
               VALUES (?, ?, ?, ?, ?, 'waiting_approval', '{}', NULL, ?, NULL)""",
            (test_id, session_id, candidate_id, phase, run["id"], utcnow()),
        )
        self.ledger.db.conn.commit()
        return self.get_test(test_id)

    def approve_and_execute_test(
        self,
        coding_test_id: str,
        *,
        reviewer: str,
        rationale: str,
        engine: OCIEngine | None = None,
    ) -> dict[str, Any]:
        row = self._test_row(coding_test_id)
        run = self.ledger.executions.get(row["execution_run_id"])
        if run["status"] == "waiting_approval":
            self.ledger.executions.approve(run["id"], reviewer=reviewer, rationale=rationale)
        self.ledger.executions.execute(run["id"], engine=engine)
        return self.finalize_test(coding_test_id)

    def _stage_repository(self, root: Path, spec: CodingTestSpec) -> list[StagedFile]:
        root = root.resolve()
        selected: set[str] = set()
        for pattern in spec.include_globs:
            for path in root.glob(pattern):
                if path.is_dir() or path.is_symlink():
                    continue
                try:
                    rel = path.resolve().relative_to(root).as_posix()
                except ValueError:
                    continue
                if rel == ".git" or rel.startswith(".git/"):
                    continue
                selected.add(rel)
        staged: list[StagedFile] = []
        total = 0
        for rel in sorted(selected):
            path = root / rel
            size = path.stat().st_size
            total += size
            if len(staged) + 1 > spec.max_files or total > spec.max_bytes:
                raise ValueError("Repository staging exceeds the test specification bounds")
            staged.append(StagedFile(rel, source=path))
        if not staged:
            raise ValueError("The test specification selected no repository files")
        return staged

    def _validate_patch(self, patch: str, allowed_paths: list[str]) -> list[str]:
        if "GIT binary patch" in patch or "Binary files " in patch:
            raise ValueError("Binary patches are not supported")
        if re.search(r"^new file mode 120000$|^old mode 120000$|^new mode 120000$", patch, re.M):
            raise ValueError("Symlink patches are forbidden")
        changed: list[str] = []
        for line in patch.splitlines():
            if not line.startswith("diff --git "):
                continue
            parts = line.split(" ")
            if len(parts) != 4 or not parts[2].startswith("a/") or not parts[3].startswith("b/"):
                raise ValueError("Patch paths must use unquoted git a/ and b/ paths")
            left = self._safe_patch_path(parts[2][2:])
            right = self._safe_patch_path(parts[3][2:])
            if left != right:
                raise ValueError("Renames are not supported in native coding v1.3")
            if not self._path_allowed(right, allowed_paths):
                raise PermissionError(f"Patch path is outside the approved scope: {right}")
            changed.append(right)
        if not changed:
            raise ValueError("No git diff file headers were found")
        for line in patch.splitlines():
            if not line.startswith("+") or line.startswith("+++"):
                continue
            added = line[1:]
            for pattern in _SECRET_PATTERNS:
                if pattern.search(added):
                    raise ValueError("Patch appears to add secret material")
        return list(dict.fromkeys(changed))

    def _static_checks(
        self,
        root: Path,
        allowed_paths: list[str],
        expected_paths: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        actual_paths = [line for line in self._git(root, ["diff", "--name-only", "--no-ext-diff", "--"])["stdout"].splitlines() if line]
        checks: list[dict[str, Any]] = []
        checks.append({
            "type": "changed_paths_match_patch",
            "ok": set(actual_paths) == set(expected_paths),
            "detail": {"expected": expected_paths, "actual": actual_paths},
        })
        checks.append({
            "type": "changed_paths_within_scope",
            "ok": all(self._path_allowed(path, allowed_paths) for path in actual_paths),
            "detail": actual_paths,
        })
        symlinks = [path for path in actual_paths if (root / path).exists() and (root / path).is_symlink()]
        checks.append({"type": "no_symlinks", "ok": not symlinks, "detail": symlinks})
        submodules = self._git(root, ["ls-files", "--stage", "--", *actual_paths])["stdout"]
        has_submodule = any(line.startswith("160000 ") for line in submodules.splitlines())
        checks.append({"type": "no_submodules", "ok": not has_submodule, "detail": "submodule entry detected" if has_submodule else "none"})
        for rel in actual_paths:
            path = root / rel
            if not path.exists():
                continue
            try:
                if path.suffix == ".py":
                    ast.parse(path.read_text(encoding="utf-8"), filename=rel)
                    checks.append({"type": "python_parse", "path": rel, "ok": True, "detail": "valid"})
                elif path.suffix == ".json":
                    json.loads(path.read_text(encoding="utf-8"))
                    checks.append({"type": "json_parse", "path": rel, "ok": True, "detail": "valid"})
                elif path.suffix == ".toml":
                    tomllib.loads(path.read_text(encoding="utf-8"))
                    checks.append({"type": "toml_parse", "path": rel, "ok": True, "detail": "valid"})
            except Exception as exc:
                checks.append({"type": "syntax_parse", "path": rel, "ok": False, "detail": f"{type(exc).__name__}: {exc}"})
        numstat = self._git(root, ["diff", "--numstat", "--no-ext-diff", "--"])["stdout"]
        insertions = deletions = 0
        for line in numstat.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                insertions += int(parts[0])
                deletions += int(parts[1])
        stats = {"files": len(actual_paths), "insertions": insertions, "deletions": deletions, "changed_paths": actual_paths}
        return checks, stats

    def _diagnose_execution(self, run: dict[str, Any]) -> dict[str, Any]:
        text = "\n".join([str(run.get("stdout", "")), str(run.get("stderr", "")), str(run.get("error", ""))])
        categories: list[str] = []
        if run.get("timed_out"):
            categories.append("timeout")
        if run.get("exit_code") not in (None, 0):
            categories.append("nonzero_exit")
        if re.search(r"SyntaxError|IndentationError", text):
            categories.append("syntax_error")
        if re.search(r"AssertionError|\bFAILED\b|failed,", text, re.I):
            categories.append("test_failure")
        if re.search(r"ModuleNotFoundError|ImportError", text):
            categories.append("import_error")
        if "output proof obligations failed" in text.lower() or any(not item.get("ok", False) for item in run.get("checks", [])):
            categories.append("proof_obligation_failure")
        failed_tests = sorted(set(re.findall(r"(?:FAILED\s+)?([\w./-]+::[\w\[\]-]+)", text)))[:50]
        exceptions = sorted(set(re.findall(r"\b([A-Z][A-Za-z]+(?:Error|Exception))\b", text)))[:20]
        locations = []
        for path, line in re.findall(r"File \"([^\"]+)\", line (\d+)", text)[:30]:
            locations.append({"path": path, "line": int(line)})
        return {
            "categories": list(dict.fromkeys(categories)) or (["passed"] if run.get("status") == "succeeded" else ["unknown_failure"]),
            "failed_tests": failed_tests,
            "exceptions": exceptions,
            "locations": locations,
            "exit_code": run.get("exit_code"),
            "timed_out": bool(run.get("timed_out")),
            "summary": (run.get("error") or text.strip())[:4000],
        }

    def _canonical_diff(self, repo: Path) -> str:
        return self._git(repo, ["diff", "--binary", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/", "--"])["stdout"]

    def _tracked_files(self, repo: Path) -> list[str]:
        output = self._git(repo, ["ls-files", "-z"])["stdout"]
        return [item for item in output.split("\x00") if item]

    def _resolve_repository(self, repository: str | Path) -> Path:
        raw = Path(repository).expanduser()
        path = raw.resolve() if raw.is_absolute() else (self.workspace / raw).resolve()
        try:
            path.relative_to(self.workspace)
        except ValueError as exc:
            raise PermissionError("Repository must be inside the configured coding workspace") from exc
        if not path.is_dir():
            raise FileNotFoundError(str(path))
        return path

    @staticmethod
    def _require_git_repository(repo: Path) -> None:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            shell=False,
            timeout=20,
        )
        if proc.returncode != 0 or proc.stdout.strip() != "true":
            raise ValueError("The target is not a Git working tree")

    @staticmethod
    def _safe_patch_path(raw: str) -> str:
        if "\\" in raw or raw.startswith("/") or "\x00" in raw:
            raise ValueError(f"Unsafe patch path: {raw}")
        path = PurePosixPath(raw)
        if not raw or raw == "." or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(f"Unsafe patch path: {raw}")
        if path.parts[0] == ".git":
            raise ValueError("Patches cannot modify .git")
        return path.as_posix()

    @classmethod
    def _normalize_scopes(cls, values: list[str] | tuple[str, ...]) -> list[str]:
        scopes: list[str] = []
        for raw in values:
            value = str(raw).replace("\\", "/").strip().rstrip("/") or "."
            if value == ".":
                scopes.append(value)
            else:
                scopes.append(cls._safe_patch_path(value))
        if not scopes:
            raise ValueError("At least one allowed path scope is required")
        return list(dict.fromkeys(scopes))

    @staticmethod
    def _path_allowed(path: str, scopes: list[str]) -> bool:
        candidate = PurePosixPath(path)
        for scope in scopes:
            if scope == ".":
                return True
            root = PurePosixPath(scope)
            if candidate == root or root in candidate.parents:
                return True
        return False

    @staticmethod
    def _looks_textual(path: Path) -> bool:
        return path.suffix.lower() in {
            ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".toml", ".yaml", ".yml",
            ".md", ".txt", ".html", ".css", ".rs", ".go", ".java", ".c", ".h", ".cpp",
        }

    @staticmethod
    def _git(repo: Path, args: list[str], *, timeout: int = 30) -> dict[str, Any]:
        if shutil.which("git") is None:
            raise RuntimeError("git is not installed")
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or f"git command failed: {' '.join(args)}")
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    @staticmethod
    def _git_with_input(repo: Path, args: list[str], text: str, *, timeout: int = 30) -> dict[str, Any]:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            input=text,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or f"git command failed: {' '.join(args)}")
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    def _remove_worktree(self, repo: Path, path: Path) -> None:
        try:
            self._git(repo, ["worktree", "remove", "--force", str(path)], timeout=60)
        except Exception:
            shutil.rmtree(path, ignore_errors=True)
            try:
                self._git(repo, ["worktree", "prune"])
            except Exception:
                pass

    def _session_row(self, session_id: str):
        row = self.ledger.db.conn.execute("SELECT * FROM coding_sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown coding session: {session_id}")
        return row

    def _candidate_row(self, candidate_id: str):
        row = self.ledger.db.conn.execute("SELECT * FROM coding_candidates WHERE id = ?", (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown coding candidate: {candidate_id}")
        return row

    def _test_row(self, coding_test_id: str):
        row = self.ledger.db.conn.execute("SELECT * FROM coding_tests WHERE id = ?", (coding_test_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown coding test: {coding_test_id}")
        return row

    def _approval_row(self, approval_id: str):
        row = self.ledger.db.conn.execute("SELECT * FROM coding_approvals WHERE id = ?", (approval_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown coding approval: {approval_id}")
        return row

    def _promotion_row(self, promotion_id: str):
        row = self.ledger.db.conn.execute("SELECT * FROM coding_promotions WHERE id = ?", (promotion_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown coding promotion: {promotion_id}")
        return row

    def _latest_test_for_session(self, session_id: str, *, phase: str) -> dict[str, Any] | None:
        row = self.ledger.db.conn.execute(
            "SELECT id FROM coding_tests WHERE session_id = ? AND phase = ? ORDER BY created_at DESC LIMIT 1",
            (session_id, phase),
        ).fetchone()
        return self.get_test(row["id"]) if row else None

    def _latest_test_for_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        row = self.ledger.db.conn.execute(
            "SELECT id FROM coding_tests WHERE candidate_id = ? ORDER BY created_at DESC LIMIT 1",
            (candidate_id,),
        ).fetchone()
        return self.get_test(row["id"]) if row else None
