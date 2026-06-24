from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .ledger import EpistemicLedger, stable_json, utcnow
from .models import ActionResult, ObligationResult, RiskLevel, StepStatus
from .planner import CandidatePlan, CandidateStep
from .support import SupportEngine


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


class SafeActionRuntime:
    """A deliberately small, capability-based action runtime.

    It does not expose a general shell. Every capability must be registered as a
    named action and all paths are confined to a dedicated workspace.
    """

    def __init__(self, ledger: EpistemicLedger, workspace: str | Path):
        self.ledger = ledger
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.support = SupportEngine(ledger)
        self.actions: dict[str, Callable[[dict[str, Any]], ActionResult]] = {
            "write_text": self._write_text,
            "hash_file": self._hash_file,
            "copy_file": self._copy_file,
        }

    def persist_plan(self, plan: CandidatePlan) -> str:
        plan_id = f"pln_{uuid.uuid4().hex[:16]}"
        self.ledger.db.conn.execute(
            "INSERT INTO plans (id, goal, status, created_at) VALUES (?, ?, ?, ?)",
            (plan_id, plan.goal, "pending", utcnow()),
        )
        for sequence, step in enumerate(plan.steps):
            step_id = f"stp_{uuid.uuid4().hex[:16]}"
            args_hash = _hash_payload(step.args)
            self.ledger.db.conn.execute(
                """INSERT INTO steps
                   (id, plan_id, sequence, intent, action_type, args_json,
                    required_claims_json, obligations_json, risk, status, args_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    step_id,
                    plan_id,
                    sequence,
                    step.intent,
                    step.action_type,
                    stable_json(step.args),
                    stable_json(step.required_claims),
                    stable_json(step.obligations),
                    step.risk.value,
                    StepStatus.PENDING.value,
                    args_hash,
                    utcnow(),
                ),
            )
        self.ledger.db.conn.commit()
        return plan_id

    def request_approval(self, step_id: str) -> str:
        step = self._step(step_id)
        approval_id = f"apr_{uuid.uuid4().hex[:16]}"
        self.ledger.db.conn.execute(
            """INSERT INTO approvals
               (id, step_id, args_hash, status, created_at)
               VALUES (?, ?, ?, 'pending', ?)""",
            (approval_id, step_id, step["args_hash"], utcnow()),
        )
        self.ledger.db.conn.execute(
            "UPDATE steps SET status = ? WHERE id = ?",
            (StepStatus.WAITING_APPROVAL.value, step_id),
        )
        self.ledger.db.conn.commit()
        return approval_id

    def approve(self, approval_id: str, approved_by: str) -> None:
        now = utcnow()
        cur = self.ledger.db.conn.execute(
            """UPDATE approvals SET status = 'approved', approved_by = ?, decided_at = ?
               WHERE id = ? AND status = 'pending'""",
            (approved_by, now, approval_id),
        )
        if cur.rowcount != 1:
            raise ValueError("Approval not found or already decided")
        self.ledger.db.conn.commit()

    def execute_plan(self, plan_id: str) -> list[dict[str, Any]]:
        steps = self.ledger.db.conn.execute(
            "SELECT id FROM steps WHERE plan_id = ? ORDER BY sequence",
            (plan_id,),
        ).fetchall()
        receipts: list[dict[str, Any]] = []
        self.ledger.db.conn.execute("UPDATE plans SET status = 'running' WHERE id = ?", (plan_id,))
        self.ledger.db.conn.commit()
        for row in steps:
            receipt = self.execute_step(row["id"])
            receipts.append(receipt)
            if not receipt["ok"]:
                self.ledger.db.conn.execute("UPDATE plans SET status = 'failed' WHERE id = ?", (plan_id,))
                self.ledger.db.conn.commit()
                return receipts
        self.ledger.db.conn.execute("UPDATE plans SET status = 'succeeded' WHERE id = ?", (plan_id,))
        self.ledger.db.conn.commit()
        return receipts

    def execute_step(self, step_id: str) -> dict[str, Any]:
        step = self._step(step_id)
        args = json.loads(step["args_json"])
        required_claims = json.loads(step["required_claims_json"])
        obligations = json.loads(step["obligations_json"])

        for claim_id in required_claims:
            state = self.support.evaluate(claim_id).state.value
            if state not in {"supported", "challenged"}:
                return self._receipt(step_id, False, {}, [], f"Required claim {claim_id} is {state}")

        risk = RiskLevel(step["risk"])
        if risk in {RiskLevel.MEDIUM, RiskLevel.HIGH}:
            approval = self.ledger.db.conn.execute(
                """SELECT * FROM approvals
                   WHERE step_id = ? AND args_hash = ? AND status = 'approved'
                   ORDER BY created_at DESC LIMIT 1""",
                (step_id, step["args_hash"]),
            ).fetchone()
            if approval is None:
                self.ledger.db.conn.execute(
                    "UPDATE steps SET status = ? WHERE id = ?",
                    (StepStatus.WAITING_APPROVAL.value, step_id),
                )
                self.ledger.db.conn.commit()
                return self._receipt(step_id, False, {}, [], "Exact step approval is required")

        action = self.actions.get(step["action_type"])
        if action is None:
            return self._receipt(step_id, False, {}, [], f"Unknown action type: {step['action_type']}")

        self.ledger.db.conn.execute(
            "UPDATE steps SET status = ? WHERE id = ?",
            (StepStatus.RUNNING.value, step_id),
        )
        self.ledger.db.conn.commit()
        try:
            result = action(args)
        except Exception as exc:  # pragma: no cover - defensive boundary
            result = ActionResult(ok=False, error=f"{type(exc).__name__}: {exc}")

        checks = [self._verify(obligation, result.outputs) for obligation in obligations]
        checks_ok = bool(obligations) and all(check.ok for check in checks)
        ok = result.ok and checks_ok
        error = result.error
        if result.ok and not obligations:
            error = "No proof obligations were supplied"
        elif result.ok and not checks_ok:
            error = "One or more proof obligations failed"
        return self._receipt(
            step_id,
            ok,
            result.outputs,
            [asdict(check) for check in checks],
            error,
        )

    def _receipt(
        self,
        step_id: str,
        ok: bool,
        outputs: dict[str, Any],
        checks: list[dict[str, Any]],
        error: str | None,
    ) -> dict[str, Any]:
        receipt_id = f"rcp_{uuid.uuid4().hex[:16]}"
        self.ledger.db.conn.execute(
            """INSERT INTO action_receipts
               (id, step_id, ok, outputs_json, checks_json, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                receipt_id,
                step_id,
                int(ok),
                stable_json(outputs),
                stable_json(checks),
                error,
                utcnow(),
            ),
        )
        self.ledger.db.conn.execute(
            "UPDATE steps SET status = ? WHERE id = ?",
            (StepStatus.SUCCEEDED.value if ok else StepStatus.FAILED.value, step_id),
        )
        self.ledger.db.conn.commit()
        return {
            "receipt_id": receipt_id,
            "step_id": step_id,
            "ok": ok,
            "outputs": outputs,
            "checks": checks,
            "error": error,
        }

    def _verify(self, obligation: dict[str, Any], outputs: dict[str, Any]) -> ObligationResult:
        kind = obligation.get("type")
        if kind == "file_exists":
            path = self._safe_path(str(obligation["path"]))
            ok = path.is_file()
            return ObligationResult(obligation, ok, f"{path} {'exists' if ok else 'does not exist'}")
        if kind == "sha256_equals":
            path = self._safe_path(str(obligation["path"]))
            expected = str(obligation["sha256"])
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
            return ObligationResult(obligation, actual == expected, f"actual={actual}")
        if kind == "output_equals":
            key = str(obligation["key"])
            expected = obligation.get("value")
            actual = outputs.get(key)
            return ObligationResult(obligation, actual == expected, f"actual={actual!r}")
        if kind == "output_contains":
            key = str(obligation["key"])
            needle = str(obligation["value"])
            actual = str(outputs.get(key, ""))
            return ObligationResult(obligation, needle in actual, f"actual={actual!r}")
        return ObligationResult(obligation, False, f"Unknown obligation type: {kind}")

    def _write_text(self, args: dict[str, Any]) -> ActionResult:
        path = self._safe_path(str(args["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        text = str(args["text"])
        path.write_text(text, encoding="utf-8")
        return ActionResult(
            ok=True,
            outputs={
                "path": str(path),
                "bytes_written": len(text.encode("utf-8")),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            },
        )

    def _hash_file(self, args: dict[str, Any]) -> ActionResult:
        path = self._safe_path(str(args["path"]))
        if not path.is_file():
            return ActionResult(ok=False, error=f"File does not exist: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return ActionResult(ok=True, outputs={"path": str(path), "sha256": digest})

    def _copy_file(self, args: dict[str, Any]) -> ActionResult:
        source = self._safe_path(str(args["source"]))
        destination = self._safe_path(str(args["destination"]))
        if not source.is_file():
            return ActionResult(ok=False, error=f"Source file does not exist: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return ActionResult(
            ok=True,
            outputs={
                "source": str(source),
                "destination": str(destination),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            },
        )

    def _safe_path(self, raw: str) -> Path:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.workspace):
            raise PermissionError(f"Path escapes workspace: {raw}")
        return resolved

    def _step(self, step_id: str):
        row = self.ledger.db.conn.execute("SELECT * FROM steps WHERE id = ?", (step_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown step: {step_id}")
        return row
