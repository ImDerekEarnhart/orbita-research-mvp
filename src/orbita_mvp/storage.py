from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orbita import EpistemicLedger


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


CASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_cases (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    goal TEXT NOT NULL,
    mode TEXT NOT NULL,
    domain_hint TEXT,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_files (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES research_cases(id),
    original_name TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    parse_status TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    extracted_path TEXT,
    profile_json TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_plans (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES research_cases(id),
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    plan_hash TEXT NOT NULL,
    compiler TEXT NOT NULL,
    created_at TEXT NOT NULL,
    approved_at TEXT,
    approved_by TEXT,
    UNIQUE(case_id, version)
);

CREATE TABLE IF NOT EXISTS case_runs (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES research_cases(id),
    plan_id TEXT NOT NULL REFERENCES analysis_plans(id),
    engine_run_id TEXT,
    status TEXT NOT NULL,
    result_json TEXT NOT NULL,
    ledger_path TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS case_claims (
    case_id TEXT NOT NULL REFERENCES research_cases(id),
    run_id TEXT REFERENCES case_runs(id),
    claim_id TEXT NOT NULL REFERENCES claims(id),
    finding_type TEXT NOT NULL,
    source_candidate_id TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY(case_id, claim_id, source_candidate_id)
);

CREATE TABLE IF NOT EXISTS case_reports (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES research_cases(id),
    run_id TEXT REFERENCES case_runs(id),
    format TEXT NOT NULL,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_case_files_case ON case_files(case_id, created_at);
CREATE INDEX IF NOT EXISTS idx_case_plans_case ON analysis_plans(case_id, version);
CREATE INDEX IF NOT EXISTS idx_case_runs_case ON case_runs(case_id, started_at);
CREATE INDEX IF NOT EXISTS idx_case_claims_case ON case_claims(case_id, created_at);
"""


class CaseStore:
    def __init__(self, ledger: EpistemicLedger, workspace: str | Path):
        self.ledger = ledger
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.ledger.db.conn.executescript(CASE_SCHEMA)
        self.ledger.db.conn.commit()

    def create_case(
        self,
        *,
        name: str,
        goal: str = "",
        domain_hint: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        case_id = new_id("case")
        now = utcnow()
        mode = "guided" if goal.strip() else "open_discovery"
        self.ledger.db.conn.execute(
            """INSERT INTO research_cases
               (id, name, goal, mode, domain_hint, status, metadata_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'created', ?, ?, ?)""",
            (case_id, name.strip() or "Untitled research case", goal.strip(), mode, domain_hint, stable_json(metadata or {}), now, now),
        )
        self.ledger.db.conn.commit()
        self.case_dir(case_id).mkdir(parents=True, exist_ok=True)
        return self.get_case(case_id)

    def case_dir(self, case_id: str) -> Path:
        return self.workspace / "cases" / case_id

    def update_case(self, case_id: str, **fields: Any) -> dict[str, Any]:
        allowed = {"name", "goal", "domain_hint", "status", "mode"}
        items = [(k, v) for k, v in fields.items() if k in allowed]
        if not items:
            return self.get_case(case_id)
        if "goal" in fields and "mode" not in fields:
            items.append(("mode", "guided" if str(fields["goal"]).strip() else "open_discovery"))
        sql = ", ".join(f"{k} = ?" for k, _ in items) + ", updated_at = ?"
        self.ledger.db.conn.execute(
            f"UPDATE research_cases SET {sql} WHERE id = ?",
            (*[v for _, v in items], utcnow(), case_id),
        )
        self.ledger.db.conn.commit()
        return self.get_case(case_id)

    def get_case(self, case_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute("SELECT * FROM research_cases WHERE id = ?", (case_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown case: {case_id}")
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json"))
        item["files"] = self.list_files(case_id)
        item["plans"] = self.list_plans(case_id)
        item["runs"] = self.list_runs(case_id)
        return item

    def list_cases(self) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute("SELECT id FROM research_cases ORDER BY created_at DESC").fetchall()
        return [self.get_case(row["id"]) for row in rows]

    def add_file_record(self, case_id: str, record: dict[str, Any]) -> dict[str, Any]:
        file_id = record.get("id") or new_id("file")
        self.ledger.db.conn.execute(
            """INSERT INTO case_files
               (id, case_id, original_name, stored_path, media_type, size_bytes, sha256,
                parse_status, artifact_kind, extracted_path, profile_json, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                file_id,
                case_id,
                record["original_name"],
                record["stored_path"],
                record.get("media_type", "application/octet-stream"),
                int(record.get("size_bytes", 0)),
                record["sha256"],
                record.get("parse_status", "preserved"),
                record.get("artifact_kind", "unknown"),
                record.get("extracted_path"),
                stable_json(record.get("profile", {})),
                record.get("error"),
                utcnow(),
            ),
        )
        self.update_case(case_id, status="ingested")
        self.ledger.db.conn.commit()
        return self.get_file(file_id)

    def get_file(self, file_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute("SELECT * FROM case_files WHERE id = ?", (file_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown file: {file_id}")
        item = dict(row)
        item["profile"] = json.loads(item.pop("profile_json"))
        return item

    def list_files(self, case_id: str) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT id FROM case_files WHERE case_id = ? ORDER BY created_at", (case_id,)
        ).fetchall()
        return [self.get_file(row["id"]) for row in rows]

    def save_plan(self, case_id: str, plan: dict[str, Any], *, compiler: str) -> dict[str, Any]:
        import hashlib

        current = self.ledger.db.conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM analysis_plans WHERE case_id = ?", (case_id,)
        ).fetchone()["version"]
        plan_id = new_id("plan")
        text = stable_json(plan)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.ledger.db.conn.execute(
            """INSERT INTO analysis_plans
               (id, case_id, version, status, plan_json, plan_hash, compiler, created_at)
               VALUES (?, ?, ?, 'proposed', ?, ?, ?, ?)""",
            (plan_id, case_id, int(current) + 1, text, digest, compiler, utcnow()),
        )
        self.update_case(case_id, status="plan_ready")
        self.ledger.db.conn.commit()
        return self.get_plan(plan_id)

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute("SELECT * FROM analysis_plans WHERE id = ?", (plan_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown plan: {plan_id}")
        item = dict(row)
        item["plan"] = json.loads(item.pop("plan_json"))
        return item

    def list_plans(self, case_id: str) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT id FROM analysis_plans WHERE case_id = ? ORDER BY version DESC", (case_id,)
        ).fetchall()
        return [self.get_plan(row["id"]) for row in rows]

    def approve_plan(self, plan_id: str, *, reviewer: str) -> dict[str, Any]:
        plan = self.get_plan(plan_id)
        self.ledger.db.conn.execute(
            "UPDATE analysis_plans SET status = 'approved', approved_at = ?, approved_by = ? WHERE id = ?",
            (utcnow(), reviewer, plan_id),
        )
        self.update_case(plan["case_id"], status="approved")
        self.ledger.db.conn.commit()
        return self.get_plan(plan_id)

    def revise_plan(self, plan_id: str, plan: dict[str, Any], *, compiler: str) -> dict[str, Any]:
        """Create a new immutable plan version from an existing proposal.

        Approved plans are never edited in place. A human or external AI review
        therefore produces a new proposal with a new hash and version number.
        """
        current = self.get_plan(plan_id)
        return self.save_plan(current["case_id"], plan, compiler=compiler)

    def create_run(self, case_id: str, plan_id: str) -> dict[str, Any]:
        run_id = new_id("run")
        self.ledger.db.conn.execute(
            """INSERT INTO case_runs
               (id, case_id, plan_id, status, result_json, started_at)
               VALUES (?, ?, ?, 'running', '{}', ?)""",
            (run_id, case_id, plan_id, utcnow()),
        )
        self.update_case(case_id, status="running")
        self.ledger.db.conn.commit()
        return self.get_run(run_id)

    def finish_run(
        self,
        run_id: str,
        *,
        result: dict[str, Any],
        engine_run_id: str | None,
        ledger_path: str | None,
        status: str = "completed",
    ) -> dict[str, Any]:
        row = self.ledger.db.conn.execute("SELECT case_id FROM case_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown run: {run_id}")
        self.ledger.db.conn.execute(
            """UPDATE case_runs SET status = ?, result_json = ?, engine_run_id = ?, ledger_path = ?, completed_at = ?
               WHERE id = ?""",
            (status, stable_json(result), engine_run_id, ledger_path, utcnow(), run_id),
        )
        self.update_case(row["case_id"], status=status)
        self.ledger.db.conn.commit()
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute("SELECT * FROM case_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown run: {run_id}")
        item = dict(row)
        item["result"] = json.loads(item.pop("result_json"))
        return item

    def list_runs(self, case_id: str) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT id FROM case_runs WHERE case_id = ? ORDER BY started_at DESC", (case_id,)
        ).fetchall()
        return [self.get_run(row["id"]) for row in rows]

    def link_claim(
        self,
        *,
        case_id: str,
        run_id: str | None,
        claim_id: str,
        finding_type: str,
        source_candidate_id: str | None,
    ) -> None:
        self.ledger.db.conn.execute(
            """INSERT OR IGNORE INTO case_claims
               (case_id, run_id, claim_id, finding_type, source_candidate_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (case_id, run_id, claim_id, finding_type, source_candidate_id, utcnow()),
        )
        self.ledger.db.conn.commit()

    def case_claims(self, case_id: str) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            """SELECT cc.*, c.canonical_text, c.status
               FROM case_claims cc JOIN claims c ON c.id = cc.claim_id
               WHERE cc.case_id = ? ORDER BY cc.created_at""",
            (case_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def add_report(self, case_id: str, run_id: str | None, *, format: str, path: str, content_hash: str) -> dict[str, Any]:
        report_id = new_id("report")
        self.ledger.db.conn.execute(
            """INSERT INTO case_reports (id, case_id, run_id, format, path, content_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (report_id, case_id, run_id, format, path, content_hash, utcnow()),
        )
        self.ledger.db.conn.commit()
        return {"id": report_id, "case_id": case_id, "run_id": run_id, "format": format, "path": path, "content_hash": content_hash}
