from __future__ import annotations

import json
import shutil
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
    finding_detail_json TEXT NOT NULL DEFAULT '{}',
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
        self._migrate()
        self.ledger.db.conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after the first deployed schema.

        The Railway volume holds databases created before finding_detail_json
        existed; CREATE TABLE IF NOT EXISTS will not add the column to them.
        """
        cols = {
            row["name"]
            for row in self.ledger.db.conn.execute("PRAGMA table_info(case_claims)").fetchall()
        }
        if "finding_detail_json" not in cols:
            self.ledger.db.conn.execute(
                "ALTER TABLE case_claims ADD COLUMN finding_detail_json TEXT NOT NULL DEFAULT '{}'"
            )
        # Phase 2A: memory-graph scoping + provenance fields (nullable; legacy rows keep NULL).
        if "graph_id" not in cols:
            self.ledger.db.conn.execute("ALTER TABLE case_claims ADD COLUMN graph_id TEXT")
        if "origin_json" not in cols:
            self.ledger.db.conn.execute("ALTER TABLE case_claims ADD COLUMN origin_json TEXT")
        if "epistemic_status" not in cols:
            self.ledger.db.conn.execute("ALTER TABLE case_claims ADD COLUMN epistemic_status TEXT")
        self.ledger.db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_case_claims_graph ON case_claims(graph_id)"
        )
        # Phase 2A placeholder: counterexample memory (no writes until Phase 2B).
        self.ledger.db.conn.execute(
            """CREATE TABLE IF NOT EXISTS counterexamples (
                id TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                graph_id TEXT,
                run_id TEXT,
                dataset_id TEXT,
                world_json TEXT,
                measurements_json TEXT,
                failure_json TEXT,
                found_by TEXT,
                minimal_known INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )"""
        )
        # Phase 2B: counterexamples become queryable by graph/claim/case.
        self.ledger.db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_counterexamples_graph ON counterexamples(graph_id)"
        )
        self.ledger.db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_counterexamples_claim ON counterexamples(claim_id)"
        )
        self.ledger.db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_counterexamples_case ON counterexamples(case_id)"
        )

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

    def delete_case(self, case_id: str) -> dict[str, Any]:
        self.get_case(case_id)
        case_root = (self.workspace / "cases").resolve()
        case_dir = self.case_dir(case_id).resolve()
        if case_dir.parent != case_root or case_dir.name != case_id:
            raise ValueError("Unsafe case workspace path")

        artifacts_removed = 0
        if case_dir.exists():
            artifacts_removed = sum(1 for path in case_dir.rglob("*") if path.is_file())

        conn = self.ledger.db.conn
        claim_rows = conn.execute(
            """SELECT DISTINCT claim_id
               FROM case_claims
               WHERE case_id = ?""",
            (case_id,),
        ).fetchall()
        claim_ids = [row["claim_id"] for row in claim_rows]
        owned_claim_ids = [
            claim_id
            for claim_id in claim_ids
            if conn.execute(
                "SELECT COUNT(*) AS n FROM case_claims WHERE claim_id = ? AND case_id <> ?",
                (claim_id, case_id),
            ).fetchone()["n"] == 0
        ]

        deleted_counts: dict[str, int] = {}

        def delete(sql: str, params: tuple[Any, ...]) -> None:
            cursor = conn.execute(sql, params)
            deleted_counts[sql.split()[2]] = deleted_counts.get(sql.split()[2], 0) + cursor.rowcount

        try:
            conn.execute("BEGIN")
            delete("DELETE FROM case_reports WHERE case_id = ?", (case_id,))
            delete("DELETE FROM case_claims WHERE case_id = ?", (case_id,))
            delete("DELETE FROM case_runs WHERE case_id = ?", (case_id,))
            delete("DELETE FROM analysis_plans WHERE case_id = ?", (case_id,))
            delete("DELETE FROM case_files WHERE case_id = ?", (case_id,))
            delete("DELETE FROM counterexamples WHERE case_id = ?", (case_id,))

            evidence_ids: list[str] = []
            if owned_claim_ids:
                placeholders = ",".join("?" for _ in owned_claim_ids)
                evidence_rows = conn.execute(
                    f"SELECT DISTINCT evidence_id FROM attestations WHERE claim_id IN ({placeholders})",
                    tuple(owned_claim_ids),
                ).fetchall()
                evidence_ids = [row["evidence_id"] for row in evidence_rows]
                delete(f"DELETE FROM contradictions WHERE claim_a IN ({placeholders}) OR claim_b IN ({placeholders})", tuple(owned_claim_ids) * 2)
                proof_rows = conn.execute(
                    f"SELECT DISTINCT proof_id FROM proof_premises WHERE premise_claim_id IN ({placeholders})",
                    tuple(owned_claim_ids),
                ).fetchall()
                proof_ids = [row["proof_id"] for row in proof_rows]
                if proof_ids:
                    proof_placeholders = ",".join("?" for _ in proof_ids)
                    delete(f"DELETE FROM proof_premises WHERE proof_id IN ({proof_placeholders})", tuple(proof_ids))
                    delete(f"DELETE FROM proofs WHERE id IN ({proof_placeholders})", tuple(proof_ids))
                delete(f"DELETE FROM proof_premises WHERE premise_claim_id IN ({placeholders})", tuple(owned_claim_ids))
                delete(f"DELETE FROM proof_premises WHERE proof_id IN (SELECT id FROM proofs WHERE conclusion_claim_id IN ({placeholders}))", tuple(owned_claim_ids))
                delete(f"DELETE FROM proofs WHERE conclusion_claim_id IN ({placeholders})", tuple(owned_claim_ids))
                delete(f"DELETE FROM attestations WHERE claim_id IN ({placeholders})", tuple(owned_claim_ids))
                delete(f"DELETE FROM relation_claims WHERE claim_id IN ({placeholders})", tuple(owned_claim_ids))
                delete(f"DELETE FROM events WHERE entity_id IN ({placeholders})", tuple(owned_claim_ids))
                delete(f"DELETE FROM claims WHERE id IN ({placeholders})", tuple(owned_claim_ids))

            orphan_evidence_ids = [
                evidence_id
                for evidence_id in evidence_ids
                if conn.execute(
                    "SELECT COUNT(*) AS n FROM attestations WHERE evidence_id = ?",
                    (evidence_id,),
                ).fetchone()["n"] == 0
            ]
            if orphan_evidence_ids:
                evidence_placeholders = ",".join("?" for _ in orphan_evidence_ids)
                delete(f"DELETE FROM events WHERE entity_id IN ({evidence_placeholders})", tuple(orphan_evidence_ids))
                delete(f"DELETE FROM evidence WHERE id IN ({evidence_placeholders})", tuple(orphan_evidence_ids))

            delete("DELETE FROM research_cases WHERE id = ?", (case_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        if case_dir.exists():
            try:
                shutil.rmtree(case_dir)
            except Exception as exc:
                raise RuntimeError("Case artifacts could not be removed") from exc

        return {
            "deleted": True,
            "case_id": case_id,
            "artifacts_removed": artifacts_removed,
            "records_removed": deleted_counts,
        }

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
        finding_detail: dict[str, Any] | None = None,
    ) -> None:
        self.ledger.db.conn.execute(
            """INSERT OR IGNORE INTO case_claims
               (case_id, run_id, claim_id, finding_type, source_candidate_id, finding_detail_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (case_id, run_id, claim_id, finding_type, source_candidate_id,
             stable_json(finding_detail or {}), utcnow()),
        )
        # If the row already existed (re-run), refresh its finding detail so the
        # public verdict reflects the most recent evaluation.
        self.ledger.db.conn.execute(
            """UPDATE case_claims SET finding_type = ?, finding_detail_json = ?, run_id = ?
               WHERE case_id = ? AND claim_id = ?
                 AND IFNULL(source_candidate_id, '') = IFNULL(?, '')""",
            (finding_type, stable_json(finding_detail or {}), run_id,
             case_id, claim_id, source_candidate_id),
        )
        self.ledger.db.conn.commit()

    def stamp_run_claims(
        self,
        *,
        case_id: str,
        run_id: str,
        graph_id: str | None,
        origin: dict[str, Any] | None = None,
    ) -> int:
        """Phase 2A provenance stamp: set graph_id/origin_json on all claims of one run.

        Applied once after import so engine call sites stay untouched. Legacy
        rows (other runs) are never modified.
        """
        cursor = self.ledger.db.conn.execute(
            """UPDATE case_claims SET graph_id = ?, origin_json = ?
               WHERE case_id = ? AND run_id = ?""",
            (graph_id, stable_json(origin or {}), case_id, run_id),
        )
        self.ledger.db.conn.commit()
        return cursor.rowcount

    def graph_claims(self, graph_id: str) -> list[dict[str, Any]]:
        """Claims scoped to a memory graph. Excludes legacy NULL-graph rows."""
        from .semantics import public_state

        rows = self.ledger.db.conn.execute(
            """SELECT cc.*, c.canonical_text, c.status,
                      (SELECT COUNT(*) FROM counterexamples cx
                       WHERE cx.claim_id = cc.claim_id AND cx.graph_id = cc.graph_id)
                      AS counterexample_count
               FROM case_claims cc JOIN claims c ON c.id = cc.claim_id
               WHERE cc.graph_id = ? ORDER BY cc.created_at""",
            (graph_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            detail = json.loads(item.pop("finding_detail_json", "{}") or "{}")
            item["finding_detail"] = detail
            item["verdict"] = public_state(item.get("finding_type"))
            origin_raw = item.pop("origin_json", None)
            item["origin"] = json.loads(origin_raw) if origin_raw else None
            out.append(item)
        return out

    # ------------------------------------------------------------------
    # Phase 2B: counterexample memory
    # ------------------------------------------------------------------
    def record_counterexample(
        self,
        *,
        claim_id: str,
        case_id: str,
        graph_id: str | None,
        run_id: str | None,
        dataset_id: str | None,
        found_by: str,
        failure: dict[str, Any],
        world: dict[str, Any] | None = None,
        measurements: dict[str, Any] | None = None,
        minimal_known: bool = False,
    ) -> dict[str, Any]:
        """Insert one counterexample row. Never touches claims or case_claims."""
        counterexample_id = new_id("cx")
        self.ledger.db.conn.execute(
            """INSERT INTO counterexamples
               (id, claim_id, case_id, graph_id, run_id, dataset_id,
                world_json, measurements_json, failure_json, found_by, minimal_known, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                counterexample_id, claim_id, case_id, graph_id, run_id, dataset_id,
                stable_json(world) if world is not None else None,
                stable_json(measurements) if measurements is not None else None,
                stable_json(failure),
                found_by,
                1 if minimal_known else 0,
                utcnow(),
            ),
        )
        self.ledger.db.conn.commit()
        return self.get_counterexample(counterexample_id)

    def get_counterexample(self, counterexample_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM counterexamples WHERE id = ?", (counterexample_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown counterexample: {counterexample_id}")
        return self._counterexample_row(row)

    @staticmethod
    def _counterexample_row(row: Any) -> dict[str, Any]:
        item = dict(row)
        for src, dst in (("world_json", "world"), ("measurements_json", "measurements"), ("failure_json", "failure")):
            raw = item.pop(src, None)
            item[dst] = json.loads(raw) if raw else None
        item["minimal_known"] = bool(item.get("minimal_known"))
        return item

    def graph_counterexamples(self, graph_id: str) -> list[dict[str, Any]]:
        """Counterexamples scoped to one memory graph (never NULL-graph rows)."""
        rows = self.ledger.db.conn.execute(
            "SELECT * FROM counterexamples WHERE graph_id = ? ORDER BY created_at",
            (graph_id,),
        ).fetchall()
        return [self._counterexample_row(row) for row in rows]

    def case_counterexamples(self, case_id: str) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT * FROM counterexamples WHERE case_id = ? ORDER BY created_at",
            (case_id,),
        ).fetchall()
        return [self._counterexample_row(row) for row in rows]

    def graph_memory_summary(self, graph_id: str) -> dict[str, Any]:
        """Aggregate memory view for one graph: claim verdicts, counterexample
        counts, observation ledger counts, and which datasets support/refute/
        challenge claims. Scoped strictly to graph_id — legacy NULL-graph rows
        and other graphs are never included."""
        from . import observations
        from .semantics import public_state

        claims = self.graph_claims(graph_id)
        counterexamples = self.graph_counterexamples(graph_id)

        verdict_counts: dict[str, int] = {}
        dataset_relations: dict[str, dict[str, int]] = {}

        def _dataset_bucket(dataset_id: str) -> dict[str, int]:
            return dataset_relations.setdefault(
                dataset_id, {"supports": 0, "refutes": 0, "challenges": 0}
            )

        supporting_verdicts = {"committed", "supported_association"}
        for claim in claims:
            verdict = claim.get("verdict") or public_state(claim.get("finding_type"))
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            if verdict in supporting_verdicts:
                for dataset_id in (claim.get("origin") or {}).get("dataset_ids", []) or []:
                    _dataset_bucket(dataset_id)["supports"] += 1

        counterexamples_by_found_by: dict[str, int] = {}
        for cx in counterexamples:
            found_by = cx.get("found_by") or "unknown"
            counterexamples_by_found_by[found_by] = counterexamples_by_found_by.get(found_by, 0) + 1
            if cx.get("dataset_id"):
                effect = (cx.get("failure") or {}).get("epistemic_effect")
                key = "refutes" if effect == "refutes" else "challenges"
                _dataset_bucket(cx["dataset_id"])[key] += 1

        case_ids = sorted({claim["case_id"] for claim in claims})
        observation_counts = {
            case_id: observations.observation_count(self.case_dir(case_id))
            for case_id in case_ids
        }
        return {
            "graph_id": graph_id,
            "claim_count": len(claims),
            "claims_by_verdict": verdict_counts,
            "counterexample_count": len(counterexamples),
            "counterexamples_by_found_by": counterexamples_by_found_by,
            "observation_count": sum(observation_counts.values()),
            "observations_by_case": observation_counts,
            "dataset_relations": dataset_relations,
        }

    def case_claims(self, case_id: str) -> list[dict[str, Any]]:
        from .semantics import public_state

        rows = self.ledger.db.conn.execute(
            """SELECT cc.*, c.canonical_text, c.status
               FROM case_claims cc JOIN claims c ON c.id = cc.claim_id
               WHERE cc.case_id = ? ORDER BY cc.created_at""",
            (case_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            detail = json.loads(item.pop("finding_detail_json", "{}") or "{}")
            finding_type = item.get("finding_type")
            verdict = public_state(finding_type)
            # Keep the raw claim lifecycle status under internal_status; the
            # public verdict is the spec-mandated state derived from finding_type.
            item["internal_status"] = item.get("status")
            item["verdict"] = verdict
            # Affirmative candidate text must be labelled as a hypothesis once it
            # is anything other than a committed finding.
            item["hypothesis_text"] = detail.get("hypothesis_text") or item.get("canonical_text")
            item["display_label"] = (
                "Committed finding" if verdict == "committed" else "Candidate hypothesis"
            )
            # Surface the structured verdict fields at the top level.
            for key in (
                "verdict_reason", "passed_checks", "failed_checks", "candidate_score",
                "baseline_score", "held_out_score", "cross_seed_summary",
                "influence_warning", "final_status", "is_candidate_hypothesis",
            ):
                if key in detail:
                    item[key] = detail[key]
            item["finding_detail"] = detail
            out.append(item)
        return out

    def case_claim_counts(self, case_id: str) -> dict[str, Any]:
        """Aggregate counts for the case dashboard, including run candidate totals."""
        from .semantics import public_state

        claims = self.case_claims(case_id)
        counts = {
            "committed": 0, "rejected": 0, "artifact": 0, "provisional": 0, "unresolved": 0,
            "not_supported": 0, "inconclusive": 0, "functional_form_rejected": 0,
            "supported_association": 0, "regime_dependent": 0,
        }
        for claim in claims:
            state = public_state(claim.get("finding_type"))
            counts[state] = counts.get(state, 0) + 1

        # generated_candidates comes from the most recent run's engine metadata,
        # never reported as zero when the run actually screened candidates.
        generated = 0
        structural = 0
        runs = self.list_runs(case_id)
        if runs:
            result = runs[0].get("result", {}) or {}
            generated = int(result.get("candidate_count", 0) or 0)
            belief = result.get("belief_import", {}) or {}
            structural = int(belief.get("artifact_count", 0) or 0)
        persisted = len(claims)
        return {
            "generated_candidates": max(generated, persisted),
            "persisted_findings": persisted,
            "committed_count": counts["committed"],
            "rejected_count": counts["rejected"],
            "artifact_count": counts["artifact"],
            "provisional_count": counts["provisional"],
            "unresolved_count": counts["unresolved"],
            "not_supported_count": counts["not_supported"],
            "inconclusive_count": counts["inconclusive"],
            "functional_form_rejected_count": counts["functional_form_rejected"],
            "supported_association_count": counts["supported_association"],
            "regime_dependent_count": counts["regime_dependent"],
            "filtered_count": max(0, max(generated, persisted) - persisted),
            "structural_relations_detected": structural,
        }

    def add_report(self, case_id: str, run_id: str | None, *, format: str, path: str, content_hash: str) -> dict[str, Any]:
        report_id = new_id("report")
        self.ledger.db.conn.execute(
            """INSERT INTO case_reports (id, case_id, run_id, format, path, content_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (report_id, case_id, run_id, format, path, content_hash, utcnow()),
        )
        self.ledger.db.conn.commit()
        return {"id": report_id, "case_id": case_id, "run_id": run_id, "format": format, "path": path, "content_hash": content_hash}
