from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from orbita import ActorRole, ClaimStatus, EpistemicLedger, EvidenceKind, Stance, SupportEngine


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _claim_key(text: str, scope: dict[str, Any] | None = None) -> str:
    raw = _stable_json({"text": " ".join(text.lower().split()), "scope": scope or {}})
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS mvp_claim_keys (
    claim_key TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claim_supersessions (
    id TEXT PRIMARY KEY,
    older_claim_id TEXT NOT NULL REFERENCES claims(id),
    newer_claim_id TEXT NOT NULL REFERENCES claims(id),
    rationale TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    UNIQUE(older_claim_id, newer_claim_id)
);

CREATE TABLE IF NOT EXISTS claim_checks (
    id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(id),
    run_id TEXT,
    check_name TEXT NOT NULL,
    passed INTEGER NOT NULL,
    score REAL,
    detail_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reexamination_queue (
    id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(id),
    trigger_type TEXT NOT NULL,
    trigger_id TEXT NOT NULL,
    impact TEXT NOT NULL,
    reason TEXT NOT NULL,
    remaining_support_json TEXT NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    resolution_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    UNIQUE(claim_id, trigger_type, trigger_id, status)
);

CREATE INDEX IF NOT EXISTS idx_supersessions_old ON claim_supersessions(older_claim_id, active);
CREATE INDEX IF NOT EXISTS idx_supersessions_new ON claim_supersessions(newer_claim_id, active);
CREATE INDEX IF NOT EXISTS idx_claim_checks_claim ON claim_checks(claim_id, created_at);
CREATE INDEX IF NOT EXISTS idx_reexam_status ON reexamination_queue(status, priority, created_at);
"""


class BeliefMemory:
    """Persistent belief lifecycle, supersession history, and dependency collapse.

    The underlying EpistemicLedger already stores claims, evidence, proofs,
    contradictions, and append-only events. This layer adds stable claim keys,
    explicit supersession links, test/check receipts, and a non-destructive
    re-examination queue.
    """

    def __init__(self, ledger: EpistemicLedger):
        self.ledger = ledger
        self.ledger.db.conn.executescript(MEMORY_SCHEMA)
        self.ledger.db.conn.commit()

    # ------------------------------------------------------------------
    # Stable claims and checks
    # ------------------------------------------------------------------
    def resolve_or_create_claim(
        self,
        text: str,
        *,
        scope: dict[str, Any] | None = None,
        claim_type: str = "finding",
        metadata: dict[str, Any] | None = None,
        actor: str = "research-compiler",
        actor_role: ActorRole = ActorRole.TOOL,
    ) -> tuple[str, bool]:
        key = _claim_key(text, scope)
        row = self.ledger.db.conn.execute(
            "SELECT claim_id FROM mvp_claim_keys WHERE claim_key = ?", (key,)
        ).fetchone()
        if row is not None:
            return row["claim_id"], False
        claim_id = self.ledger.add_claim(
            text,
            claim_type=claim_type,
            scope=scope or {},
            metadata={**(metadata or {}), "stable_claim_key": key},
            actor=actor,
            actor_role=actor_role,
        )
        self.ledger.db.conn.execute(
            "INSERT INTO mvp_claim_keys (claim_key, claim_id, created_at) VALUES (?, ?, ?)",
            (key, claim_id, _utcnow()),
        )
        self.ledger.db.conn.commit()
        return claim_id, True

    def record_check(
        self,
        claim_id: str,
        *,
        name: str,
        passed: bool,
        score: float | None = None,
        detail: dict[str, Any] | None = None,
        run_id: str | None = None,
        actor: str = "discovery-engine",
    ) -> str:
        self.ledger._require_claim(claim_id)
        check_id = _new_id("chk")
        self.ledger.db.conn.execute(
            """INSERT INTO claim_checks
               (id, claim_id, run_id, check_name, passed, score, detail_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                check_id,
                claim_id,
                run_id,
                name,
                int(bool(passed)),
                score,
                _stable_json(detail or {}),
                _utcnow(),
            ),
        )
        self.ledger._event(
            "claim",
            claim_id,
            "CHECK_RECORDED",
            {
                "check_id": check_id,
                "run_id": run_id,
                "name": name,
                "passed": bool(passed),
                "score": score,
                "detail": detail or {},
            },
            actor,
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()
        return check_id

    def attach_run_evidence(
        self,
        claim_id: str,
        *,
        run_id: str,
        finding: dict[str, Any],
        source_uri: str,
        support: bool,
    ) -> str:
        evidence_id = self.ledger.add_evidence(
            source_uri,
            finding.get("candidate", {}).get("statement", "Discovery finding"),
            source_kind=EvidenceKind.DATASET_ANALYSIS_RECEIPT,
            independence_key=f"run:{run_id}",
            content=_stable_json(finding),
            metadata={
                "run_id": run_id,
                "finding_hash": finding.get("sha256"),
                "final_status": finding.get("final_status"),
            },
            actor="discovery-engine",
            actor_role=ActorRole.TOOL,
        )
        raw_score = float(finding.get("verdict", {}).get("score", 0.0) or 0.0)
        confidence = max(0.0, min(1.0, abs(raw_score)))
        self.ledger.attest(
            claim_id,
            evidence_id,
            Stance.SUPPORT if support else Stance.REFUTE,
            confidence=confidence,
            actor="discovery-engine",
            actor_role=ActorRole.TOOL,
        )
        return evidence_id

    def synchronize_status(self, claim_id: str, *, rationale: str) -> str:
        claim = self.ledger.get_claim(claim_id)
        report = SupportEngine(self.ledger).evaluate(claim_id)
        current = claim["status"]
        if report.state.value == "supported" and current != ClaimStatus.COMMITTED.value:
            self.ledger.commit_claim(
                claim_id,
                actor="belief-policy",
                actor_role=ActorRole.POLICY,
                rationale=rationale,
            )
        elif report.state.value == "challenged" and current != ClaimStatus.CHALLENGED.value:
            self.ledger.set_claim_status(
                claim_id,
                ClaimStatus.CHALLENGED,
                rationale=rationale,
                actor="belief-policy",
                actor_role=ActorRole.POLICY,
            )
        elif report.state.value == "unsupported" and current == ClaimStatus.COMMITTED.value:
            # A failed support chain does not erase or automatically reject a claim.
            self.ledger.set_claim_status(
                claim_id,
                ClaimStatus.CHALLENGED,
                rationale=rationale,
                actor="belief-policy",
                actor_role=ActorRole.POLICY,
            )
        return SupportEngine(self.ledger).evaluate(claim_id).state.value

    def add_evidence(
        self,
        claim_id: str,
        *,
        source_uri: str,
        excerpt: str,
        source_kind: EvidenceKind,
        independence_key: str,
        stance: Stance,
        confidence: float = 1.0,
        content: str | bytes | None = None,
        metadata: dict[str, Any] | None = None,
        actor: str = "researcher",
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> dict[str, Any]:
        """Attach evidence and propagate any support-state change forward."""
        self.ledger._require_claim(claim_id)
        affected = [claim_id, *self.ledger.descendants_of_claim(claim_id)]
        before = self._reports(affected)
        evidence_id = self.ledger.add_evidence(
            source_uri,
            excerpt,
            source_kind=source_kind,
            independence_key=independence_key,
            content=content,
            metadata=metadata or {},
            actor=actor,
            actor_role=actor_role,
        )
        attestation_id = self.ledger.attest(
            claim_id,
            evidence_id,
            stance,
            confidence=confidence,
            actor=actor,
            actor_role=actor_role,
        )
        self.synchronize_status(claim_id, rationale=f"Evidence {evidence_id} attached")
        impacts = self.analyze_collapse(
            roots=[claim_id],
            before_reports=before,
            trigger_type="evidence_added",
            trigger_id=evidence_id,
            reason=f"{stance.value} evidence added from {source_uri}",
        )
        return {
            "claim_id": claim_id,
            "evidence_id": evidence_id,
            "attestation_id": attestation_id,
            "support": SupportEngine(self.ledger).evaluate(claim_id).as_dict(),
            "impacts": impacts,
        }

    def add_derivation(
        self,
        conclusion_claim_id: str,
        premise_claim_ids: Iterable[str],
        *,
        rule: str,
        metadata: dict[str, Any] | None = None,
        actor: str = "researcher",
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> dict[str, Any]:
        """Record an AND-connected derivation and propagate recovered support."""
        premises = list(dict.fromkeys(premise_claim_ids))
        affected = [conclusion_claim_id, *self.ledger.descendants_of_claim(conclusion_claim_id)]
        before = self._reports(affected)
        proof_id = self.ledger.add_proof(
            conclusion_claim_id,
            premises,
            rule=rule,
            metadata=metadata or {},
            actor=actor,
            actor_role=actor_role,
        )
        self.synchronize_status(conclusion_claim_id, rationale=f"Derivation {proof_id} recorded")
        impacts = self.analyze_collapse(
            roots=[conclusion_claim_id],
            before_reports=before,
            trigger_type="derivation_added",
            trigger_id=proof_id,
            reason=f"Derivation recorded using rule {rule}",
        )
        return {
            "proof_id": proof_id,
            "conclusion_claim_id": conclusion_claim_id,
            "premise_claim_ids": premises,
            "support": SupportEngine(self.ledger).evaluate(conclusion_claim_id).as_dict(),
            "impacts": impacts,
        }

    def add_contradiction(
        self,
        claim_a: str,
        claim_b: str,
        *,
        rationale: str,
        actor: str = "researcher",
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> dict[str, Any]:
        """Link contradictory claims and propagate the impact to descendants."""
        affected = {
            claim_a,
            claim_b,
            *self.ledger.descendants_of_claim(claim_a),
            *self.ledger.descendants_of_claim(claim_b),
        }
        before = self._reports(affected)
        contradiction_id = self.ledger.add_contradiction(
            claim_a,
            claim_b,
            rationale=rationale,
            actor=actor,
            actor_role=actor_role,
        )
        self.synchronize_status(claim_a, rationale=f"Contradiction {contradiction_id} linked")
        self.synchronize_status(claim_b, rationale=f"Contradiction {contradiction_id} linked")
        impacts = self.analyze_collapse(
            roots=[claim_a, claim_b],
            before_reports=before,
            trigger_type="contradiction",
            trigger_id=contradiction_id,
            reason=rationale,
        )
        return {
            "contradiction_id": contradiction_id,
            "claim_a": claim_a,
            "claim_b": claim_b,
            "impacts": impacts,
        }

    def impact_view(self, claim_id: str) -> dict[str, Any]:
        """Return current dependencies and the forward blast radius of a claim."""
        self.ledger._require_claim(claim_id)
        return {
            "claim": self.ledger.get_claim(claim_id),
            "current_support": SupportEngine(self.ledger).evaluate(claim_id).as_dict(),
            "dependencies": self._dependencies(claim_id),
            "dependents": self.ledger.descendants_of_claim(claim_id),
            "open_reexamination": [
                item for item in self.list_reexamination("open") if item["claim_id"] == claim_id
            ],
        }

    # ------------------------------------------------------------------
    # Supersession
    # ------------------------------------------------------------------
    def supersede(
        self,
        older_claim_id: str,
        newer_text: str,
        *,
        rationale: str,
        scope: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        actor: str = "researcher",
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> dict[str, Any]:
        self.ledger._require_claim(older_claim_id)
        descendants = self.ledger.descendants_of_claim(older_claim_id)
        before = self._reports([older_claim_id, *descendants])
        newer_claim_id, created = self.resolve_or_create_claim(
            newer_text,
            scope=scope,
            claim_type=self.ledger.get_claim(older_claim_id)["claim_type"],
            metadata={**(metadata or {}), "supersedes": older_claim_id},
            actor=actor,
            actor_role=actor_role,
        )
        relation_id = _new_id("sup")
        self.ledger.db.conn.execute(
            """INSERT OR IGNORE INTO claim_supersessions
               (id, older_claim_id, newer_claim_id, rationale, active, created_at)
               VALUES (?, ?, ?, ?, 1, ?)""",
            (relation_id, older_claim_id, newer_claim_id, rationale, _utcnow()),
        )
        self.ledger.set_claim_status(
            older_claim_id,
            ClaimStatus.SUPERSEDED,
            rationale=rationale,
            actor=actor,
            actor_role=actor_role,
        )
        self.ledger._event(
            "claim",
            newer_claim_id,
            "CLAIM_SUPERSEDES",
            {
                "older_claim_id": older_claim_id,
                "supersession_id": relation_id,
                "rationale": rationale,
            },
            actor,
            actor_role,
        )
        self.ledger.db.conn.commit()
        impacts = self.analyze_collapse(
            roots=[older_claim_id],
            before_reports=before,
            trigger_type="supersession",
            trigger_id=relation_id,
            reason=rationale,
        )
        return {
            "supersession_id": relation_id,
            "older_claim_id": older_claim_id,
            "newer_claim_id": newer_claim_id,
            "new_claim_created": created,
            "impacts": impacts,
        }

    def supersession_chain(self, claim_id: str) -> list[dict[str, Any]]:
        self.ledger._require_claim(claim_id)
        conn = self.ledger.db.conn
        earliest = claim_id
        visited: set[str] = set()
        while earliest not in visited:
            visited.add(earliest)
            row = conn.execute(
                """SELECT older_claim_id FROM claim_supersessions
                   WHERE newer_claim_id = ? AND active = 1 ORDER BY created_at DESC LIMIT 1""",
                (earliest,),
            ).fetchone()
            if row is None:
                break
            earliest = row["older_claim_id"]

        chain: list[dict[str, Any]] = []
        current = earliest
        visited.clear()
        while current not in visited:
            visited.add(current)
            claim = self.ledger.get_claim(current)
            row = conn.execute(
                """SELECT * FROM claim_supersessions
                   WHERE older_claim_id = ? AND active = 1 ORDER BY created_at DESC LIMIT 1""",
                (current,),
            ).fetchone()
            item = {
                "claim_id": current,
                "canonical_text": claim["canonical_text"],
                "status": claim["status"],
                "created_at": claim["created_at"],
                "supersession": None,
            }
            if row is not None:
                item["supersession"] = {
                    "id": row["id"],
                    "newer_claim_id": row["newer_claim_id"],
                    "rationale": row["rationale"],
                    "created_at": row["created_at"],
                }
            chain.append(item)
            if row is None:
                break
            current = row["newer_claim_id"]
        return chain

    # ------------------------------------------------------------------
    # Collapse analysis and queue
    # ------------------------------------------------------------------
    def revoke_evidence(
        self,
        evidence_id: str,
        *,
        rationale: str,
        actor: str = "researcher",
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> dict[str, Any]:
        affected = self.ledger.descendants_of_evidence(evidence_id)
        before = self._reports(affected)
        self.ledger.revoke_evidence(
            evidence_id, rationale=rationale, actor=actor, actor_role=actor_role
        )
        impacts = self.analyze_collapse(
            roots=affected,
            before_reports=before,
            trigger_type="evidence_revocation",
            trigger_id=evidence_id,
            reason=rationale,
        )
        return {"evidence_id": evidence_id, "impacts": impacts}

    def analyze_collapse(
        self,
        *,
        roots: Iterable[str],
        before_reports: dict[str, dict[str, Any]] | None,
        trigger_type: str,
        trigger_id: str,
        reason: str,
    ) -> list[dict[str, Any]]:
        claim_ids: set[str] = set(roots)
        for root in list(claim_ids):
            claim_ids.update(self.ledger.descendants_of_claim(root))
        before = before_reports or self._reports(claim_ids)
        after = self._reports(claim_ids)
        impacts: list[dict[str, Any]] = []
        for claim_id in sorted(claim_ids):
            left = before.get(claim_id)
            right = after.get(claim_id)
            if left is None or right is None:
                continue
            impact = self._classify_impact(left, right)
            if impact == "unaffected":
                continue
            item = {
                "claim_id": claim_id,
                "impact": impact,
                "before": left,
                "after": right,
                "reason": reason,
            }
            impacts.append(item)
            self.ledger._event(
                "claim",
                claim_id,
                "DEPENDENCY_IMPACT_DETECTED",
                {
                    "trigger_type": trigger_type,
                    "trigger_id": trigger_id,
                    "impact": impact,
                    "before_state": left["state"],
                    "after_state": right["state"],
                    "reason": reason,
                },
                "collapse-analyzer",
                ActorRole.TOOL,
            )
            if impact not in {"recovered"}:
                self._enqueue(
                    claim_id,
                    trigger_type=trigger_type,
                    trigger_id=trigger_id,
                    impact=impact,
                    reason=reason,
                    remaining_support=right,
                )
        self.ledger.db.conn.commit()
        return impacts

    def _classify_impact(self, before: dict[str, Any], after: dict[str, Any]) -> str:
        b_state, a_state = before["state"], after["state"]
        if b_state == a_state:
            changed_support = (
                before.get("direct_support_sources") != after.get("direct_support_sources")
                or before.get("satisfied_proofs") != after.get("satisfied_proofs")
                or before.get("broken_proofs") != after.get("broken_proofs")
                or before.get("contradictions") != after.get("contradictions")
            )
            if changed_support and a_state in {"supported", "challenged"}:
                return "weakened_but_still_supported"
            return "unaffected"
        if b_state in {"supported", "challenged"} and a_state in {"unsupported", "unknown"}:
            return "unsupported_must_reexamine"
        if b_state == "supported" and a_state == "challenged":
            return "challenged"
        if b_state in {"unsupported", "unknown", "challenged"} and a_state == "supported":
            return "recovered"
        return "support_changed"

    def _enqueue(
        self,
        claim_id: str,
        *,
        trigger_type: str,
        trigger_id: str,
        impact: str,
        reason: str,
        remaining_support: dict[str, Any],
    ) -> str:
        queue_id = _new_id("rq")
        priority = "high" if impact == "unsupported_must_reexamine" else "medium"
        try:
            self.ledger.db.conn.execute(
                """INSERT INTO reexamination_queue
                   (id, claim_id, trigger_type, trigger_id, impact, reason,
                    remaining_support_json, priority, status, resolution_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', '{}', ?)""",
                (
                    queue_id,
                    claim_id,
                    trigger_type,
                    trigger_id,
                    impact,
                    reason,
                    _stable_json(remaining_support),
                    priority,
                    _utcnow(),
                ),
            )
        except Exception as exc:
            if "UNIQUE constraint failed" not in str(exc):
                raise
            row = self.ledger.db.conn.execute(
                """SELECT id FROM reexamination_queue
                   WHERE claim_id = ? AND trigger_type = ? AND trigger_id = ? AND status = 'open'""",
                (claim_id, trigger_type, trigger_id),
            ).fetchone()
            return row["id"]
        return queue_id

    def list_reexamination(self, status: str = "open") -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            """SELECT q.*, c.canonical_text
               FROM reexamination_queue q JOIN claims c ON c.id = q.claim_id
               WHERE q.status = ?
               ORDER BY CASE q.priority WHEN 'high' THEN 0 ELSE 1 END, q.created_at""",
            (status,),
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["remaining_support"] = json.loads(item.pop("remaining_support_json"))
            item["resolution"] = json.loads(item.pop("resolution_json"))
            out.append(item)
        return out

    def resolve_reexamination(
        self,
        queue_id: str,
        *,
        resolution: dict[str, Any],
        actor: str = "researcher",
    ) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM reexamination_queue WHERE id = ?", (queue_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown re-examination task: {queue_id}")
        self.ledger.db.conn.execute(
            """UPDATE reexamination_queue
               SET status = 'resolved', resolution_json = ?, resolved_at = ? WHERE id = ?""",
            (_stable_json(resolution), _utcnow(), queue_id),
        )
        self.ledger._event(
            "claim",
            row["claim_id"],
            "REEXAMINATION_RESOLVED",
            {"queue_id": queue_id, "resolution": resolution},
            actor,
            ActorRole.HUMAN,
        )
        self.ledger.db.conn.commit()
        return {"queue_id": queue_id, "status": "resolved", "resolution": resolution}

    # ------------------------------------------------------------------
    # Full history reconstruction
    # ------------------------------------------------------------------
    def reconstruct_history(self, claim_id: str) -> dict[str, Any]:
        chain = self.supersession_chain(claim_id)
        claim_ids = [item["claim_id"] for item in chain]
        timeline: list[dict[str, Any]] = []
        for cid in claim_ids:
            for event in self.ledger.history("claim", cid):
                timeline.append(
                    {
                        "created_at": event["created_at"],
                        "claim_id": cid,
                        "kind": "event",
                        "event_type": event["event_type"],
                        "actor": event["actor"],
                        "actor_role": event["actor_role"],
                        "detail": event["payload"],
                    }
                )
            for check in self._checks(cid):
                timeline.append(
                    {
                        "created_at": check["created_at"],
                        "claim_id": cid,
                        "kind": "check",
                        "event_type": "CHECK_PASSED" if check["passed"] else "CHECK_FAILED",
                        "actor": "discovery-engine",
                        "actor_role": "tool",
                        "detail": check,
                    }
                )
        timeline.sort(key=lambda item: (item["created_at"], item["claim_id"], item["event_type"]))
        current_id = chain[-1]["claim_id"]
        return {
            "requested_claim_id": claim_id,
            "current_claim_id": current_id,
            "current_claim": self.ledger.get_claim(current_id),
            "current_support": SupportEngine(self.ledger).evaluate(current_id).as_dict(),
            "supersession_chain": chain,
            "timeline": timeline,
            "evidence": {cid: self._evidence(cid) for cid in claim_ids},
            "checks": {cid: self._checks(cid) for cid in claim_ids},
            "contradictions": {cid: self._contradictions(cid) for cid in claim_ids},
            "dependencies": {cid: self._dependencies(cid) for cid in claim_ids},
            "dependents": {cid: self.ledger.descendants_of_claim(cid) for cid in claim_ids},
            "reexamination": [
                row for row in self.list_reexamination("open") if row["claim_id"] in claim_ids
            ],
        }

    def _reports(self, claim_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        unique = list(dict.fromkeys(claim_ids))
        # Evaluate each root independently. A shared memo across roots can make a
        # symmetric contradiction appear directional because the first recursive
        # walk memoizes the other side while it is still on the call stack.
        engine = SupportEngine(self.ledger)
        return {claim_id: engine.evaluate(claim_id).as_dict() for claim_id in unique}

    def _checks(self, claim_id: str) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT * FROM claim_checks WHERE claim_id = ? ORDER BY created_at", (claim_id,)
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["passed"] = bool(item["passed"])
            item["detail"] = json.loads(item.pop("detail_json"))
            out.append(item)
        return out

    def _evidence(self, claim_id: str) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            """SELECT e.*, a.stance, a.confidence, a.created_at AS attested_at
               FROM attestations a JOIN evidence e ON e.id = a.evidence_id
               WHERE a.claim_id = ? ORDER BY a.created_at""",
            (claim_id,),
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["active"] = bool(item["active"])
            item["metadata"] = json.loads(item.pop("metadata_json"))
            out.append(item)
        return out

    def _contradictions(self, claim_id: str) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            """SELECT *, CASE WHEN claim_a = ? THEN claim_b ELSE claim_a END AS other_claim_id
               FROM contradictions WHERE active = 1 AND (claim_a = ? OR claim_b = ?)
               ORDER BY created_at""",
            (claim_id, claim_id, claim_id),
        ).fetchall()
        return [dict(row) for row in rows]

    def _dependencies(self, claim_id: str) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            """SELECT p.id AS proof_id, p.rule, pp.premise_claim_id, pp.position, p.active
               FROM proofs p JOIN proof_premises pp ON pp.proof_id = p.id
               WHERE p.conclusion_claim_id = ? ORDER BY p.created_at, pp.position""",
            (claim_id,),
        ).fetchall()
        return [dict(row) for row in rows]
