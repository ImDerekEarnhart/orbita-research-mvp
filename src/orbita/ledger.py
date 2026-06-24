from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .analysis import DatasetAnalysisRuntime
from .db import Database
from .graph import EpistemicGraphRuntime
from .execution import ContainerExecutionRuntime
from .discovery import GovernedDiscoveryRuntime
from .evaluation import ComparativeEvaluationRuntime
from .research import EmpiricalResearchRuntime
from .language import WarrantedLanguageRuntime
from .agent_os import ComputerAgentRuntime
from .coding import CodingRuntime
from .integrations import IntegrationRuntime, ScheduledTaskRuntime
from .adaptive import AdaptiveSkillRuntime
from .models import ActorRole, ClaimStatus, EvidenceKind, Stance
from .policy import CommitPolicy
from .relations import RelationStore
from .proposals import ProposalAdapter


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class EpistemicLedger:
    def __init__(self, db_path: str | Path, policy: CommitPolicy | None = None):
        self.db = Database(db_path)
        self.policy = policy or CommitPolicy()
        self.relations = RelationStore(self)
        self.analyses = DatasetAnalysisRuntime(self)
        self.proposals = ProposalAdapter(self)
        self.graphs = EpistemicGraphRuntime(self)
        self.executions = ContainerExecutionRuntime(self)
        self.discovery = GovernedDiscoveryRuntime(self)
        self.evaluations = ComparativeEvaluationRuntime(self)
        self.research = EmpiricalResearchRuntime(self)
        self.language = WarrantedLanguageRuntime(self)
        self.agent = ComputerAgentRuntime(self)
        self.coding = CodingRuntime(self)
        self.integrations = IntegrationRuntime(self)
        self.adaptive = AdaptiveSkillRuntime(self)
        self.scheduler = ScheduledTaskRuntime(self)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "EpistemicLedger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _event(
        self,
        entity_type: str,
        entity_id: str,
        event_type: str,
        payload: dict[str, Any],
        actor: str,
        actor_role: ActorRole,
    ) -> None:
        self.db.conn.execute(
            """INSERT INTO events
               (entity_type, entity_id, event_type, payload_json, actor, actor_role, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entity_type,
                entity_id,
                event_type,
                stable_json(payload),
                actor,
                actor_role.value,
                utcnow(),
            ),
        )

    def add_claim(
        self,
        canonical_text: str,
        *,
        claim_type: str = "fact",
        scope: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        actor: str = "user",
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> str:
        if not canonical_text.strip():
            raise ValueError("canonical_text cannot be empty")
        claim_id = new_id("clm")
        now = utcnow()
        self.db.conn.execute(
            """INSERT INTO claims
               (id, canonical_text, claim_type, status, scope_json, metadata_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                claim_id,
                canonical_text.strip(),
                claim_type,
                ClaimStatus.PROVISIONAL.value,
                stable_json(scope or {}),
                stable_json(metadata or {}),
                now,
                now,
            ),
        )
        self._event(
            "claim",
            claim_id,
            "CLAIM_PROPOSED",
            {"canonical_text": canonical_text, "claim_type": claim_type, "scope": scope or {}},
            actor,
            actor_role,
        )
        self.db.conn.commit()
        return claim_id

    def add_evidence(
        self,
        source_uri: str,
        excerpt: str,
        *,
        source_kind: EvidenceKind,
        independence_key: str,
        content: str | bytes | None = None,
        metadata: dict[str, Any] | None = None,
        actor: str = "tool",
        actor_role: ActorRole = ActorRole.TOOL,
    ) -> str:
        if not independence_key.strip():
            raise ValueError("independence_key is required")
        evidence_id = new_id("evd")
        raw = content if content is not None else excerpt
        raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else raw
        digest = hashlib.sha256(raw_bytes).hexdigest()
        self.db.conn.execute(
            """INSERT INTO evidence
               (id, source_uri, source_kind, content_hash, excerpt, independence_key,
                metadata_json, active, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)""",
            (
                evidence_id,
                source_uri,
                source_kind.value,
                digest,
                excerpt,
                independence_key,
                stable_json(metadata or {}),
                utcnow(),
            ),
        )
        self._event(
            "evidence",
            evidence_id,
            "EVIDENCE_ADDED",
            {
                "source_uri": source_uri,
                "source_kind": source_kind.value,
                "independence_key": independence_key,
                "content_hash": digest,
            },
            actor,
            actor_role,
        )
        self.db.conn.commit()
        return evidence_id

    def attest(
        self,
        claim_id: str,
        evidence_id: str,
        stance: Stance,
        *,
        confidence: float = 1.0,
        actor: str = "tool",
        actor_role: ActorRole = ActorRole.TOOL,
    ) -> str:
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be between 0 and 1")
        self._require_claim(claim_id)
        self._require_evidence(evidence_id)
        att_id = new_id("att")
        self.db.conn.execute(
            """INSERT INTO attestations
               (id, claim_id, evidence_id, stance, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (att_id, claim_id, evidence_id, stance.value, confidence, utcnow()),
        )
        self._event(
            "claim",
            claim_id,
            "ATTESTATION_ADDED",
            {"attestation_id": att_id, "evidence_id": evidence_id, "stance": stance.value},
            actor,
            actor_role,
        )
        self.db.conn.commit()
        return att_id

    def add_proof(
        self,
        conclusion_claim_id: str,
        premise_claim_ids: Iterable[str],
        *,
        rule: str,
        metadata: dict[str, Any] | None = None,
        actor: str = "reasoner",
        actor_role: ActorRole = ActorRole.TOOL,
    ) -> str:
        premises = list(dict.fromkeys(premise_claim_ids))
        if not premises:
            raise ValueError("A proof must have at least one premise")
        self._require_claim(conclusion_claim_id)
        for premise in premises:
            self._require_claim(premise)
        if conclusion_claim_id in premises:
            raise ValueError("A claim cannot directly prove itself")
        proof_id = new_id("prf")
        self.db.conn.execute(
            """INSERT INTO proofs
               (id, conclusion_claim_id, rule, metadata_json, active, created_at)
               VALUES (?, ?, ?, ?, 1, ?)""",
            (proof_id, conclusion_claim_id, rule, stable_json(metadata or {}), utcnow()),
        )
        self.db.conn.executemany(
            "INSERT INTO proof_premises (proof_id, premise_claim_id, position) VALUES (?, ?, ?)",
            [(proof_id, premise, idx) for idx, premise in enumerate(premises)],
        )
        self._event(
            "claim",
            conclusion_claim_id,
            "PROOF_ADDED",
            {"proof_id": proof_id, "premises": premises, "rule": rule},
            actor,
            actor_role,
        )
        self.db.conn.commit()
        return proof_id

    def add_contradiction(
        self,
        claim_a: str,
        claim_b: str,
        *,
        rationale: str,
        actor: str = "reviewer",
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> str:
        if claim_a == claim_b:
            raise ValueError("A claim cannot contradict itself")
        self._require_claim(claim_a)
        self._require_claim(claim_b)
        contradiction_id = new_id("ctr")
        self.db.conn.execute(
            """INSERT INTO contradictions
               (id, claim_a, claim_b, rationale, active, created_at)
               VALUES (?, ?, ?, ?, 1, ?)""",
            (contradiction_id, claim_a, claim_b, rationale, utcnow()),
        )
        for claim_id, other in ((claim_a, claim_b), (claim_b, claim_a)):
            self._event(
                "claim",
                claim_id,
                "CONTRADICTION_LINKED",
                {"contradiction_id": contradiction_id, "other_claim_id": other, "rationale": rationale},
                actor,
                actor_role,
            )
        self.db.conn.commit()
        return contradiction_id

    def revoke_evidence(
        self,
        evidence_id: str,
        *,
        rationale: str,
        actor: str = "reviewer",
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> list[str]:
        self._require_evidence(evidence_id)
        self.db.conn.execute("UPDATE evidence SET active = 0 WHERE id = ?", (evidence_id,))
        self._event(
            "evidence",
            evidence_id,
            "EVIDENCE_REVOKED",
            {"rationale": rationale},
            actor,
            actor_role,
        )
        affected = self.descendants_of_evidence(evidence_id)
        self.db.conn.commit()
        return affected

    def commit_claim(
        self,
        claim_id: str,
        *,
        actor: str,
        actor_role: ActorRole,
        rationale: str,
        override: bool = False,
    ) -> None:
        claim = self.get_claim(claim_id)
        if not self.policy.actor_can_commit(actor_role, claim["claim_type"]):
            raise PermissionError(
                f"Actor role {actor_role.value} cannot commit claim type {claim['claim_type']}"
            )
        if not override:
            from .support import SupportEngine

            report = SupportEngine(self).evaluate(claim_id)
            if report.state.value not in {"supported", "challenged"}:
                raise ValueError(f"Claim is not sufficiently supported: {report.reasons}")
            if claim["metadata"].get("requires_independent_replication"):
                minimum = int(claim["metadata"].get("replication_min_independent_sources", 2))
                rows = self.db.conn.execute(
                    """SELECT DISTINCT e.independence_key, e.source_kind
                       FROM attestations a
                       JOIN evidence e ON e.id = a.evidence_id
                       WHERE a.claim_id = ? AND a.stance = ? AND e.active = 1""",
                    (claim_id, Stance.SUPPORT.value),
                ).fetchall()
                warranting = {
                    row["independence_key"]
                    for row in rows
                    if EvidenceKind(row["source_kind"]) not in self.policy.non_warrant_source_kinds
                }
                if len(warranting) < minimum:
                    raise ValueError(
                        "Claim requires independent replication before commitment: "
                        f"{len(warranting)}/{minimum} independent warranting sources"
                    )
        self.db.conn.execute(
            "UPDATE claims SET status = ?, updated_at = ? WHERE id = ?",
            (ClaimStatus.COMMITTED.value, utcnow(), claim_id),
        )
        self._event(
            "claim",
            claim_id,
            "CLAIM_COMMITTED",
            {"rationale": rationale, "override": override},
            actor,
            actor_role,
        )
        self.db.conn.commit()

    def set_claim_status(
        self,
        claim_id: str,
        status: ClaimStatus,
        *,
        rationale: str,
        actor: str,
        actor_role: ActorRole,
    ) -> None:
        self._require_claim(claim_id)
        self.db.conn.execute(
            "UPDATE claims SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, utcnow(), claim_id),
        )
        self._event(
            "claim",
            claim_id,
            f"CLAIM_{status.value.upper()}",
            {"rationale": rationale},
            actor,
            actor_role,
        )
        self.db.conn.commit()

    def get_claim(self, claim_id: str) -> dict[str, Any]:
        row = self.db.conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown claim: {claim_id}")
        result = dict(row)
        result["scope"] = json.loads(result.pop("scope_json"))
        result["metadata"] = json.loads(result.pop("metadata_json"))
        structured = self.db.conn.execute(
            "SELECT 1 FROM relation_claims WHERE claim_id = ?", (claim_id,)
        ).fetchone()
        if structured is not None:
            result["relation"] = self.relations.get_relation_claim(claim_id)
        return result

    def list_claims(self) -> list[dict[str, Any]]:
        rows = self.db.conn.execute("SELECT * FROM claims ORDER BY created_at").fetchall()
        return [self.get_claim(row["id"]) for row in rows]

    # Typed knowledge-graph convenience API. The implementation lives in
    # RelationStore so the core event/evidence ledger remains independently usable.
    def add_entity(self, *args, **kwargs) -> str:
        return self.relations.add_entity(*args, **kwargs)

    def add_predicate(self, *args, **kwargs) -> str:
        return self.relations.add_predicate(*args, **kwargs)

    def add_relation_claim(self, *args, **kwargs) -> str:
        return self.relations.add_relation_claim(*args, **kwargs)

    def get_relation_claim(self, claim_id: str) -> dict[str, Any]:
        return self.relations.get_relation_claim(claim_id)

    def find_relation_claims(self, **kwargs) -> list[dict[str, Any]]:
        return self.relations.find_relation_claims(**kwargs)

    def negate_relation_claim(self, claim_id: str, **kwargs) -> str:
        return self.relations.negate_relation_claim(claim_id, **kwargs)

    # Hash-bound dataset analysis receipts using the built-in safe vocabulary.
    def run_analysis(self, *args, **kwargs) -> dict[str, Any]:
        return self.analyses.run(*args, **kwargs)

    def reproduce_analysis(self, receipt_id: str, **kwargs) -> dict[str, Any]:
        return self.analyses.reproduce(receipt_id, **kwargs)

    def get_analysis_receipt(self, receipt_id: str) -> dict[str, Any]:
        return self.analyses.get(receipt_id)

    def list_analysis_receipts(self) -> list[dict[str, Any]]:
        return self.analyses.list()

    # Manifest-bound container execution. Every run requires human approval tied
    # to the exact image digest, code, inputs, command, limits, and obligations.
    def submit_execution(self, *args, **kwargs) -> dict[str, Any]:
        return self.executions.submit(*args, **kwargs)

    def approve_execution(self, run_id: str, **kwargs) -> dict[str, Any]:
        return self.executions.approve(run_id, **kwargs)

    def execute_container(self, run_id: str, **kwargs) -> dict[str, Any]:
        return self.executions.execute(run_id, **kwargs)

    def get_execution(self, run_id: str) -> dict[str, Any]:
        return self.executions.get(run_id)

    def list_executions(self) -> list[dict[str, Any]]:
        return self.executions.list()

    # Restart-safe governed discovery investigations. Candidate mining is
    # non-warranting; exact tests are staged through manifest-bound execution.
    def create_discovery(self, *args, **kwargs) -> dict[str, Any]:
        return self.discovery.create(*args, **kwargs)

    def advance_discovery(self, investigation_id: str, **kwargs) -> dict[str, Any]:
        return self.discovery.advance(investigation_id, **kwargs)

    def get_discovery(self, investigation_id: str) -> dict[str, Any]:
        return self.discovery.get(investigation_id)

    def list_discoveries(self) -> list[dict[str, Any]]:
        return self.discovery.list()

    # Strict LLM proposal boundary. Model output may create provisional claims,
    # but MODEL_PROPOSAL attestations never count as warrant.
    def ingest_model_response(self, *args, **kwargs) -> dict[str, Any]:
        return self.proposals.ingest_response(*args, **kwargs)

    def get_proposal_batch(self, batch_id: str) -> dict[str, Any]:
        return self.proposals.get_batch(batch_id)

    def list_proposal_batches(self, **kwargs) -> list[dict[str, Any]]:
        return self.proposals.list_batches(**kwargs)

    def review_proposal_item(self, *args, **kwargs) -> dict[str, Any]:
        return self.proposals.review_item(*args, **kwargs)

    # Deterministic epistemic graph snapshots, collapse diffs, and browser artifacts.
    def capture_graph(self, *args, **kwargs) -> dict[str, Any]:
        return self.graphs.capture(*args, **kwargs)

    def compare_graphs(self, before, after, **kwargs) -> dict[str, Any]:
        return self.graphs.compare(before, after, **kwargs)

    def get_graph_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        return self.graphs.get_snapshot(snapshot_id)

    def get_graph_diff(self, diff_id: str) -> dict[str, Any]:
        return self.graphs.get_diff(diff_id)

    def history(self, entity_type: str, entity_id: str) -> list[dict[str, Any]]:
        rows = self.db.conn.execute(
            """SELECT * FROM events
               WHERE entity_type = ? AND entity_id = ? ORDER BY id""",
            (entity_type, entity_id),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def descendants_of_claim(self, claim_id: str) -> list[str]:
        seen: set[str] = set()
        queue = [claim_id]
        while queue:
            current = queue.pop(0)
            rows = self.db.conn.execute(
                """SELECT DISTINCT p.conclusion_claim_id
                   FROM proof_premises pp
                   JOIN proofs p ON p.id = pp.proof_id
                   WHERE pp.premise_claim_id = ? AND p.active = 1""",
                (current,),
            ).fetchall()
            for row in rows:
                child = row["conclusion_claim_id"]
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
        return sorted(seen)

    def descendants_of_evidence(self, evidence_id: str) -> list[str]:
        roots = [
            row["claim_id"]
            for row in self.db.conn.execute(
                "SELECT DISTINCT claim_id FROM attestations WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchall()
        ]
        affected = set(roots)
        for root in roots:
            affected.update(self.descendants_of_claim(root))
        return sorted(affected)

    def _require_claim(self, claim_id: str) -> None:
        if self.db.conn.execute("SELECT 1 FROM claims WHERE id = ?", (claim_id,)).fetchone() is None:
            raise KeyError(f"Unknown claim: {claim_id}")

    def _require_evidence(self, evidence_id: str) -> None:
        if self.db.conn.execute("SELECT 1 FROM evidence WHERE id = ?", (evidence_id,)).fetchone() is None:
            raise KeyError(f"Unknown evidence: {evidence_id}")
