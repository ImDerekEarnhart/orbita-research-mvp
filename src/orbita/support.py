from __future__ import annotations

from collections import defaultdict

from .ledger import EpistemicLedger
from .models import ClaimStatus, EvidenceKind, Stance, SupportReport, SupportState


class SupportEngine:
    """Evaluate current warrant using AND-within-proof / OR-between-proofs semantics."""

    def __init__(self, ledger: EpistemicLedger):
        self.ledger = ledger

    def evaluate(self, claim_id: str) -> SupportReport:
        memo: dict[str, SupportReport] = {}
        return self._evaluate(claim_id, memo, stack=[])

    def evaluate_many(self, claim_ids: list[str]) -> dict[str, SupportReport]:
        memo: dict[str, SupportReport] = {}
        return {claim_id: self._evaluate(claim_id, memo, stack=[]) for claim_id in claim_ids}

    def collapse_report(self, affected_claim_ids: list[str]) -> dict[str, dict]:
        reports = self.evaluate_many(affected_claim_ids)
        return {claim_id: report.as_dict() for claim_id, report in reports.items()}

    def _evaluate(
        self,
        claim_id: str,
        memo: dict[str, SupportReport],
        stack: list[str],
    ) -> SupportReport:
        if claim_id in memo:
            return memo[claim_id]
        if claim_id in stack:
            report = SupportReport(
                claim_id=claim_id,
                state=SupportState.UNSUPPORTED,
                reasons=["Cyclic derivation detected"],
            )
            memo[claim_id] = report
            return report

        claim = self.ledger.get_claim(claim_id)
        if claim["status"] in {ClaimStatus.REJECTED.value, ClaimStatus.SUPERSEDED.value}:
            report = SupportReport(
                claim_id=claim_id,
                state=SupportState.UNSUPPORTED,
                reasons=[f"Claim status is {claim['status']}"],
            )
            memo[claim_id] = report
            return report

        conn = self.ledger.db.conn
        rows = conn.execute(
            """SELECT a.stance, e.id AS evidence_id, e.independence_key,
                      e.source_kind, e.active
               FROM attestations a
               JOIN evidence e ON e.id = a.evidence_id
               WHERE a.claim_id = ?""",
            (claim_id,),
        ).fetchall()

        support_by_key: dict[str, list] = defaultdict(list)
        refute_by_key: dict[str, list] = defaultdict(list)
        strong_support = False
        for row in rows:
            if not row["active"]:
                continue
            source_kind = EvidenceKind(row["source_kind"])
            if source_kind in self.ledger.policy.non_warrant_source_kinds:
                continue
            target = support_by_key if row["stance"] == Stance.SUPPORT.value else refute_by_key
            target[row["independence_key"]].append(row)
            if (
                row["stance"] == Stance.SUPPORT.value
                and source_kind in self.ledger.policy.strong_single_source_kinds
            ):
                strong_support = True

        direct_supported = (
            len(support_by_key) >= self.ledger.policy.min_independent_sources or strong_support
        )

        proof_rows = conn.execute(
            "SELECT id, rule FROM proofs WHERE conclusion_claim_id = ? AND active = 1",
            (claim_id,),
        ).fetchall()
        satisfied_proofs: list[str] = []
        broken_proofs: dict[str, list[str]] = {}
        next_stack = [*stack, claim_id]
        for proof in proof_rows:
            premise_rows = conn.execute(
                """SELECT premise_claim_id FROM proof_premises
                   WHERE proof_id = ? ORDER BY position""",
                (proof["id"],),
            ).fetchall()
            missing: list[str] = []
            for premise_row in premise_rows:
                premise_id = premise_row["premise_claim_id"]
                premise_report = self._evaluate(premise_id, memo, next_stack)
                if premise_report.state not in {SupportState.SUPPORTED, SupportState.CHALLENGED}:
                    missing.append(premise_id)
            if missing:
                broken_proofs[proof["id"]] = missing
            else:
                satisfied_proofs.append(proof["id"])

        contradiction_rows = conn.execute(
            """SELECT CASE WHEN claim_a = ? THEN claim_b ELSE claim_a END AS other_claim
               FROM contradictions
               WHERE active = 1 AND (claim_a = ? OR claim_b = ?)""",
            (claim_id, claim_id, claim_id),
        ).fetchall()
        active_contradictions: list[str] = []
        for row in contradiction_rows:
            other = row["other_claim"]
            if other in stack:
                continue
            other_report = self._evaluate(other, memo, next_stack)
            if other_report.state in {SupportState.SUPPORTED, SupportState.CHALLENGED}:
                active_contradictions.append(other)

        has_support = direct_supported or bool(satisfied_proofs)
        has_refutation = bool(refute_by_key) or bool(active_contradictions)
        if has_support and has_refutation:
            state = SupportState.CHALLENGED
        elif has_support:
            state = SupportState.SUPPORTED
        elif has_refutation:
            state = SupportState.UNSUPPORTED
        else:
            state = SupportState.UNKNOWN

        reasons: list[str] = []
        if direct_supported:
            reasons.append("Direct evidence meets the current support policy")
        if satisfied_proofs:
            reasons.append("At least one complete proof set is satisfied")
        if broken_proofs:
            reasons.append("One or more proof sets have missing premises")
        if refute_by_key:
            reasons.append("Active refuting evidence exists")
        if active_contradictions:
            reasons.append("An actively supported contradictory claim exists")
        if not reasons:
            reasons.append("No active evidence or complete proof supports this claim")

        report = SupportReport(
            claim_id=claim_id,
            state=state,
            direct_support_sources=sorted(support_by_key),
            direct_refute_sources=sorted(refute_by_key),
            satisfied_proofs=satisfied_proofs,
            broken_proofs=broken_proofs,
            contradictions=active_contradictions,
            reasons=reasons,
        )
        memo[claim_id] = report
        return report
