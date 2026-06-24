from __future__ import annotations

from dataclasses import dataclass, field

from .models import ActorRole, EvidenceKind


@dataclass(slots=True)
class CommitPolicy:
    """Rules for moving a proposal into committed knowledge.

    The policy deliberately separates *support* from *commitment*. A claim can be
    supported by evidence yet remain provisional until an authorized actor commits it.
    """

    min_independent_sources: int = 2
    strong_single_source_kinds: set[EvidenceKind] = field(
        default_factory=lambda: {
            EvidenceKind.FORMAL_PROOF,
            EvidenceKind.EXPERIMENT_RECEIPT,
            EvidenceKind.CODE_TEST,
            EvidenceKind.DATASET_ANALYSIS_RECEIPT,
            EvidenceKind.CODE_EXECUTION_RECEIPT,
        }
    )
    human_review_claim_types: set[str] = field(
        default_factory=lambda: {"meta", "code", "safety_policy"}
    )
    non_warrant_source_kinds: set[EvidenceKind] = field(
        default_factory=lambda: {EvidenceKind.MODEL_PROPOSAL}
    )

    def actor_can_commit(self, actor_role: ActorRole, claim_type: str) -> bool:
        if actor_role == ActorRole.LLM:
            return False
        if claim_type in self.human_review_claim_types:
            return actor_role == ActorRole.HUMAN
        return actor_role in {ActorRole.HUMAN, ActorRole.POLICY}
