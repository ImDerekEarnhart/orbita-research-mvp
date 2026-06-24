from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ClaimStatus(StrEnum):
    PROVISIONAL = "provisional"
    COMMITTED = "committed"
    CHALLENGED = "challenged"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class SupportState(StrEnum):
    SUPPORTED = "supported"
    CHALLENGED = "challenged"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class Stance(StrEnum):
    SUPPORT = "support"
    REFUTE = "refute"


class EvidenceKind(StrEnum):
    WEB_SOURCE = "web_source"
    DATASET = "dataset"
    EXPERIMENT_RECEIPT = "experiment_receipt"
    FORMAL_PROOF = "formal_proof"
    HUMAN_TESTIMONY = "human_testimony"
    MODEL_PROPOSAL = "model_proposal"
    CODE_TEST = "code_test"
    DATASET_ANALYSIS_RECEIPT = "dataset_analysis_receipt"
    CODE_EXECUTION_RECEIPT = "code_execution_receipt"


class ActorRole(StrEnum):
    HUMAN = "human"
    POLICY = "policy"
    LLM = "llm"
    TOOL = "tool"




class AnalysisStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    REPRODUCED = "reproduced"
    DIVERGED = "diverged"
    INPUT_MISMATCH = "input_mismatch"
    CODE_MISMATCH = "code_mismatch"


class AnalysisOutcome(StrEnum):
    SUPPORT = "support"
    REFUTE = "refute"
    INCONCLUSIVE = "inconclusive"




class ProposalBatchStatus(StrEnum):
    PROCESSING = "processing"
    APPLIED = "applied"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class ProposalItemStatus(StrEnum):
    PENDING = "pending"
    APPLIED = "applied"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


class ProposalItemType(StrEnum):
    ENTITY = "entity"
    PREDICATE = "predicate"
    CLAIM = "claim"
    PROOF = "proof"
    EXTRACTION = "extraction"


class ReviewDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StepStatus(StrEnum):
    PENDING = "pending"
    WAITING_APPROVAL = "waiting_approval"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ObjectKind(StrEnum):
    ENTITY = "entity"
    LITERAL = "literal"


class PredicateRangeKind(StrEnum):
    ENTITY = "entity"
    LITERAL = "literal"
    EITHER = "either"


class LiteralDatatype(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    URI = "uri"
    JSON = "json"


@dataclass(frozen=True, slots=True)
class TypedLiteral:
    """A canonical literal value used as the object of a structured claim.

    ``unit`` is optional and intentionally separate from the numeric value so
    that later receipt and conversion layers can reason about it explicitly.
    """

    value: Any
    datatype: LiteralDatatype | str | None = None
    unit: str | None = None

    def resolved_datatype(self) -> LiteralDatatype:
        if self.datatype is not None:
            return LiteralDatatype(self.datatype)
        if isinstance(self.value, bool):
            return LiteralDatatype.BOOLEAN
        if isinstance(self.value, int):
            return LiteralDatatype.INTEGER
        if isinstance(self.value, float):
            return LiteralDatatype.FLOAT
        if isinstance(self.value, (dict, list, tuple)):
            return LiteralDatatype.JSON
        return LiteralDatatype.STRING


@dataclass(slots=True)
class SupportReport:
    claim_id: str
    state: SupportState
    direct_support_sources: list[str] = field(default_factory=list)
    direct_refute_sources: list[str] = field(default_factory=list)
    satisfied_proofs: list[str] = field(default_factory=list)
    broken_proofs: dict[str, list[str]] = field(default_factory=dict)
    contradictions: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "state": self.state.value,
            "direct_support_sources": self.direct_support_sources,
            "direct_refute_sources": self.direct_refute_sources,
            "satisfied_proofs": self.satisfied_proofs,
            "broken_proofs": self.broken_proofs,
            "contradictions": self.contradictions,
            "reasons": self.reasons,
        }


@dataclass(slots=True)
class ActionResult:
    ok: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class ObligationResult:
    obligation: dict[str, Any]
    ok: bool
    detail: str
