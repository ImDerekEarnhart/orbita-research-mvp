from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from .ledger import new_id, stable_json, utcnow
from .models import ActorRole, SupportState

if TYPE_CHECKING:  # pragma: no cover
    from .language import GroundingReport, PlannedSentence, SemanticFrame, WarrantedLanguageRuntime


DISCOURSE_API_VERSION = "1.2"


@dataclass(slots=True)
class DiscourseMove:
    id: str
    semantic_act: str
    purpose: str
    sentence: "PlannedSentence"
    features: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    allowed: bool = True
    gate_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "semantic_act": self.semantic_act,
            "purpose": self.purpose,
            "sentence": self.sentence.as_dict(),
            "features": dict(sorted(self.features.items())),
            "score": self.score,
            "allowed": self.allowed,
            "gate_reasons": list(self.gate_reasons),
        }


@dataclass(slots=True)
class DiscoursePlan:
    id: str
    utterance: str
    frame: dict[str, Any]
    context: dict[str, Any]
    candidates: list[DiscourseMove]
    selected: list[DiscourseMove]
    policy_version: str
    policy_trace: dict[str, Any]
    plan_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "utterance": self.utterance,
            "frame": self.frame,
            "context": self.context,
            "candidates": [move.as_dict() for move in self.candidates],
            "selected": [move.as_dict() for move in self.selected],
            "policy_version": self.policy_version,
            "policy_trace": self.policy_trace,
            "plan_hash": self.plan_hash,
        }


DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "state_supported_claim": {
        "bias": 0.2,
        "direct_answer": 3.5,
        "supported": 3.0,
        "has_evidence": 1.2,
        "has_proof": 0.6,
        "description_intent": 0.8,
        "relation_intent": 1.2,
    },
    "state_challenged_claim": {
        "bias": 0.1,
        "direct_answer": 3.0,
        "challenged": 3.5,
        "has_evidence": 1.0,
        "caution": 1.5,
        "relation_intent": 1.0,
    },
    "state_refuted_claim": {
        "bias": 0.1,
        "direct_answer": 3.2,
        "supported": 2.4,
        "has_evidence": 1.0,
        "relation_intent": 1.2,
    },
    "explain_warrant": {
        "bias": -0.2,
        "explanation_intent": 3.5,
        "has_evidence": 1.2,
        "has_proof": 2.0,
        "after_direct": 1.0,
    },
    "explain_contradiction": {
        "bias": -0.2,
        "contradiction_intent": 3.5,
        "caution": 1.2,
        "after_direct": 1.0,
    },
    "abstain": {
        "bias": -0.3,
        "no_warranted_fact": 4.0,
        "unresolved": 2.0,
        "caution": 1.0,
    },
    "ask_clarification": {
        "bias": -0.3,
        "unresolved": 4.0,
        "unknown_intent": 3.0,
    },
}


class SemanticActionRanker:
    """A tiny transparent learned model over semantic acts, never tokens.

    The ranker is deliberately subordinate to hard warrant gates. Training may
    change ordering among permissible semantic moves, but it can never make an
    unsupported factual move permissible.
    """

    def __init__(
        self,
        weights: dict[str, dict[str, float]] | None = None,
        *,
        name: str = "orbita-linear-semantic-policy",
        version: str = DISCOURSE_API_VERSION,
    ):
        self.name = name
        self.version = version
        self.weights = json.loads(json.dumps(weights or DEFAULT_WEIGHTS))

    def score(self, label: str, features: dict[str, float]) -> float:
        weights = self.weights.get(label, {})
        total = float(weights.get("bias", 0.0))
        for key, value in features.items():
            total += float(weights.get(key, 0.0)) * float(value)
        return total

    def update(self, preferred_label: str, rejected_label: str, features: dict[str, float], *, rate: float = 0.1) -> None:
        if preferred_label == rejected_label:
            return
        self.weights.setdefault(preferred_label, {})
        self.weights.setdefault(rejected_label, {})
        for key, value in {"bias": 1.0, **features}.items():
            self.weights[preferred_label][key] = self.weights[preferred_label].get(key, 0.0) + rate * value
            self.weights[rejected_label][key] = self.weights[rejected_label].get(key, 0.0) - rate * value

    def train_pairwise(self, examples: Iterable[dict[str, Any]], *, epochs: int = 5, rate: float = 0.1) -> dict[str, Any]:
        records = list(examples)
        mistakes = 0
        for _ in range(max(1, epochs)):
            for example in records:
                features = {str(k): float(v) for k, v in dict(example["features"]).items()}
                preferred = str(example["preferred"])
                alternatives = [str(x) for x in example.get("alternatives", []) if str(x) != preferred]
                if not alternatives:
                    continue
                best_alt = max(alternatives, key=lambda label: self.score(label, features))
                if self.score(preferred, features) <= self.score(best_alt, features):
                    self.update(preferred, best_alt, features, rate=rate)
                    mistakes += 1
        return {"examples": len(records), "epochs": epochs, "updates": mistakes}

    def model_payload(self, training_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        labels = sorted(self.weights)
        features = sorted({feature for weights in self.weights.values() for feature in weights})
        return {
            "name": self.name,
            "version": self.version,
            "labels": labels,
            "feature_schema": features,
            "weights": self.weights,
            "training_metadata": training_metadata or {},
        }

    def model_hash(self, training_metadata: dict[str, Any] | None = None) -> str:
        return hashlib.sha256(stable_json(self.model_payload(training_metadata)).encode("utf-8")).hexdigest()


class GuardedSemanticActionPolicy:
    """Hard warrant gates plus a compact semantic-action ranker."""

    def __init__(self, ranker: SemanticActionRanker | None = None):
        self.ranker = ranker or SemanticActionRanker()

    @property
    def version(self) -> str:
        return f"guarded-linear/{self.ranker.version}"

    def features(
        self,
        frame: "SemanticFrame",
        sentence: "PlannedSentence",
        *,
        position: int,
        has_any_warranted_fact: bool,
    ) -> dict[str, float]:
        intent = frame.intent.value
        act = sentence.semantic_act.value
        factual = act in {"state_supported_claim", "state_challenged_claim", "state_refuted_claim"}
        direct = factual and position == 0
        return {
            "direct_answer": float(direct),
            "after_direct": float(position > 0),
            "supported": float(sentence.support_state == SupportState.SUPPORTED.value),
            "challenged": float(sentence.support_state == SupportState.CHALLENGED.value),
            "has_evidence": float(bool(sentence.evidence_ids)),
            "has_proof": float(bool(sentence.proof_ids)),
            "caution": float(act in {"state_challenged_claim", "explain_contradiction", "abstain"}),
            "description_intent": float(intent in {"define", "describe", "list_objects"}),
            "relation_intent": float(intent in {"verify_relation", "explain_relation"}),
            "explanation_intent": float(intent == "explain_relation"),
            "contradiction_intent": float(intent == "list_contradictions"),
            "unknown_intent": float(intent == "unknown"),
            "unresolved": float(act == "ask_clarification"),
            "no_warranted_fact": float(not has_any_warranted_fact),
        }

    def gate(self, sentence: "PlannedSentence") -> tuple[bool, list[str]]:
        act = sentence.semantic_act.value
        reasons: list[str] = []
        if act == "state_supported_claim":
            if sentence.support_state != SupportState.SUPPORTED.value:
                reasons.append("supported factual wording requires supported state")
            if not sentence.claim_ids:
                reasons.append("factual wording requires at least one claim")
            if not sentence.evidence_ids and not sentence.proof_ids:
                reasons.append("supported factual wording requires evidence or proof")
        elif act == "state_challenged_claim":
            if sentence.support_state != SupportState.CHALLENGED.value:
                reasons.append("challenged wording requires challenged state")
            if not sentence.claim_ids:
                reasons.append("challenged wording requires at least one claim")
        elif act == "state_refuted_claim":
            if sentence.support_state not in {SupportState.SUPPORTED.value, SupportState.CHALLENGED.value}:
                reasons.append("negative factual wording requires a warranted negative claim")
            if not sentence.claim_ids:
                reasons.append("negative factual wording requires at least one claim")
        elif act == "explain_warrant":
            if not sentence.evidence_ids and not sentence.proof_ids:
                reasons.append("warrant explanation requires evidence or proof lineage")
        return not reasons, reasons

    def select(self, frame: "SemanticFrame", sentences: list["PlannedSentence"], *, max_sentences: int = 7) -> tuple[list[DiscourseMove], list[DiscourseMove], dict[str, Any]]:
        factual_acts = {"state_supported_claim", "state_challenged_claim", "state_refuted_claim"}
        has_any_warranted_fact = any(
            s.semantic_act.value in factual_acts and self.gate(s)[0] for s in sentences
        )
        candidates: list[DiscourseMove] = []
        for position, sentence in enumerate(sentences):
            allowed, reasons = self.gate(sentence)
            features = self.features(
                frame, sentence, position=position, has_any_warranted_fact=has_any_warranted_fact
            )
            score = self.ranker.score(sentence.semantic_act.value, features)
            purpose = {
                "state_supported_claim": "direct warranted answer",
                "state_challenged_claim": "qualified answer",
                "state_refuted_claim": "warranted negative answer",
                "explain_warrant": "support explanation",
                "explain_contradiction": "contradiction explanation",
                "abstain": "epistemic abstention",
                "ask_clarification": "semantic clarification",
            }.get(sentence.semantic_act.value, "semantic move")
            move_digest = hashlib.sha256(
                stable_json({"position": position, "sentence": sentence.as_dict()}).encode("utf-8")
            ).hexdigest()[:16]
            candidates.append(
                DiscourseMove(
                    id=f"dsm_{move_digest}",
                    semantic_act=sentence.semantic_act.value,
                    purpose=purpose,
                    sentence=sentence,
                    features=features,
                    score=score,
                    allowed=allowed,
                    gate_reasons=reasons,
                )
            )

        allowed_moves = [m for m in candidates if m.allowed]
        factual = [m for m in allowed_moves if m.semantic_act in factual_acts]
        explanations = [m for m in allowed_moves if m.semantic_act in {"explain_warrant", "explain_contradiction"}]
        nonfactual = [m for m in allowed_moves if m.semantic_act in {"abstain", "ask_clarification"}]

        selected: list[DiscourseMove] = []
        if factual:
            factual.sort(key=lambda m: (-m.score, m.id))
            selected.extend(factual[:5])
            explanations.sort(key=lambda m: (-m.score, m.id))
            selected.extend(explanations[: max(0, max_sentences - len(selected))])
        elif nonfactual:
            nonfactual.sort(key=lambda m: (-m.score, m.id))
            selected.append(nonfactual[0])
        elif explanations:
            explanations.sort(key=lambda m: (-m.score, m.id))
            selected.extend(explanations[:max_sentences])

        # Preserve stable direct-answer-before-explanation discourse order.
        order = {
            "state_supported_claim": 0,
            "state_challenged_claim": 0,
            "state_refuted_claim": 0,
            "explain_warrant": 1,
            "explain_contradiction": 2,
            "abstain": 3,
            "ask_clarification": 3,
        }
        selected.sort(key=lambda m: (order.get(m.semantic_act, 9), -m.score, m.id))
        selected = selected[:max_sentences]
        trace = {
            "hard_gates_applied": True,
            "ranker": self.ranker.name,
            "ranker_version": self.ranker.version,
            "candidate_count": len(candidates),
            "allowed_count": len(allowed_moves),
            "selected_count": len(selected),
            "selection_rule": "warrant gate, then semantic-act score, then discourse order",
        }
        return candidates, selected, trace


class DiscoursePlanner:
    def __init__(self, runtime: "WarrantedLanguageRuntime"):
        self.runtime = runtime
        self.policy = GuardedSemanticActionPolicy(self._load_active_ranker())

    def _load_active_ranker(self) -> SemanticActionRanker:
        row = self.runtime.ledger.db.conn.execute(
            "SELECT * FROM semantic_policy_models WHERE active = 1 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return SemanticActionRanker()
        return SemanticActionRanker(
            json.loads(row["weights_json"]), name=row["name"], version=row["version"]
        )

    def save_policy(self, *, name: str | None = None, training_metadata: dict[str, Any] | None = None, activate: bool = True) -> dict[str, Any]:
        ranker = self.policy.ranker
        if name:
            ranker.name = name
        payload = ranker.model_payload(training_metadata)
        digest = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
        model_id = new_id("spm")
        if activate:
            self.runtime.ledger.db.conn.execute("UPDATE semantic_policy_models SET active = 0")
        self.runtime.ledger.db.conn.execute(
            """INSERT INTO semantic_policy_models
               (id, name, version, labels_json, feature_schema_json, weights_json,
                training_metadata_json, model_hash, active, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                model_id,
                payload["name"],
                payload["version"],
                stable_json(payload["labels"]),
                stable_json(payload["feature_schema"]),
                stable_json(payload["weights"]),
                stable_json(payload["training_metadata"]),
                digest,
                int(activate),
                utcnow(),
            ),
        )
        self.runtime.ledger._event(
            "semantic_policy_model",
            model_id,
            "SEMANTIC_POLICY_MODEL_SAVED",
            {"model_hash": digest, "active": activate, "training_metadata": training_metadata or {}},
            "language_runtime",
            ActorRole.TOOL,
        )
        self.runtime.ledger.db.conn.commit()
        return {"id": model_id, "model_hash": digest, "active": activate, **payload}

    def train(self, examples: Iterable[dict[str, Any]], *, epochs: int = 5, rate: float = 0.1, activate: bool = True) -> dict[str, Any]:
        records = list(examples)
        result = self.policy.ranker.train_pairwise(records, epochs=epochs, rate=rate)
        saved = self.save_policy(training_metadata={**result, "training_kind": "pairwise_semantic_act_preferences"}, activate=activate)
        return {"training": result, "model": saved}

    def plan(
        self,
        frame: "SemanticFrame",
        grounding: dict[str, "GroundingReport"],
        candidate_sentences: list["PlannedSentence"],
    ) -> DiscoursePlan:
        candidates, selected, trace = self.policy.select(frame, candidate_sentences)
        context = {
            "grounding": {key: report.as_dict() for key, report in grounding.items()},
            "candidate_sentence_count": len(candidate_sentences),
        }
        payload = {
            "schema_version": DISCOURSE_API_VERSION,
            "utterance": frame.utterance,
            "frame": frame.as_dict(),
            "context": context,
            "candidates": [move.as_dict() for move in candidates],
            "selected": [move.as_dict() for move in selected],
            "policy_version": self.policy.version,
            "policy_trace": trace,
        }
        digest = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
        plan_id = new_id("dsp")
        self.runtime.ledger.db.conn.execute(
            """INSERT INTO discourse_plans
               (id, response_id, utterance, frame_json, context_json,
                candidate_moves_json, selected_moves_json, policy_version,
                policy_trace_json, plan_hash, created_at)
               VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                plan_id,
                frame.utterance,
                stable_json(frame.as_dict()),
                stable_json(context),
                stable_json([move.as_dict() for move in candidates]),
                stable_json([move.as_dict() for move in selected]),
                self.policy.version,
                stable_json(trace),
                digest,
                utcnow(),
            ),
        )
        self.runtime.ledger.db.conn.commit()
        for move in selected:
            move.sentence.trace = {
                **move.sentence.trace,
                "discourse_plan_hash": digest,
                "discourse_move_id": move.id,
                "semantic_policy_score": move.score,
                "semantic_policy_version": self.policy.version,
            }
        return DiscoursePlan(
            plan_id,
            frame.utterance,
            frame.as_dict(),
            context,
            candidates,
            selected,
            self.policy.version,
            trace,
            digest,
        )

    def bind_response(self, plan_id: str, response_id: str) -> None:
        self.runtime.ledger.db.conn.execute(
            "UPDATE discourse_plans SET response_id = ? WHERE id = ? AND response_id IS NULL",
            (response_id, plan_id),
        )
        self.runtime.ledger.db.conn.commit()

    def get(self, plan_id: str) -> dict[str, Any]:
        row = self.runtime.ledger.db.conn.execute(
            "SELECT * FROM discourse_plans WHERE id = ?", (plan_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown discourse plan: {plan_id}")
        result = dict(row)
        for key in ("frame_json", "context_json", "candidate_moves_json", "selected_moves_json", "policy_trace_json"):
            result[key.removesuffix("_json")] = json.loads(result.pop(key))
        return result

    def verify(self, plan_id: str) -> bool:
        plan = self.get(plan_id)
        payload = {
            "schema_version": DISCOURSE_API_VERSION,
            "utterance": plan["utterance"],
            "frame": plan["frame"],
            "context": plan["context"],
            "candidates": plan["candidate_moves"],
            "selected": plan["selected_moves"],
            "policy_version": plan["policy_version"],
            "policy_trace": plan["policy_trace"],
        }
        return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest() == plan["plan_hash"]
