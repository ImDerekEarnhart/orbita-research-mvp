from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Iterable

from .models import ActorRole, EvidenceKind, ObjectKind, Stance, SupportState
from .relations import normalize_identifier, normalize_name

if TYPE_CHECKING:  # pragma: no cover
    from .ledger import EpistemicLedger


LANGUAGE_API_VERSION = "1.2"


class SemanticIntent(StrEnum):
    DEFINE = "define"
    DESCRIBE = "describe"
    VERIFY_RELATION = "verify_relation"
    EXPLAIN_RELATION = "explain_relation"
    LIST_OBJECTS = "list_objects"
    LIST_CONTRADICTIONS = "list_contradictions"
    UNKNOWN = "unknown"


class SemanticAct(StrEnum):
    STATE_SUPPORTED_CLAIM = "state_supported_claim"
    STATE_CHALLENGED_CLAIM = "state_challenged_claim"
    STATE_REFUTED_CLAIM = "state_refuted_claim"
    EXPLAIN_WARRANT = "explain_warrant"
    EXPLAIN_CONTRADICTION = "explain_contradiction"
    ABSTAIN = "abstain"
    ASK_CLARIFICATION = "ask_clarification"


class GroundingState(StrEnum):
    GROUNDED = "grounded"
    GROUNDED_BY_USE = "grounded_by_use"
    UNKNOWN_ENTITY = "unknown_entity"
    AMBIGUOUS_ENTITY = "ambiguous_entity"
    UNRESOLVED_CYCLE = "unresolved_cycle"
    DEPTH_LIMIT = "depth_limit"
    UNGROUNDED = "ungrounded"


@dataclass(slots=True)
class SemanticFrame:
    utterance: str
    intent: SemanticIntent
    subject: str | None = None
    predicate: str | None = None
    object_value: str | None = None
    predicate_candidates: list[str] = field(default_factory=list)
    polarity: bool = True
    confidence: float = 0.0
    parser_trace: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "utterance": self.utterance,
            "intent": self.intent.value,
            "subject": self.subject,
            "predicate": self.predicate,
            "object_value": self.object_value,
            "predicate_candidates": list(self.predicate_candidates),
            "polarity": self.polarity,
            "confidence": self.confidence,
            "parser_trace": list(self.parser_trace),
        }


@dataclass(slots=True)
class GroundingReport:
    reference: str
    state: GroundingState
    entity_id: str | None = None
    entity_name: str | None = None
    path: list[str] = field(default_factory=list)
    cycles: list[list[str]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    depth: int = 0

    @property
    def resolved(self) -> bool:
        return self.entity_id is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "state": self.state.value,
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "path": list(self.path),
            "cycles": [list(cycle) for cycle in self.cycles],
            "missing": list(self.missing),
            "depth": self.depth,
            "resolved": self.resolved,
        }


@dataclass(slots=True)
class PlannedSentence:
    semantic_act: SemanticAct
    text: str
    claim_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    proof_ids: list[str] = field(default_factory=list)
    support_state: str = SupportState.UNKNOWN.value
    trace: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "semantic_act": self.semantic_act.value,
            "text": self.text,
            "claim_ids": list(dict.fromkeys(self.claim_ids)),
            "evidence_ids": list(dict.fromkeys(self.evidence_ids)),
            "proof_ids": list(dict.fromkeys(self.proof_ids)),
            "support_state": self.support_state,
            "trace": self.trace,
        }


@dataclass(slots=True)
class LanguageResponse:
    response_id: str
    utterance: str
    frame: SemanticFrame
    answer_text: str
    sentences: list[PlannedSentence]
    grounding: dict[str, GroundingReport]
    status: str
    response_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.response_id,
            "utterance": self.utterance,
            "frame": self.frame.as_dict(),
            "answer_text": self.answer_text,
            "sentences": [sentence.as_dict() for sentence in self.sentences],
            "grounding": {key: value.as_dict() for key, value in self.grounding.items()},
            "status": self.status,
            "response_hash": self.response_hash,
        }


def _stable_json(value: Any) -> str:
    from .ledger import stable_json

    return stable_json(value)


def _new_id(prefix: str) -> str:
    from .ledger import new_id

    return new_id(prefix)


def _utcnow() -> str:
    from .ledger import utcnow

    return utcnow()


def _clean_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    cleaned = re.sub(r"^(please\s+|could you\s+|would you\s+|can you\s+)", "", cleaned, flags=re.I)
    return cleaned.strip()


def _strip_determiner(value: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", value.strip(), flags=re.I).strip()


def _sentence_case(value: str) -> str:
    if not value:
        return value
    return value[:1].upper() + value[1:]


def _lower_initial(value: str) -> str:
    if not value or value.isupper():
        return value
    return value[:1].lower() + value[1:]


def _article(phrase: str) -> str:
    if not phrase:
        return "a"
    return "an" if phrase[0].lower() in "aeiou" else "a"


def _humanize_predicate(predicate: str) -> str:
    return predicate.replace("_", " ")


DEFAULT_PREDICATE_ALIASES: dict[str, str] = {
    "is a": "is_a",
    "is an": "is_a",
    "are a": "is_a",
    "are an": "is_a",
    "has property": "has_property",
    "has": "has_property",
    "have": "has_property",
    "lives in": "lives_in",
    "live in": "lives_in",
    "works at": "works_at",
    "work at": "works_at",
    "eats": "eats",
    "eat": "eats",
    "causes": "causes",
    "cause": "causes",
    "positively correlates with": "positively_correlates_with",
    "positively correlate with": "positively_correlates_with",
    "correlates positively with": "positively_correlates_with",
    "is positively associated with": "positively_correlates_with",
    "is associated with": "associated_with",
    "defined as": "defined_as",
    "means": "defined_as",
}


class ControlledSemanticParser:
    """Parse a bounded, auditable subset of English into semantic frames.

    This parser deliberately prefers abstention over guessing. It uses the
    runtime's registered predicate aliases and emits a trace explaining which
    pattern fired.
    """

    def __init__(self, runtime: "WarrantedLanguageRuntime"):
        self.runtime = runtime

    def parse(self, utterance: str) -> SemanticFrame:
        text = _clean_text(utterance)
        if not text:
            return SemanticFrame(utterance, SemanticIntent.UNKNOWN, confidence=0.0)
        body = text.rstrip(" ?.!")
        lower = body.casefold()

        for prefix in ("what evidence supports the claim that ", "what supports the claim that "):
            if lower.startswith(prefix):
                nested = body[len(prefix):]
                frame = self._parse_relation(nested, explain=True)
                frame.parser_trace.insert(0, "support_question_wrapper")
                return frame

        for prefix in ("what contradicts ", "which claims contradict "):
            if lower.startswith(prefix):
                target = _strip_determiner(body[len(prefix):])
                return SemanticFrame(
                    utterance,
                    SemanticIntent.LIST_CONTRADICTIONS,
                    subject=target,
                    confidence=0.92,
                    parser_trace=["contradiction_query"],
                )

        match = re.fullmatch(r"(?:what|who)\s+(?:is|are)\s+(.+)", lower)
        if match:
            subject = _strip_determiner(body[match.start(1): match.end(1)])
            return SemanticFrame(
                utterance,
                SemanticIntent.DEFINE,
                subject=subject,
                confidence=0.96,
                parser_trace=["definition_question"],
            )

        match = re.fullmatch(r"(?:tell me about|describe|explain)\s+(.+)", lower)
        if match:
            subject = _strip_determiner(body[match.start(1): match.end(1)])
            return SemanticFrame(
                utterance,
                SemanticIntent.DESCRIBE,
                subject=subject,
                confidence=0.93,
                parser_trace=["description_request"],
            )

        if lower.startswith("why "):
            return self._parse_relation(body[4:], explain=True, utterance=utterance)
        if lower.startswith("is ") or lower.startswith("are ") or lower.startswith("does "):
            return self._parse_relation(body, explain=False, utterance=utterance)

        match = re.fullmatch(r"what\s+does\s+(.+?)\s+(.+)", lower)
        if match:
            subject = body[match.start(1): match.end(1)]
            predicate_phrase = body[match.start(2): match.end(2)]
            predicate = self.runtime.resolve_predicate_phrase(predicate_phrase, require_existing=False)
            return SemanticFrame(
                utterance,
                SemanticIntent.LIST_OBJECTS,
                subject=_strip_determiner(subject),
                predicate=predicate,
                confidence=0.78 if predicate else 0.45,
                parser_trace=["open_object_question"],
            )

        return SemanticFrame(
            utterance,
            SemanticIntent.UNKNOWN,
            confidence=0.15,
            parser_trace=["no_safe_pattern"],
        )

    def _parse_relation(
        self,
        body: str,
        *,
        explain: bool,
        utterance: str | None = None,
    ) -> SemanticFrame:
        original = utterance or body
        stripped = body.strip().rstrip(" ?.!")
        lower = stripped.casefold()
        intent = SemanticIntent.EXPLAIN_RELATION if explain else SemanticIntent.VERIFY_RELATION

        be_match = re.fullmatch(r"(?:is|are)\s+(.+)", lower)
        if be_match:
            content = stripped[be_match.start(1): be_match.end(1)]
            split = self.runtime.split_copular_entities(content)
            if split is not None:
                subject, obj = split
                return SemanticFrame(
                    original,
                    intent,
                    subject=subject,
                    object_value=obj,
                    predicate_candidates=["is_a", "has_property"],
                    confidence=0.9,
                    parser_trace=["copular_relation_question", "entity_split_by_ledger"],
                )

        does_match = re.fullmatch(r"does\s+(.+)", lower)
        if does_match:
            content_start = does_match.start(1)
            content = stripped[content_start:]
            split = self.runtime.split_known_predicate(content)
            if split is not None:
                subject, predicate, obj, phrase = split
                return SemanticFrame(
                    original,
                    intent,
                    subject=_strip_determiner(subject),
                    predicate=predicate,
                    object_value=_strip_determiner(obj),
                    confidence=0.9,
                    parser_trace=[f"does_relation_question:{phrase}"],
                )

        split = self.runtime.split_known_predicate(stripped)
        if split is not None:
            subject, predicate, obj, phrase = split
            return SemanticFrame(
                original,
                intent,
                subject=_strip_determiner(subject),
                predicate=predicate,
                object_value=_strip_determiner(obj),
                confidence=0.76,
                parser_trace=[f"bare_relation:{phrase}"],
            )

        return SemanticFrame(
            original,
            SemanticIntent.UNKNOWN,
            confidence=0.2,
            parser_trace=["relation_pattern_unresolved"],
        )


class WarrantedLanguageRuntime:
    """Meaning-first language interface over Orbita's canonical ledger.

    The runtime never promotes text directly into truth. It parses the question,
    grounds references against typed entities, queries current warrant, selects
    semantic acts, realizes them with controlled grammar, and persists a
    sentence-level receipt for every response.
    """

    def __init__(self, ledger: "EpistemicLedger"):
        self.ledger = ledger
        from .support import SupportEngine

        self.support = SupportEngine(ledger)
        self.parser = ControlledSemanticParser(self)
        from .discourse import DiscoursePlanner

        self.discourse = DiscoursePlanner(self)

    # ------------------------------------------------------------------
    # Lexicon and parsing
    # ------------------------------------------------------------------
    def register_predicate_alias(
        self,
        alias: str,
        predicate: str,
        *,
        actor: str = "user",
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> None:
        predicate_id = self.ledger.relations.resolve_predicate(predicate)
        normalized = normalize_name(alias)
        if not normalized:
            raise ValueError("alias cannot be empty")
        existing = self.ledger.db.conn.execute(
            "SELECT predicate_id FROM language_predicate_aliases WHERE alias_normalized = ?",
            (normalized,),
        ).fetchone()
        if existing is not None and existing["predicate_id"] != predicate_id:
            raise ValueError(f"Predicate phrase {alias!r} is already assigned")
        self.ledger.db.conn.execute(
            """INSERT OR REPLACE INTO language_predicate_aliases
               (alias_normalized, alias_text, predicate_id, created_at)
               VALUES (?, ?, ?, ?)""",
            (normalized, alias.strip(), predicate_id, _utcnow()),
        )
        self.ledger._event(
            "predicate",
            predicate_id,
            "LANGUAGE_PREDICATE_ALIAS_REGISTERED",
            {"alias": alias, "alias_normalized": normalized},
            actor,
            actor_role,
        )
        self.ledger.db.conn.commit()

    def predicate_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        rows = self.ledger.db.conn.execute(
            """SELECT l.alias_normalized, p.normalized_name
               FROM language_predicate_aliases l
               JOIN predicates p ON p.id = l.predicate_id"""
        ).fetchall()
        for row in rows:
            aliases[row["alias_normalized"]] = row["normalized_name"]
        for phrase, predicate in DEFAULT_PREDICATE_ALIASES.items():
            try:
                self.ledger.relations.resolve_predicate(predicate)
            except KeyError:
                continue
            aliases.setdefault(normalize_name(phrase), predicate)
        for row in self.ledger.db.conn.execute(
            "SELECT normalized_name FROM predicates"
        ).fetchall():
            predicate = row["normalized_name"]
            aliases.setdefault(predicate, predicate)
            aliases.setdefault(predicate.replace("_", " "), predicate)
        return aliases

    def resolve_predicate_phrase(self, phrase: str, *, require_existing: bool = True) -> str | None:
        normalized = normalize_name(phrase)
        aliases = self.predicate_aliases()
        predicate = aliases.get(normalized)
        if predicate is not None:
            return predicate
        candidate = normalize_identifier(phrase)
        if candidate:
            try:
                self.ledger.relations.resolve_predicate(candidate)
                return candidate
            except KeyError:
                pass
        if require_existing:
            raise KeyError(f"Unknown predicate phrase: {phrase}")
        return None

    def split_copular_entities(self, text: str) -> tuple[str, str] | None:
        words = text.strip().split()
        if words and words[0].casefold() in {"a", "an", "the"}:
            words = words[1:]
        candidates: list[tuple[int, str, str]] = []
        for index in range(1, len(words)):
            subject = " ".join(words[:index]).strip()
            object_words = words[index:]
            if object_words and object_words[0].casefold() in {"a", "an", "the"}:
                object_words = object_words[1:]
            obj = " ".join(object_words).strip()
            if not subject or not obj:
                continue
            try:
                subject_id = self.ledger.relations.resolve_entity(subject)
                object_id = self.ledger.relations.resolve_entity(obj)
            except (KeyError, ValueError):
                continue
            score = 0
            for predicate in ("is_a", "has_property"):
                try:
                    rows = self.ledger.relations.find_relation_claims(
                        subject=subject_id,
                        predicate=predicate,
                        object_value=object_id,
                        object_kind=ObjectKind.ENTITY,
                    )
                except KeyError:
                    rows = []
                if rows:
                    score += 10
            candidates.append((score, subject, obj))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], -(len(item[1]) + len(item[2]))))
        _, subject, obj = candidates[0]
        return subject, obj

    def split_known_predicate(self, text: str) -> tuple[str, str, str, str] | None:
        normalized_text = normalize_name(text)
        aliases = sorted(self.predicate_aliases().items(), key=lambda item: len(item[0]), reverse=True)
        for phrase, predicate in aliases:
            marker = f" {phrase} "
            index = normalized_text.find(marker)
            if index <= 0:
                continue
            subject = normalized_text[:index].strip()
            obj = normalized_text[index + len(marker):].strip()
            if subject and obj:
                return subject, predicate, obj, phrase
        return None

    def interpret(self, utterance: str) -> dict[str, Any]:
        return self.parser.parse(utterance).as_dict()

    # ------------------------------------------------------------------
    # Grounding
    # ------------------------------------------------------------------
    def ground_reference(self, reference: str, *, max_depth: int = 12) -> GroundingReport:
        try:
            entity_id = self.ledger.relations.resolve_entity(reference)
        except KeyError:
            return GroundingReport(reference, GroundingState.UNKNOWN_ENTITY, missing=[reference])
        except ValueError:
            return GroundingReport(reference, GroundingState.AMBIGUOUS_ENTITY, missing=[reference])
        entity = self.ledger.relations.get_entity(entity_id)
        return self._ground_entity(entity_id, reference=reference, max_depth=max_depth, stack=[])

    def _ground_entity(
        self,
        entity_id: str,
        *,
        reference: str,
        max_depth: int,
        stack: list[str],
    ) -> GroundingReport:
        entity = self.ledger.relations.get_entity(entity_id)
        if entity.get("metadata", {}).get("primitive") is True or entity["entity_type"] == "primitive":
            return GroundingReport(
                reference,
                GroundingState.GROUNDED,
                entity_id=entity_id,
                entity_name=entity["canonical_name"],
                path=[*stack, entity_id],
                depth=len(stack),
            )
        if entity_id in stack:
            cycle_start = stack.index(entity_id)
            cycle = stack[cycle_start:] + [entity_id]
            return GroundingReport(
                reference,
                GroundingState.UNRESOLVED_CYCLE,
                entity_id=entity_id,
                entity_name=entity["canonical_name"],
                path=[*stack, entity_id],
                cycles=[cycle],
                depth=len(stack),
            )
        if len(stack) >= max_depth:
            return GroundingReport(
                reference,
                GroundingState.DEPTH_LIMIT,
                entity_id=entity_id,
                entity_name=entity["canonical_name"],
                path=[*stack, entity_id],
                depth=len(stack),
            )

        definitions: list[dict[str, Any]] = []
        for predicate in ("defined_as", "is_a"):
            try:
                definitions.extend(
                    self.ledger.relations.find_relation_claims(
                        subject=entity_id,
                        predicate=predicate,
                        polarity=True,
                    )
                )
            except KeyError:
                continue
        supported_defs = [
            item for item in definitions
            if self.support.evaluate(item["claim_id"]).state in {SupportState.SUPPORTED, SupportState.CHALLENGED}
        ]
        if not supported_defs:
            usage = self.ledger.db.conn.execute(
                """SELECT rc.claim_id FROM relation_claims rc
                   WHERE rc.subject_entity_id = ? OR rc.object_entity_id = ?""",
                (entity_id, entity_id),
            ).fetchall()
            if any(
                self.support.evaluate(row["claim_id"]).state in {SupportState.SUPPORTED, SupportState.CHALLENGED}
                for row in usage
            ):
                return GroundingReport(
                    reference,
                    GroundingState.GROUNDED_BY_USE,
                    entity_id=entity_id,
                    entity_name=entity["canonical_name"],
                    path=[*stack, entity_id],
                    depth=len(stack),
                )
            return GroundingReport(
                reference,
                GroundingState.UNGROUNDED,
                entity_id=entity_id,
                entity_name=entity["canonical_name"],
                path=[*stack, entity_id],
                missing=[entity["canonical_name"]],
                depth=len(stack),
            )

        children: list[GroundingReport] = []
        for definition in supported_defs:
            if definition["object_kind"] == ObjectKind.LITERAL.value:
                return GroundingReport(
                    reference,
                    GroundingState.GROUNDED,
                    entity_id=entity_id,
                    entity_name=entity["canonical_name"],
                    path=[*stack, entity_id],
                    depth=len(stack),
                )
            children.append(
                self._ground_entity(
                    definition["object_entity_id"],
                    reference=definition["object"]["name"],
                    max_depth=max_depth,
                    stack=[*stack, entity_id],
                )
            )
        if any(child.state in {GroundingState.GROUNDED, GroundingState.GROUNDED_BY_USE} for child in children):
            chosen = next(
                child for child in children
                if child.state in {GroundingState.GROUNDED, GroundingState.GROUNDED_BY_USE}
            )
            return GroundingReport(
                reference,
                GroundingState.GROUNDED,
                entity_id=entity_id,
                entity_name=entity["canonical_name"],
                path=[entity_id, *chosen.path],
                depth=chosen.depth + 1,
            )
        cycles = [cycle for child in children for cycle in child.cycles]
        if cycles:
            return GroundingReport(
                reference,
                GroundingState.UNRESOLVED_CYCLE,
                entity_id=entity_id,
                entity_name=entity["canonical_name"],
                path=[*stack, entity_id],
                cycles=cycles,
                depth=max((child.depth for child in children), default=len(stack)),
            )
        if any(child.state == GroundingState.DEPTH_LIMIT for child in children):
            return GroundingReport(
                reference,
                GroundingState.DEPTH_LIMIT,
                entity_id=entity_id,
                entity_name=entity["canonical_name"],
                path=[*stack, entity_id],
                depth=max((child.depth for child in children), default=len(stack)),
            )
        return GroundingReport(
            reference,
            GroundingState.UNGROUNDED,
            entity_id=entity_id,
            entity_name=entity["canonical_name"],
            path=[*stack, entity_id],
            missing=list(dict.fromkeys(item for child in children for item in child.missing)),
            depth=max((child.depth for child in children), default=len(stack)),
        )

    # ------------------------------------------------------------------
    # Answering and sentence warrants
    # ------------------------------------------------------------------
    def ask(self, utterance: str) -> dict[str, Any]:
        frame = self.parser.parse(utterance)
        grounding: dict[str, GroundingReport] = {}
        if frame.subject:
            grounding["subject"] = self.ground_reference(frame.subject)
        if frame.object_value:
            grounding["object"] = self.ground_reference(frame.object_value)
        candidate_sentences = self._plan(frame, grounding)
        discourse_plan = self.discourse.plan(frame, grounding, candidate_sentences)
        sentences = [move.sentence for move in discourse_plan.selected]
        if not sentences:
            sentences = [self._clarification("I could not select a warranted semantic move.")]
        answer_text = " ".join(sentence.text for sentence in sentences).strip()
        status = "answered" if any(s.semantic_act not in {SemanticAct.ABSTAIN, SemanticAct.ASK_CLARIFICATION} for s in sentences) else "abstained"
        response = self._persist_response(frame, grounding, sentences, answer_text, status)
        self.discourse.bind_response(discourse_plan.id, response.response_id)
        result = response.as_dict()
        result["discourse_plan_id"] = discourse_plan.id
        result["discourse_plan_hash"] = discourse_plan.plan_hash
        result["discourse_policy_version"] = discourse_plan.policy_version
        return result

    def plan_discourse(self, utterance: str) -> dict[str, Any]:
        """Return the meaning-level answer plan without emitting a response."""
        frame = self.parser.parse(utterance)
        grounding: dict[str, GroundingReport] = {}
        if frame.subject:
            grounding["subject"] = self.ground_reference(frame.subject)
        if frame.object_value:
            grounding["object"] = self.ground_reference(frame.object_value)
        candidate_sentences = self._plan(frame, grounding)
        return self.discourse.plan(frame, grounding, candidate_sentences).as_dict()

    def train_semantic_policy(
        self,
        examples: Iterable[dict[str, Any]],
        *,
        epochs: int = 5,
        rate: float = 0.1,
        activate: bool = True,
    ) -> dict[str, Any]:
        return self.discourse.train(examples, epochs=epochs, rate=rate, activate=activate)

    def _plan(
        self,
        frame: SemanticFrame,
        grounding: dict[str, GroundingReport],
    ) -> list[PlannedSentence]:
        if frame.intent == SemanticIntent.UNKNOWN:
            return [self._clarification("I could not map that request to a safe semantic query.")]
        subject_ground = grounding.get("subject")
        if subject_ground is None or not subject_ground.resolved:
            target = frame.subject or "that subject"
            return [self._clarification(f"I do not recognize {target!r} as a grounded entity.")]

        if frame.intent in {SemanticIntent.DEFINE, SemanticIntent.DESCRIBE}:
            return self._plan_description(subject_ground.entity_id or "", frame.intent)
        if frame.intent == SemanticIntent.LIST_CONTRADICTIONS:
            return self._plan_contradictions(subject_ground.entity_id or "")
        if frame.intent == SemanticIntent.LIST_OBJECTS:
            if not frame.predicate:
                return [self._clarification("I recognized the subject but not the requested relation.")]
            return self._plan_open_relation(subject_ground.entity_id or "", frame.predicate)
        if frame.intent in {SemanticIntent.VERIFY_RELATION, SemanticIntent.EXPLAIN_RELATION}:
            object_ground = grounding.get("object")
            if object_ground is None or not object_ground.resolved:
                target = frame.object_value or "that object"
                return [self._clarification(f"I do not recognize {target!r} as a grounded entity.")]
            return self._plan_relation(
                subject_ground.entity_id or "",
                object_ground.entity_id or "",
                frame,
                explain=frame.intent == SemanticIntent.EXPLAIN_RELATION,
            )
        return [self._clarification("I could not safely answer that request.")]

    def _plan_description(self, subject_id: str, intent: SemanticIntent) -> list[PlannedSentence]:
        relations = self._relations_for_subject(subject_id)
        if not relations:
            subject = self.ledger.relations.get_entity(subject_id)["canonical_name"]
            return [self._abstain(f"I do not have enough warranted information to describe {subject}.")]
        preferred = sorted(
            relations,
            key=lambda item: (
                {SupportState.SUPPORTED: 0, SupportState.CHALLENGED: 1}.get(item["support"].state, 2),
                {"defined_as": 0, "is_a": 1, "has_property": 2}.get(item["predicate"], 3),
                item["claim_id"],
            ),
        )
        limit = 1 if intent == SemanticIntent.DEFINE else 5
        sentences: list[PlannedSentence] = []
        for relation in preferred[:limit]:
            if relation["support"].state == SupportState.SUPPORTED:
                sentences.append(self._factual_sentence(relation, SemanticAct.STATE_SUPPORTED_CLAIM))
            elif relation["support"].state == SupportState.CHALLENGED:
                sentences.append(self._challenged_sentence(relation))
        return sentences or [self._abstain("No currently warranted description is available.")]

    def _plan_open_relation(self, subject_id: str, predicate: str) -> list[PlannedSentence]:
        try:
            relations = self.ledger.relations.find_relation_claims(
                subject=subject_id, predicate=predicate, polarity=True
            )
        except KeyError:
            return [self._clarification(f"I do not recognize the relation {_humanize_predicate(predicate)!r}.")]
        enriched = [self._enrich_relation(item) for item in relations]
        usable = [item for item in enriched if item["support"].state in {SupportState.SUPPORTED, SupportState.CHALLENGED}]
        if not usable:
            return [self._abstain("I do not have a warranted answer for that relation.")]
        return [
            self._factual_sentence(item, SemanticAct.STATE_SUPPORTED_CLAIM)
            if item["support"].state == SupportState.SUPPORTED
            else self._challenged_sentence(item)
            for item in usable[:5]
        ]

    def _plan_relation(
        self,
        subject_id: str,
        object_id: str,
        frame: SemanticFrame,
        *,
        explain: bool,
    ) -> list[PlannedSentence]:
        candidates = [frame.predicate] if frame.predicate else list(frame.predicate_candidates)
        matches: list[dict[str, Any]] = []
        negative_matches: list[dict[str, Any]] = []
        for predicate in [item for item in candidates if item]:
            try:
                matches.extend(
                    self.ledger.relations.find_relation_claims(
                        subject=subject_id,
                        predicate=predicate,
                        object_value=object_id,
                        object_kind=ObjectKind.ENTITY,
                        polarity=True,
                    )
                )
                negative_matches.extend(
                    self.ledger.relations.find_relation_claims(
                        subject=subject_id,
                        predicate=predicate,
                        object_value=object_id,
                        object_kind=ObjectKind.ENTITY,
                        polarity=False,
                    )
                )
            except KeyError:
                continue
        positives = [self._enrich_relation(item) for item in matches]
        negatives = [self._enrich_relation(item) for item in negative_matches]
        positive = self._best_relation(positives)
        negative = self._best_relation(negatives)

        sentences: list[PlannedSentence] = []
        if positive and positive["support"].state == SupportState.SUPPORTED:
            sentences.append(self._factual_sentence(positive, SemanticAct.STATE_SUPPORTED_CLAIM, yes_no=True))
            if explain:
                sentences.extend(self._explanation_sentences(positive))
            return sentences
        if negative and negative["support"].state == SupportState.SUPPORTED:
            sentences.append(self._factual_sentence(negative, SemanticAct.STATE_REFUTED_CLAIM, yes_no=True))
            if explain:
                sentences.extend(self._explanation_sentences(negative))
            return sentences
        challenged = next(
            (item for item in [positive, negative] if item and item["support"].state == SupportState.CHALLENGED),
            None,
        )
        if challenged:
            sentences.append(self._challenged_sentence(challenged, yes_no=True))
            if explain:
                sentences.extend(self._explanation_sentences(challenged))
            return sentences

        subject = self.ledger.relations.get_entity(subject_id)["canonical_name"]
        obj = self.ledger.relations.get_entity(object_id)["canonical_name"]
        predicate = frame.predicate or (frame.predicate_candidates[0] if frame.predicate_candidates else "related_to")
        sentence = f"I do not have enough warranted information to determine whether {subject} {_humanize_predicate(predicate)} {obj}."
        claim_ids = [item["claim_id"] for item in [positive, negative] if item]
        return [
            PlannedSentence(
                SemanticAct.ABSTAIN,
                sentence,
                claim_ids=claim_ids,
                support_state=SupportState.UNKNOWN.value,
                trace={"reason": "no supported positive or negative claim"},
            )
        ]

    def _plan_contradictions(self, subject_id: str) -> list[PlannedSentence]:
        relations = self._relations_for_subject(subject_id, include_unknown=True)
        contradictory: list[dict[str, Any]] = []
        for relation in relations:
            if relation["support"].contradictions:
                contradictory.append(relation)
        if not contradictory:
            subject = self.ledger.relations.get_entity(subject_id)["canonical_name"]
            return [self._abstain(f"I found no actively supported contradiction involving {subject}.")]
        sentences: list[PlannedSentence] = []
        for relation in contradictory[:5]:
            lineage = self._collect_warrant(relation["claim_id"], include_refute=True)
            text = f"The claim that {self._relation_clause(relation)} has an actively supported contradiction."
            sentences.append(
                PlannedSentence(
                    SemanticAct.EXPLAIN_CONTRADICTION,
                    text,
                    claim_ids=lineage["claim_ids"],
                    evidence_ids=lineage["evidence_ids"],
                    proof_ids=lineage["proof_ids"],
                    support_state=relation["support"].state.value,
                    trace={"contradictions": relation["support"].contradictions},
                )
            )
        return sentences

    def _relations_for_subject(self, subject_id: str, *, include_unknown: bool = False) -> list[dict[str, Any]]:
        relations = self.ledger.relations.find_relation_claims(subject=subject_id, polarity=True)
        enriched = [self._enrich_relation(item) for item in relations]
        if include_unknown:
            return enriched
        return [
            item for item in enriched
            if item["support"].state in {SupportState.SUPPORTED, SupportState.CHALLENGED}
        ]

    def _enrich_relation(self, relation: dict[str, Any]) -> dict[str, Any]:
        result = dict(relation)
        result["support"] = self.support.evaluate(relation["claim_id"])
        return result

    @staticmethod
    def _best_relation(relations: list[dict[str, Any]]) -> dict[str, Any] | None:
        ranking = {
            SupportState.SUPPORTED: 0,
            SupportState.CHALLENGED: 1,
            SupportState.UNSUPPORTED: 2,
            SupportState.UNKNOWN: 3,
        }
        return min(relations, key=lambda item: ranking[item["support"].state], default=None)

    def _factual_sentence(
        self,
        relation: dict[str, Any],
        act: SemanticAct,
        *,
        yes_no: bool = False,
    ) -> PlannedSentence:
        lineage = self._collect_warrant(relation["claim_id"])
        factual = self._realize_relation(relation)
        text = f"Yes. {factual}" if yes_no and relation["polarity"] else f"No. {factual}" if yes_no else factual
        return PlannedSentence(
            act,
            text,
            claim_ids=lineage["claim_ids"],
            evidence_ids=lineage["evidence_ids"],
            proof_ids=lineage["proof_ids"],
            support_state=relation["support"].state.value,
            trace={
                "template": relation["predicate"],
                "grammar_words": ["is", "a", "an", "not", "with", "in", "at"],
                "lineage": lineage,
            },
        )

    def _challenged_sentence(self, relation: dict[str, Any], *, yes_no: bool = False) -> PlannedSentence:
        lineage = self._collect_warrant(relation["claim_id"], include_refute=True)
        clause = self._relation_clause(relation)
        prefix = "The evidence is mixed" if not yes_no else "The evidence is mixed, so I cannot give a clean yes or no"
        text = f"{prefix}: there is support for the claim that {clause}, but active contrary evidence also exists."
        return PlannedSentence(
            SemanticAct.STATE_CHALLENGED_CLAIM,
            text,
            claim_ids=lineage["claim_ids"],
            evidence_ids=lineage["evidence_ids"],
            proof_ids=lineage["proof_ids"],
            support_state=SupportState.CHALLENGED.value,
            trace={"contradictions": relation["support"].contradictions, "lineage": lineage},
        )

    def _explanation_sentences(self, relation: dict[str, Any]) -> list[PlannedSentence]:
        report = relation["support"]
        lineage = self._collect_warrant(relation["claim_id"], include_refute=True)
        parts: list[str] = []
        if report.direct_support_sources:
            count = len(report.direct_support_sources)
            parts.append(f"{count} independent warranting source{'s' if count != 1 else ''}")
        if report.satisfied_proofs:
            count = len(report.satisfied_proofs)
            parts.append(f"{count} complete proof path{'s' if count != 1 else ''}")
        if report.direct_refute_sources or report.contradictions:
            parts.append("active contrary evidence")
        if not parts:
            return []
        text = "Its current status is based on " + ", ".join(parts) + "."
        return [
            PlannedSentence(
                SemanticAct.EXPLAIN_WARRANT,
                text,
                claim_ids=lineage["claim_ids"],
                evidence_ids=lineage["evidence_ids"],
                proof_ids=lineage["proof_ids"],
                support_state=report.state.value,
                trace={"support_report": report.as_dict()},
            )
        ]

    def _collect_warrant(
        self,
        claim_id: str,
        *,
        include_refute: bool = False,
        _visited: set[str] | None = None,
    ) -> dict[str, list[str]]:
        visited = _visited or set()
        if claim_id in visited:
            return {"claim_ids": [], "evidence_ids": [], "proof_ids": []}
        visited.add(claim_id)
        report = self.support.evaluate(claim_id)
        claim_ids = [claim_id]
        evidence_ids: list[str] = []
        proof_ids = list(report.satisfied_proofs)
        rows = self.ledger.db.conn.execute(
            """SELECT a.stance, e.id, e.active, e.source_kind
               FROM attestations a JOIN evidence e ON e.id = a.evidence_id
               WHERE a.claim_id = ?""",
            (claim_id,),
        ).fetchall()
        for row in rows:
            if not row["active"]:
                continue
            if EvidenceKind(row["source_kind"]) in self.ledger.policy.non_warrant_source_kinds:
                continue
            if row["stance"] == Stance.SUPPORT.value or include_refute:
                evidence_ids.append(row["id"])
        for proof_id in report.satisfied_proofs:
            premises = self.ledger.db.conn.execute(
                "SELECT premise_claim_id FROM proof_premises WHERE proof_id = ? ORDER BY position",
                (proof_id,),
            ).fetchall()
            for premise in premises:
                nested = self._collect_warrant(
                    premise["premise_claim_id"], include_refute=include_refute, _visited=visited
                )
                claim_ids.extend(nested["claim_ids"])
                evidence_ids.extend(nested["evidence_ids"])
                proof_ids.extend(nested["proof_ids"])
        if include_refute:
            for contradiction in report.contradictions:
                nested = self._collect_warrant(
                    contradiction, include_refute=False, _visited=visited
                )
                claim_ids.extend(nested["claim_ids"])
                evidence_ids.extend(nested["evidence_ids"])
                proof_ids.extend(nested["proof_ids"])
        return {
            "claim_ids": list(dict.fromkeys(claim_ids)),
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
            "proof_ids": list(dict.fromkeys(proof_ids)),
        }

    def _relation_clause(self, relation: dict[str, Any]) -> str:
        subject = relation["subject_name"]
        predicate = relation["predicate"]
        if relation["object_kind"] == ObjectKind.ENTITY.value:
            obj = relation["object"]["name"]
        else:
            obj = str(relation["object"]["value"])
            if relation["object"].get("unit"):
                obj += f" {relation['object']['unit']}"
        neg = not relation["polarity"]
        if predicate == "is_a":
            obj = _lower_initial(obj)
            article = _article(obj)
            return f"{subject} is {'not ' if neg else ''}{article} {obj}"
        if predicate == "has_property":
            obj = _lower_initial(obj)
            return f"{subject} is {'not ' if neg else ''}{obj}"
        if predicate == "defined_as":
            return f"{subject} {'does not mean' if neg else 'means'} {obj}"
        if predicate == "positively_correlates_with":
            return f"{subject} is {'not ' if neg else ''}positively associated with {obj}"
        phrase = _humanize_predicate(predicate)
        return f"{subject} {'does not ' if neg else ''}{phrase} {obj}"

    def _realize_relation(self, relation: dict[str, Any]) -> str:
        return _sentence_case(self._relation_clause(relation).rstrip(".")) + "."

    @staticmethod
    def _abstain(text: str) -> PlannedSentence:
        return PlannedSentence(
            SemanticAct.ABSTAIN,
            text,
            support_state=SupportState.UNKNOWN.value,
            trace={"reason": "warrant threshold not met"},
        )

    @staticmethod
    def _clarification(text: str) -> PlannedSentence:
        return PlannedSentence(
            SemanticAct.ASK_CLARIFICATION,
            text,
            support_state=SupportState.UNKNOWN.value,
            trace={"reason": "semantic frame unresolved"},
        )

    def _persist_response(
        self,
        frame: SemanticFrame,
        grounding: dict[str, GroundingReport],
        sentences: list[PlannedSentence],
        answer_text: str,
        status: str,
    ) -> LanguageResponse:
        response_id = _new_id("lng")
        created_at = _utcnow()
        sentence_payloads = [sentence.as_dict() for sentence in sentences]
        payload = {
            "schema_version": LANGUAGE_API_VERSION,
            "utterance": frame.utterance,
            "frame": frame.as_dict(),
            "grounding": {key: value.as_dict() for key, value in grounding.items()},
            "answer_text": answer_text,
            "sentences": sentence_payloads,
            "status": status,
        }
        response_hash = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
        self.ledger.db.conn.execute(
            """INSERT INTO language_responses
               (id, utterance, intent, frame_json, grounding_json, answer_text,
                status, response_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                response_id,
                frame.utterance,
                frame.intent.value,
                _stable_json(frame.as_dict()),
                _stable_json({key: value.as_dict() for key, value in grounding.items()}),
                answer_text,
                status,
                response_hash,
                created_at,
            ),
        )
        for position, sentence in enumerate(sentences):
            sentence_payload = sentence.as_dict()
            sentence_hash = hashlib.sha256(_stable_json(sentence_payload).encode("utf-8")).hexdigest()
            self.ledger.db.conn.execute(
                """INSERT INTO sentence_warrants
                   (id, response_id, position, sentence, semantic_act,
                    claim_ids_json, evidence_ids_json, proof_ids_json,
                    support_state, trace_json, sentence_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    _new_id("swr"),
                    response_id,
                    position,
                    sentence.text,
                    sentence.semantic_act.value,
                    _stable_json(sentence_payload["claim_ids"]),
                    _stable_json(sentence_payload["evidence_ids"]),
                    _stable_json(sentence_payload["proof_ids"]),
                    sentence.support_state,
                    _stable_json(sentence.trace),
                    sentence_hash,
                    created_at,
                ),
            )
        self.ledger._event(
            "language_response",
            response_id,
            "WARRANTED_LANGUAGE_RESPONSE_CREATED",
            {
                "intent": frame.intent.value,
                "status": status,
                "response_hash": response_hash,
                "sentence_count": len(sentences),
            },
            "language_runtime",
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()
        return LanguageResponse(
            response_id,
            frame.utterance,
            frame,
            answer_text,
            sentences,
            grounding,
            status,
            response_hash,
        )

    def get_response(self, response_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM language_responses WHERE id = ?", (response_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown language response: {response_id}")
        sentences = self.ledger.db.conn.execute(
            "SELECT * FROM sentence_warrants WHERE response_id = ? ORDER BY position",
            (response_id,),
        ).fetchall()
        result = dict(row)
        result["frame"] = json.loads(result.pop("frame_json"))
        result["grounding"] = json.loads(result.pop("grounding_json"))
        result["sentences"] = []
        for item in sentences:
            sentence = dict(item)
            sentence["claim_ids"] = json.loads(sentence.pop("claim_ids_json"))
            sentence["evidence_ids"] = json.loads(sentence.pop("evidence_ids_json"))
            sentence["proof_ids"] = json.loads(sentence.pop("proof_ids_json"))
            sentence["trace"] = json.loads(sentence.pop("trace_json"))
            result["sentences"].append(sentence)
        discourse = self.ledger.db.conn.execute(
            "SELECT id, plan_hash, policy_version FROM discourse_plans WHERE response_id = ?",
            (response_id,),
        ).fetchone()
        if discourse is not None:
            result["discourse_plan_id"] = discourse["id"]
            result["discourse_plan_hash"] = discourse["plan_hash"]
            result["discourse_policy_version"] = discourse["policy_version"]
        return result

    def verify_response(self, response_id: str) -> bool:
        response = self.get_response(response_id)
        for sentence in response["sentences"]:
            payload = {
                "semantic_act": sentence["semantic_act"],
                "text": sentence["sentence"],
                "claim_ids": sentence["claim_ids"],
                "evidence_ids": sentence["evidence_ids"],
                "proof_ids": sentence["proof_ids"],
                "support_state": sentence["support_state"],
                "trace": sentence["trace"],
            }
            digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
            if digest != sentence["sentence_hash"]:
                return False
        payload = {
            "schema_version": LANGUAGE_API_VERSION,
            "utterance": response["utterance"],
            "frame": response["frame"],
            "grounding": response["grounding"],
            "answer_text": response["answer_text"],
            "sentences": [
                {
                    "semantic_act": sentence["semantic_act"],
                    "text": sentence["sentence"],
                    "claim_ids": sentence["claim_ids"],
                    "evidence_ids": sentence["evidence_ids"],
                    "proof_ids": sentence["proof_ids"],
                    "support_state": sentence["support_state"],
                    "trace": sentence["trace"],
                }
                for sentence in response["sentences"]
            ],
            "status": response["status"],
        }
        return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest() == response["response_hash"]

    # ------------------------------------------------------------------
    # Conservative controlled-text extraction
    # ------------------------------------------------------------------
    def extract_relation_proposals(self, text: str) -> list[dict[str, Any]]:
        proposals: list[dict[str, Any]] = []
        for raw_sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip()):
            sentence = raw_sentence.strip().rstrip(".!?")
            if not sentence:
                continue
            patterns = [
                (r"(.+?)\s+is\s+not\s+(?:a\s+|an\s+)?(.+)", "is_a", False),
                (r"(.+?)\s+is\s+(?:a\s+|an\s+)(.+)", "is_a", True),
                (r"(.+?)\s+lives\s+in\s+(.+)", "lives_in", True),
                (r"(.+?)\s+works\s+at\s+(.+)", "works_at", True),
                (r"(.+?)\s+eats\s+(.+)", "eats", True),
                (r"(.+?)\s+has\s+(.+)", "has_property", True),
            ]
            for pattern, predicate, polarity in patterns:
                match = re.fullmatch(pattern, sentence, flags=re.I)
                if not match:
                    continue
                proposals.append(
                    {
                        "subject": _strip_determiner(match.group(1)),
                        "predicate": predicate,
                        "object": _strip_determiner(match.group(2)),
                        "polarity": polarity,
                        "source_sentence": raw_sentence.strip(),
                        "status": "proposal_only",
                    }
                )
                break
        return proposals

    def ingest_controlled_text(
        self,
        text: str,
        *,
        source_uri: str | None = None,
        independence_key: str | None = None,
        create_evidence: bool = False,
        actor: str = "user",
    ) -> dict[str, Any]:
        proposals = self.extract_relation_proposals(text)
        created: list[str] = []
        evidence_ids: list[str] = []
        for proposal in proposals:
            claim_id = self.ledger.relations.add_relation_claim(
                proposal["subject"],
                proposal["predicate"],
                proposal["object"],
                polarity=proposal["polarity"],
                actor=actor,
                actor_role=ActorRole.HUMAN,
            )
            created.append(claim_id)
            if create_evidence:
                if not source_uri or not independence_key:
                    raise ValueError("source_uri and independence_key are required when create_evidence=True")
                evidence_id = self.ledger.add_evidence(
                    source_uri,
                    proposal["source_sentence"],
                    source_kind=EvidenceKind.HUMAN_TESTIMONY,
                    independence_key=independence_key,
                    content=proposal["source_sentence"],
                    metadata={"extractor": "orbita_controlled_text_v1.1"},
                    actor=actor,
                    actor_role=ActorRole.HUMAN,
                )
                self.ledger.attest(
                    claim_id,
                    evidence_id,
                    Stance.SUPPORT,
                    actor=actor,
                    actor_role=ActorRole.HUMAN,
                )
                evidence_ids.append(evidence_id)
        return {
            "proposals": proposals,
            "claim_ids": list(dict.fromkeys(created)),
            "evidence_ids": evidence_ids,
            "evidence_created": create_evidence,
        }
