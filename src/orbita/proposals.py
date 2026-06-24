from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any, Callable, Protocol, runtime_checkable

from jsonschema import Draft202012Validator

from .models import (
    ActorRole,
    EvidenceKind,
    ObjectKind,
    PredicateRangeKind,
    ProposalBatchStatus,
    ProposalItemStatus,
    ProposalItemType,
    ReviewDecision,
    Stance,
    TypedLiteral,
)
from .relations import normalize_identifier

if False:  # pragma: no cover
    from .ledger import EpistemicLedger


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_proposal_schema() -> dict[str, Any]:
    schema_path = files("orbita").joinpath("schemas/llm_proposal_v1.json")
    return json.loads(schema_path.read_text(encoding="utf-8"))


PROPOSAL_SCHEMA_VERSION = "1.0"
PROPOSAL_SCHEMA = load_proposal_schema()


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    provider: str
    model_name: str
    model_version: str | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider is required")
        if not self.model_name.strip():
            raise ValueError("model_name is required")

    @property
    def actor(self) -> str:
        version = f"@{self.model_version}" if self.model_version else ""
        return f"{self.provider}:{self.model_name}{version}"


@dataclass(frozen=True, slots=True)
class ProposalRequest:
    task: str
    context: dict[str, Any] = field(default_factory=dict)
    allowed_predicates: tuple[str, ...] = ()
    max_proposals: int = 25
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("task is required")
        if not (1 <= self.max_proposals <= 100):
            raise ValueError("max_proposals must be between 1 and 100")


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    response_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    identity: ModelIdentity

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
        generation_parameters: dict[str, Any],
    ) -> LLMResponse: ...


class CallableProvider:
    """Adapter for any SDK without making Orbita depend on a vendor package.

    The callable receives keyword arguments matching ``LLMProvider.generate`` and
    may return an ``LLMResponse``, a raw JSON string, or a mapping containing a
    ``text`` field. This keeps network credentials and vendor code outside the
    epistemic runtime while preserving exact model and prompt provenance.
    """

    def __init__(
        self,
        identity: ModelIdentity,
        fn: Callable[..., LLMResponse | str | dict[str, Any]],
    ) -> None:
        self.identity = identity
        self._fn = fn

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
        generation_parameters: dict[str, Any],
    ) -> LLMResponse:
        value = self._fn(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_schema,
            generation_parameters=generation_parameters,
        )
        if isinstance(value, LLMResponse):
            return value
        if isinstance(value, str):
            return LLMResponse(value)
        if isinstance(value, dict) and isinstance(value.get("text"), str):
            return LLMResponse(
                value["text"],
                response_id=value.get("response_id"),
                usage=dict(value.get("usage") or {}),
                metadata=dict(value.get("metadata") or {}),
            )
        raise TypeError("Provider callable must return LLMResponse, str, or {'text': ...}")


class ProposalValidationError(ValueError):
    def __init__(self, batch_id: str, errors: list[dict[str, Any]]):
        self.batch_id = batch_id
        self.errors = errors
        super().__init__(f"Proposal batch {batch_id} failed validation")


class ProposalAdapter:
    """Strict model-to-ledger boundary.

    Model output is never treated as evidence of truth. Valid model claims are
    inserted only as provisional claims with MODEL_PROPOSAL attestations, which
    the support policy deliberately excludes from warrant. Unknown predicates,
    proof suggestions, and sensitive claim types remain quarantined until a
    human reviews the exact item payload.
    """

    MAX_RESPONSE_BYTES = 1_000_000

    def __init__(self, ledger: "EpistemicLedger") -> None:
        self.ledger = ledger
        self.validator = Draft202012Validator(PROPOSAL_SCHEMA)

    # ------------------------------------------------------------------
    # Prompt and provider boundary
    # ------------------------------------------------------------------
    def build_prompts(self, request: ProposalRequest) -> tuple[str, str]:
        allowed = list(request.allowed_predicates)
        system_prompt = (
            "You are a proposal generator for Orbita, an evidence-governed knowledge runtime. "
            "Return exactly one JSON object conforming to the supplied JSON Schema. Do not use "
            "markdown fences or prose outside JSON. You may propose claims, entities, predicates, "
            "proofs, or extractions, but proposals are not facts and must include a rationale. "
            "Prefer existing predicates. Never claim that your own output is evidence. Use explicit "
            "polarity, types, units, temporal bounds, and qualifiers when known. If information is "
            "missing, omit the proposal rather than inventing it."
        )
        user_payload = {
            "task": request.task,
            "context": request.context,
            "allowed_predicates": allowed,
            "max_proposals": request.max_proposals,
            "required_schema_version": PROPOSAL_SCHEMA_VERSION,
        }
        user_prompt = _stable_json(user_payload)
        return system_prompt, user_prompt

    def run(
        self,
        provider: LLMProvider,
        request: ProposalRequest,
        *,
        generation_parameters: dict[str, Any] | None = None,
        raise_on_invalid: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(provider, LLMProvider):
            raise TypeError("provider must implement the LLMProvider protocol")
        system_prompt, user_prompt = self.build_prompts(request)
        parameters = dict(generation_parameters or {})
        response = provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=PROPOSAL_SCHEMA,
            generation_parameters=parameters,
        )
        metadata = {"request_metadata": request.metadata, "response_metadata": response.metadata}
        return self.ingest_response(
            response.text,
            identity=provider.identity,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            generation_parameters=parameters,
            response_id=response.response_id,
            usage=response.usage,
            metadata=metadata,
            raise_on_invalid=raise_on_invalid,
        )

    # ------------------------------------------------------------------
    # Strict ingestion
    # ------------------------------------------------------------------
    def ingest_response(
        self,
        raw_response: str,
        *,
        identity: ModelIdentity,
        system_prompt: str,
        user_prompt: str,
        generation_parameters: dict[str, Any] | None = None,
        response_id: str | None = None,
        usage: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        raise_on_invalid: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(raw_response, str):
            raise TypeError("raw_response must be a string")
        errors: list[dict[str, Any]] = []
        parsed: dict[str, Any] | None = None
        if len(raw_response.encode("utf-8")) > self.MAX_RESPONSE_BYTES:
            errors.append({"kind": "size", "message": "Response exceeds 1,000,000 bytes"})
        else:
            try:
                candidate = json.loads(raw_response)
                if not isinstance(candidate, dict):
                    errors.append({"kind": "json_type", "message": "Top-level JSON must be an object"})
                else:
                    parsed = candidate
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "kind": "json_parse",
                        "message": exc.msg,
                        "line": exc.lineno,
                        "column": exc.colno,
                    }
                )

        if parsed is not None:
            for error in sorted(self.validator.iter_errors(parsed), key=lambda e: list(e.path)):
                errors.append(
                    {
                        "kind": "schema",
                        "message": error.message,
                        "path": list(error.absolute_path),
                        "schema_path": list(error.absolute_schema_path),
                    }
                )
            if not errors:
                errors.extend(self._semantic_validation_errors(parsed))

        if errors:
            batch_id = self._persist_rejected_batch(
                raw_response=raw_response,
                parsed=parsed,
                identity=identity,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                generation_parameters=generation_parameters or {},
                response_id=response_id,
                usage=usage or {},
                metadata=metadata or {},
                errors=errors,
            )
            if raise_on_invalid:
                raise ProposalValidationError(batch_id, errors)
            return self.get_batch(batch_id)

        assert parsed is not None
        batch_id = self._persist_valid_batch(
            raw_response=raw_response,
            parsed=parsed,
            identity=identity,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            generation_parameters=generation_parameters or {},
            response_id=response_id,
            usage=usage or {},
            metadata=metadata or {},
        )
        self._apply_batch(batch_id)
        return self.get_batch(batch_id)

    def _semantic_validation_errors(self, parsed: dict[str, Any]) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []
        proposals = parsed["proposals"]
        local_types: dict[str, str] = {}
        for position, item in enumerate(proposals):
            local_id = item["local_id"]
            if local_id in local_types:
                errors.append(
                    {
                        "kind": "semantic",
                        "message": f"Duplicate local_id: {local_id}",
                        "path": ["proposals", position, "local_id"],
                    }
                )
            local_types[local_id] = item["type"]

        for position, item in enumerate(proposals):
            refs: list[tuple[str, str]] = []
            if item["type"] in {"claim", "extraction"}:
                relation = item.get("relation")
                if relation:
                    subject_ref = relation["subject"].get("local_ref")
                    predicate_ref = relation["predicate"].get("local_ref")
                    if subject_ref:
                        refs.append((subject_ref, "entity"))
                    if predicate_ref:
                        refs.append((predicate_ref, "predicate"))
                    obj = relation["object"]
                    if obj["kind"] == "entity" and obj["entity"].get("local_ref"):
                        refs.append((obj["entity"]["local_ref"], "entity"))
            elif item["type"] == "proof":
                all_refs = [item["conclusion_ref"], *item["premise_refs"]]
                for ref in all_refs:
                    if ref in local_types:
                        refs.append((ref, "claim"))
                    elif not ref.startswith("clm_"):
                        errors.append(
                            {
                                "kind": "semantic",
                                "message": f"Proof reference {ref!r} is neither a local claim nor claim ID",
                                "path": ["proposals", position],
                            }
                        )
            for ref, expected in refs:
                actual = local_types.get(ref)
                if actual is None:
                    errors.append(
                        {
                            "kind": "semantic",
                            "message": f"Unknown local_ref: {ref}",
                            "path": ["proposals", position],
                        }
                    )
                elif expected == "claim" and actual not in {"claim", "extraction"}:
                    errors.append(
                        {
                            "kind": "semantic",
                            "message": f"Reference {ref} must target a claim or extraction, not {actual}",
                            "path": ["proposals", position],
                        }
                    )
                elif expected != "claim" and actual != expected:
                    errors.append(
                        {
                            "kind": "semantic",
                            "message": f"Reference {ref} must target {expected}, not {actual}",
                            "path": ["proposals", position],
                        }
                    )
        return errors

    # ------------------------------------------------------------------
    # Persistence and application
    # ------------------------------------------------------------------
    def _batch_values(
        self,
        *,
        batch_id: str,
        raw_response: str,
        parsed: dict[str, Any] | None,
        identity: ModelIdentity,
        system_prompt: str,
        user_prompt: str,
        generation_parameters: dict[str, Any],
        response_id: str | None,
        usage: dict[str, Any],
        metadata: dict[str, Any],
        status: ProposalBatchStatus,
        errors: list[dict[str, Any]],
        completed_at: str | None,
    ) -> tuple[Any, ...]:
        request_payload = {
            "provider": identity.provider,
            "model_name": identity.model_name,
            "model_version": identity.model_version,
            "system_prompt_hash": _sha256_text(system_prompt),
            "user_prompt_hash": _sha256_text(user_prompt),
            "generation_parameters": generation_parameters,
        }
        return (
            batch_id,
            str((parsed or {}).get("schema_version", PROPOSAL_SCHEMA_VERSION)),
            identity.provider,
            identity.model_name,
            identity.model_version,
            response_id,
            status.value,
            system_prompt,
            user_prompt,
            _sha256_text(system_prompt),
            _sha256_text(user_prompt),
            _sha256_text(_stable_json(request_payload)),
            raw_response,
            _sha256_text(raw_response),
            _stable_json(parsed or {}),
            _stable_json(generation_parameters),
            _stable_json(usage),
            _stable_json(metadata),
            _stable_json(errors),
            _utcnow(),
            completed_at,
        )

    def _insert_batch(self, values: tuple[Any, ...]) -> None:
        self.ledger.db.conn.execute(
            """INSERT INTO proposal_batches
               (id, schema_version, provider, model_name, model_version, response_id,
                status, system_prompt, user_prompt, system_prompt_hash, user_prompt_hash,
                request_hash, raw_response, response_hash, parsed_json,
                generation_parameters_json, usage_json, metadata_json, errors_json,
                created_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values,
        )

    def _persist_rejected_batch(self, **kwargs: Any) -> str:
        batch_id = _new_id("prb")
        completed_at = _utcnow()
        values = self._batch_values(
            batch_id=batch_id,
            status=ProposalBatchStatus.REJECTED,
            completed_at=completed_at,
            **kwargs,
        )
        self._insert_batch(values)
        self.ledger._event(
            "proposal_batch",
            batch_id,
            "PROPOSAL_BATCH_REJECTED",
            {"errors": kwargs["errors"], "response_hash": _sha256_text(kwargs["raw_response"])},
            kwargs["identity"].actor,
            ActorRole.LLM,
        )
        self.ledger.db.conn.commit()
        return batch_id

    def _persist_valid_batch(self, **kwargs: Any) -> str:
        batch_id = _new_id("prb")
        values = self._batch_values(
            batch_id=batch_id,
            status=ProposalBatchStatus.PROCESSING,
            errors=[],
            completed_at=None,
            **kwargs,
        )
        self._insert_batch(values)
        parsed = kwargs["parsed"]
        local_to_item: dict[str, str] = {}
        for position, payload in enumerate(parsed["proposals"]):
            item_id = _new_id("pri")
            local_to_item[payload["local_id"]] = item_id
            self.ledger.db.conn.execute(
                """INSERT INTO proposal_items
                   (id, batch_id, position, local_id, item_type, status, payload_json,
                    payload_hash, rationale, requires_human_review, review_reason,
                    durable_entity_type, durable_entity_id, error_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, NULL, '[]', ?)""",
                (
                    item_id,
                    batch_id,
                    position,
                    payload["local_id"],
                    payload["type"],
                    ProposalItemStatus.PENDING.value,
                    _stable_json(payload),
                    _sha256_text(_stable_json(payload)),
                    payload["rationale"],
                    _utcnow(),
                ),
            )
        for payload in parsed["proposals"]:
            item_id = local_to_item[payload["local_id"]]
            for dependency_ref, kind in self._local_dependencies(payload):
                self.ledger.db.conn.execute(
                    """INSERT OR IGNORE INTO proposal_dependencies
                       (item_id, depends_on_item_id, dependency_kind) VALUES (?, ?, ?)""",
                    (item_id, local_to_item[dependency_ref], kind),
                )
        self.ledger._event(
            "proposal_batch",
            batch_id,
            "PROPOSAL_BATCH_VALIDATED",
            {
                "schema_version": parsed["schema_version"],
                "proposal_count": len(parsed["proposals"]),
                "response_hash": _sha256_text(kwargs["raw_response"]),
            },
            kwargs["identity"].actor,
            ActorRole.LLM,
        )
        self.ledger.db.conn.commit()
        return batch_id

    @staticmethod
    def _local_dependencies(payload: dict[str, Any]) -> list[tuple[str, str]]:
        deps: list[tuple[str, str]] = []
        if payload["type"] in {"claim", "extraction"} and payload.get("relation"):
            relation = payload["relation"]
            for ref, kind in (
                (relation["subject"].get("local_ref"), "subject"),
                (relation["predicate"].get("local_ref"), "predicate"),
            ):
                if ref:
                    deps.append((ref, kind))
            obj = relation["object"]
            if obj["kind"] == "entity" and obj["entity"].get("local_ref"):
                deps.append((obj["entity"]["local_ref"], "object"))
        elif payload["type"] == "proof":
            for ref in [payload["conclusion_ref"], *payload["premise_refs"]]:
                if not ref.startswith("clm_"):
                    deps.append((ref, "proof_claim"))
        return deps

    def _apply_batch(self, batch_id: str) -> None:
        # Vocabulary first, then claims/extractions, then proof suggestions.
        for item_type in (
            ProposalItemType.ENTITY,
            ProposalItemType.PREDICATE,
            ProposalItemType.CLAIM,
            ProposalItemType.EXTRACTION,
            ProposalItemType.PROOF,
        ):
            rows = self.ledger.db.conn.execute(
                """SELECT id FROM proposal_items
                   WHERE batch_id = ? AND item_type = ? ORDER BY position""",
                (batch_id, item_type.value),
            ).fetchall()
            for row in rows:
                self._apply_item(row["id"], human_approved=False)
        self._refresh_batch_status(batch_id)

    def _apply_item(
        self,
        item_id: str,
        *,
        human_approved: bool,
        reviewer: str | None = None,
    ) -> None:
        item = self.get_item(item_id)
        if item["status"] in {ProposalItemStatus.APPLIED.value, ProposalItemStatus.REJECTED.value}:
            return
        payload = item["payload"]
        try:
            item_type = ProposalItemType(payload["type"])
            if item_type == ProposalItemType.ENTITY:
                self._apply_entity(item, payload)
            elif item_type == ProposalItemType.PREDICATE:
                self._apply_predicate(item, payload, human_approved, reviewer)
            elif item_type in {ProposalItemType.CLAIM, ProposalItemType.EXTRACTION}:
                self._apply_claim(item, payload, human_approved, reviewer)
            elif item_type == ProposalItemType.PROOF:
                self._apply_proof(item, payload, human_approved, reviewer)
        except (KeyError, ValueError, TypeError) as exc:
            self._quarantine_item(item_id, str(exc), error={"type": type(exc).__name__, "message": str(exc)})

    def _identity_for_batch(self, batch_id: str) -> ModelIdentity:
        row = self.ledger.db.conn.execute(
            "SELECT provider, model_name, model_version FROM proposal_batches WHERE id = ?",
            (batch_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown proposal batch: {batch_id}")
        return ModelIdentity(row["provider"], row["model_name"], row["model_version"])

    def _apply_entity(self, item: dict[str, Any], payload: dict[str, Any]) -> None:
        identity = self._identity_for_batch(item["batch_id"])
        try:
            entity_id = self.ledger.relations.resolve_entity(
                payload["canonical_name"], entity_type=payload["entity_type"], create=False
            )
            for alias in payload.get("aliases", []):
                self.ledger.relations.add_alias(entity_id, alias)
        except KeyError:
            metadata = dict(payload.get("metadata") or {})
            metadata.update(
                {
                    "proposal_batch_id": item["batch_id"],
                    "proposal_item_id": item["id"],
                    "origin": "llm_proposal",
                }
            )
            entity_id = self.ledger.add_entity(
                payload["canonical_name"],
                entity_type=payload["entity_type"],
                aliases=payload.get("aliases", []),
                metadata=metadata,
                actor=identity.actor,
                actor_role=ActorRole.LLM,
            )
        self._mark_applied(item["id"], "entity", entity_id)

    def _apply_predicate(
        self,
        item: dict[str, Any],
        payload: dict[str, Any],
        human_approved: bool,
        reviewer: str | None,
    ) -> None:
        try:
            predicate_id = self.ledger.relations.resolve_predicate(payload["canonical_name"])
            predicate = self.ledger.relations.get_predicate(predicate_id)
            self._validate_existing_predicate_contract(predicate, payload)
            self._mark_applied(item["id"], "predicate", predicate_id)
            return
        except KeyError:
            pass
        if not human_approved:
            self._quarantine_item(
                item["id"],
                "Unknown predicate definitions require human review before entering the ontology",
                requires_human_review=True,
            )
            return
        predicate_id = self.ledger.add_predicate(
            payload["canonical_name"],
            domain_type=payload.get("domain_type"),
            range_kind=payload["range_kind"],
            range_type=payload.get("range_type"),
            symmetric=bool(payload.get("symmetric", False)),
            metadata={
                **dict(payload.get("metadata") or {}),
                "proposal_batch_id": item["batch_id"],
                "proposal_item_id": item["id"],
                "human_approved_by": reviewer,
            },
            actor=reviewer or "reviewer",
            actor_role=ActorRole.HUMAN,
        )
        self._mark_applied(item["id"], "predicate", predicate_id, reviewer=reviewer)

    @staticmethod
    def _validate_existing_predicate_contract(
        predicate: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        proposed_domain = normalize_identifier(payload.get("domain_type") or "") or None
        proposed_range = normalize_identifier(payload.get("range_type") or "") or None
        if proposed_domain and predicate["domain_type"] not in {None, proposed_domain}:
            raise TypeError("Proposed predicate domain conflicts with the registered predicate")
        if predicate["range_kind"] != payload["range_kind"] and predicate["range_kind"] != "either":
            raise TypeError("Proposed predicate range kind conflicts with the registered predicate")
        if proposed_range and predicate["range_type"] not in {None, proposed_range}:
            raise TypeError("Proposed predicate range type conflicts with the registered predicate")

    def _apply_claim(
        self,
        item: dict[str, Any],
        payload: dict[str, Any],
        human_approved: bool,
        reviewer: str | None,
    ) -> None:
        identity = self._identity_for_batch(item["batch_id"])
        is_extraction = payload["type"] == ProposalItemType.EXTRACTION.value
        if is_extraction:
            claim_format = "relation"
            claim_type = "extraction_candidate"
        else:
            claim_format = payload["claim_format"]
            claim_type = payload.get("claim_type", "fact")

        semantic_type = claim_type
        if claim_format == "relation":
            semantic_type = payload["relation"].get("semantic_claim_type") or claim_type
        if semantic_type in self.ledger.policy.human_review_claim_types and not human_approved:
            self._quarantine_item(
                item["id"],
                f"Claim type {semantic_type!r} requires human review",
                requires_human_review=True,
            )
            return

        proposal_metadata = {
            "proposal_batch_id": item["batch_id"],
            "proposal_item_id": item["id"],
            "origin": "llm_proposal",
            "model_identity": identity.actor,
            "model_confidence": payload.get("confidence"),
            "human_approved_by": reviewer if human_approved else None,
        }
        if claim_format == "text":
            metadata = {**dict(payload.get("metadata") or {}), **proposal_metadata}
            claim_id = self.ledger.add_claim(
                payload["canonical_text"],
                claim_type=claim_type,
                scope=payload.get("scope") or {},
                metadata=metadata,
                actor=identity.actor,
                actor_role=ActorRole.LLM,
            )
        else:
            relation = payload["relation"]
            subject_id = self._resolve_entity_ref(relation["subject"], item["batch_id"], identity)
            predicate_id = self._resolve_predicate_ref(relation["predicate"], item["batch_id"])
            obj = relation["object"]
            object_value: Any
            object_kind: ObjectKind
            object_type: str | None = None
            if obj["kind"] == "entity":
                object_value = self._resolve_entity_ref(obj["entity"], item["batch_id"], identity)
                object_kind = ObjectKind.ENTITY
                object_type = self.ledger.relations.get_entity(object_value)["entity_type"]
            else:
                object_value = TypedLiteral(obj["value"], obj["datatype"], obj.get("unit"))
                object_kind = ObjectKind.LITERAL
            metadata = {**dict(relation.get("metadata") or {}), **proposal_metadata}
            if is_extraction:
                metadata.update(
                    {
                        "source_text": payload["source_text"],
                        "source_locator": payload.get("source_locator"),
                        "extraction_candidate": True,
                    }
                )
            claim_id = self.ledger.add_relation_claim(
                subject_id,
                predicate_id,
                object_value,
                object_kind=object_kind,
                object_type=object_type,
                polarity=relation.get("polarity", True),
                valid_from=relation.get("valid_from"),
                valid_to=relation.get("valid_to"),
                qualifiers=relation.get("qualifiers") or {},
                scope=relation.get("scope") or {},
                metadata=metadata,
                create_missing=False,
                actor=identity.actor,
                actor_role=ActorRole.LLM,
            )
        self._attach_model_proposal_evidence(item, payload, claim_id, identity)
        self._mark_applied(item["id"], "claim", claim_id, reviewer=reviewer)

    def _resolve_entity_ref(
        self,
        ref: dict[str, Any],
        batch_id: str,
        identity: ModelIdentity,
    ) -> str:
        if "local_ref" in ref:
            dep = self._item_by_local_id(batch_id, ref["local_ref"])
            if dep["status"] != ProposalItemStatus.APPLIED.value or dep["durable_entity_type"] != "entity":
                raise ValueError(f"Entity dependency {ref['local_ref']} is not applied")
            return dep["durable_entity_id"]
        if "entity_id" in ref:
            self.ledger.relations.get_entity(ref["entity_id"])
            return ref["entity_id"]
        name = ref["name"]
        entity_type = ref.get("entity_type")
        try:
            return self.ledger.relations.resolve_entity(name, entity_type=entity_type, create=False)
        except KeyError:
            if not entity_type:
                raise ValueError(
                    f"Unknown entity {name!r} lacks entity_type; cannot create it unambiguously"
                )
            return self.ledger.add_entity(
                name,
                entity_type=entity_type,
                metadata={"origin": "llm_proposal", "proposal_batch_id": batch_id},
                actor=identity.actor,
                actor_role=ActorRole.LLM,
            )

    def _resolve_predicate_ref(self, ref: dict[str, Any], batch_id: str) -> str:
        if "local_ref" in ref:
            dep = self._item_by_local_id(batch_id, ref["local_ref"])
            if dep["status"] != ProposalItemStatus.APPLIED.value or dep["durable_entity_type"] != "predicate":
                raise ValueError(f"Predicate dependency {ref['local_ref']} is not approved")
            return dep["durable_entity_id"]
        if "predicate_id" in ref:
            self.ledger.relations.get_predicate(ref["predicate_id"])
            return ref["predicate_id"]
        try:
            return self.ledger.relations.resolve_predicate(ref["name"])
        except KeyError as exc:
            raise ValueError(
                f"Unknown predicate {ref['name']!r}; propose it separately for human ontology review"
            ) from exc

    def _attach_model_proposal_evidence(
        self,
        item: dict[str, Any],
        payload: dict[str, Any],
        claim_id: str,
        identity: ModelIdentity,
    ) -> None:
        excerpt = payload.get("source_text") or payload["rationale"]
        evidence_id = self.ledger.add_evidence(
            f"model://{identity.provider}/{identity.model_name}/{item['batch_id']}/{item['id']}",
            excerpt,
            source_kind=EvidenceKind.MODEL_PROPOSAL,
            independence_key=f"model_proposal:{item['batch_id']}:{item['id']}",
            content=_stable_json(payload),
            metadata={
                "proposal_batch_id": item["batch_id"],
                "proposal_item_id": item["id"],
                "model_provider": identity.provider,
                "model_name": identity.model_name,
                "model_version": identity.model_version,
            },
            actor=identity.actor,
            actor_role=ActorRole.LLM,
        )
        self.ledger.attest(
            claim_id,
            evidence_id,
            Stance.SUPPORT,
            confidence=float(payload.get("confidence", 0.5)),
            actor=identity.actor,
            actor_role=ActorRole.LLM,
        )

    def _apply_proof(
        self,
        item: dict[str, Any],
        payload: dict[str, Any],
        human_approved: bool,
        reviewer: str | None,
    ) -> None:
        if not human_approved:
            self._quarantine_item(
                item["id"],
                "Model-suggested proof edges require human or formal-verifier review",
                requires_human_review=True,
            )
            return
        conclusion = self._resolve_claim_ref(item["batch_id"], payload["conclusion_ref"])
        premises = [self._resolve_claim_ref(item["batch_id"], ref) for ref in payload["premise_refs"]]
        proof_id = self.ledger.add_proof(
            conclusion,
            premises,
            rule=payload["rule"],
            metadata={
                **dict(payload.get("metadata") or {}),
                "proposal_batch_id": item["batch_id"],
                "proposal_item_id": item["id"],
                "human_approved_by": reviewer,
            },
            actor=reviewer or "reviewer",
            actor_role=ActorRole.HUMAN,
        )
        self._mark_applied(item["id"], "proof", proof_id, reviewer=reviewer)

    def _resolve_claim_ref(self, batch_id: str, ref: str) -> str:
        if ref.startswith("clm_"):
            self.ledger.get_claim(ref)
            return ref
        item = self._item_by_local_id(batch_id, ref)
        if item["status"] != ProposalItemStatus.APPLIED.value or item["durable_entity_type"] != "claim":
            raise ValueError(f"Claim dependency {ref} is not applied")
        return item["durable_entity_id"]

    def _item_by_local_id(self, batch_id: str, local_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT id FROM proposal_items WHERE batch_id = ? AND local_id = ?",
            (batch_id, local_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown local proposal reference: {local_id}")
        return self.get_item(row["id"])

    def _mark_applied(
        self,
        item_id: str,
        durable_type: str,
        durable_id: str,
        *,
        reviewer: str | None = None,
    ) -> None:
        self.ledger.db.conn.execute(
            """UPDATE proposal_items
               SET status = ?, durable_entity_type = ?, durable_entity_id = ?,
                   requires_human_review = 0, review_reason = NULL, error_json = '[]',
                   reviewed_at = COALESCE(reviewed_at, ?), reviewed_by = COALESCE(reviewed_by, ?)
               WHERE id = ?""",
            (
                ProposalItemStatus.APPLIED.value,
                durable_type,
                durable_id,
                _utcnow() if reviewer else None,
                reviewer,
                item_id,
            ),
        )
        self.ledger._event(
            "proposal_item",
            item_id,
            "PROPOSAL_ITEM_APPLIED",
            {"durable_entity_type": durable_type, "durable_entity_id": durable_id},
            reviewer or "proposal_policy",
            ActorRole.HUMAN if reviewer else ActorRole.POLICY,
        )
        self.ledger.db.conn.commit()

    def _quarantine_item(
        self,
        item_id: str,
        reason: str,
        *,
        requires_human_review: bool = False,
        error: dict[str, Any] | None = None,
    ) -> None:
        self.ledger.db.conn.execute(
            """UPDATE proposal_items
               SET status = ?, requires_human_review = ?, review_reason = ?, error_json = ?
               WHERE id = ?""",
            (
                ProposalItemStatus.QUARANTINED.value,
                int(requires_human_review),
                reason,
                _stable_json([error] if error else []),
                item_id,
            ),
        )
        self.ledger._event(
            "proposal_item",
            item_id,
            "PROPOSAL_ITEM_QUARANTINED",
            {"reason": reason, "requires_human_review": requires_human_review},
            "proposal_policy",
            ActorRole.POLICY,
        )
        self.ledger.db.conn.commit()

    def _refresh_batch_status(self, batch_id: str) -> None:
        rows = self.ledger.db.conn.execute(
            "SELECT status, COUNT(*) AS n FROM proposal_items WHERE batch_id = ? GROUP BY status",
            (batch_id,),
        ).fetchall()
        counts = {row["status"]: row["n"] for row in rows}
        if counts.get(ProposalItemStatus.QUARANTINED.value, 0) or counts.get(
            ProposalItemStatus.PENDING.value, 0
        ):
            status = ProposalBatchStatus.NEEDS_REVIEW
        elif counts.get(ProposalItemStatus.REJECTED.value, 0):
            status = ProposalBatchStatus.NEEDS_REVIEW
        else:
            status = ProposalBatchStatus.APPLIED
        self.ledger.db.conn.execute(
            "UPDATE proposal_batches SET status = ?, completed_at = ? WHERE id = ?",
            (status.value, _utcnow(), batch_id),
        )
        self.ledger._event(
            "proposal_batch",
            batch_id,
            "PROPOSAL_BATCH_COMPLETED",
            {"status": status.value, "item_counts": counts},
            "proposal_policy",
            ActorRole.POLICY,
        )
        self.ledger.db.conn.commit()

    # ------------------------------------------------------------------
    # Human review
    # ------------------------------------------------------------------
    def review_item(
        self,
        item_id: str,
        decision: ReviewDecision | str,
        *,
        reviewer: str,
        rationale: str,
    ) -> dict[str, Any]:
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        if not rationale.strip():
            raise ValueError("rationale is required")
        decision = ReviewDecision(decision)
        item = self.get_item(item_id)
        if item["status"] == ProposalItemStatus.APPLIED.value:
            raise ValueError("Applied proposal items cannot be reviewed again")
        if item["status"] == ProposalItemStatus.REJECTED.value:
            raise ValueError("Rejected proposal items cannot be reviewed again")
        if decision == ReviewDecision.REJECT:
            self.ledger.db.conn.execute(
                """UPDATE proposal_items SET status = ?, review_reason = ?, reviewed_at = ?,
                   reviewed_by = ?, requires_human_review = 0 WHERE id = ?""",
                (
                    ProposalItemStatus.REJECTED.value,
                    rationale,
                    _utcnow(),
                    reviewer,
                    item_id,
                ),
            )
            self.ledger._event(
                "proposal_item",
                item_id,
                "PROPOSAL_ITEM_REJECTED",
                {"rationale": rationale},
                reviewer,
                ActorRole.HUMAN,
            )
            self.ledger.db.conn.commit()
        else:
            self.ledger._event(
                "proposal_item",
                item_id,
                "PROPOSAL_ITEM_APPROVED",
                {"rationale": rationale, "payload_hash": item["payload_hash"]},
                reviewer,
                ActorRole.HUMAN,
            )
            self.ledger.db.conn.commit()
            self._apply_item(item_id, human_approved=True, reviewer=reviewer)
        self._refresh_batch_status(item["batch_id"])
        return self.get_item(item_id)

    def retry_ready_items(self, batch_id: str) -> dict[str, Any]:
        self.get_batch(batch_id)
        rows = self.ledger.db.conn.execute(
            """SELECT id FROM proposal_items
               WHERE batch_id = ? AND status = ? ORDER BY position""",
            (batch_id, ProposalItemStatus.QUARANTINED.value),
        ).fetchall()
        for row in rows:
            item = self.get_item(row["id"])
            if not item["requires_human_review"]:
                self._apply_item(row["id"], human_approved=False)
        self._refresh_batch_status(batch_id)
        return self.get_batch(batch_id)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    def get_item(self, item_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM proposal_items WHERE id = ?", (item_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown proposal item: {item_id}")
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        item["errors"] = json.loads(item.pop("error_json"))
        item["requires_human_review"] = bool(item["requires_human_review"])
        item["dependencies"] = [
            dict(dep)
            for dep in self.ledger.db.conn.execute(
                """SELECT d.depends_on_item_id, d.dependency_kind,
                          p.local_id AS depends_on_local_id, p.status AS depends_on_status
                   FROM proposal_dependencies d
                   JOIN proposal_items p ON p.id = d.depends_on_item_id
                   WHERE d.item_id = ? ORDER BY p.position""",
                (item_id,),
            ).fetchall()
        ]
        return item

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM proposal_batches WHERE id = ?", (batch_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown proposal batch: {batch_id}")
        batch = dict(row)
        for source, target in (
            ("parsed_json", "parsed"),
            ("generation_parameters_json", "generation_parameters"),
            ("usage_json", "usage"),
            ("metadata_json", "metadata"),
            ("errors_json", "errors"),
        ):
            batch[target] = json.loads(batch.pop(source))
        item_rows = self.ledger.db.conn.execute(
            "SELECT id FROM proposal_items WHERE batch_id = ? ORDER BY position", (batch_id,)
        ).fetchall()
        batch["items"] = [self.get_item(item["id"]) for item in item_rows]
        return batch

    def list_batches(self, *, status: str | None = None) -> list[dict[str, Any]]:
        if status is None:
            rows = self.ledger.db.conn.execute(
                "SELECT id FROM proposal_batches ORDER BY created_at"
            ).fetchall()
        else:
            ProposalBatchStatus(status)
            rows = self.ledger.db.conn.execute(
                "SELECT id FROM proposal_batches WHERE status = ? ORDER BY created_at",
                (status,),
            ).fetchall()
        return [self.get_batch(row["id"]) for row in rows]
