from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable

from .models import (
    ActorRole,
    ClaimStatus,
    LiteralDatatype,
    ObjectKind,
    PredicateRangeKind,
    TypedLiteral,
)

if TYPE_CHECKING:  # pragma: no cover
    from .ledger import EpistemicLedger


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip()
    return " ".join(value.split()).casefold()


def normalize_identifier(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().casefold()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _new_id(prefix: str) -> str:
    from .ledger import new_id

    return new_id(prefix)


def _stable_json(value: Any) -> str:
    from .ledger import stable_json

    return stable_json(value)


def _utcnow() -> str:
    from .ledger import utcnow

    return utcnow()


def normalize_temporal(value: str | date | datetime | None) -> str | None:
    """Normalize temporal boundaries to comparable UTC ISO-8601 strings.

    Naive datetimes are treated as UTC. Date-only values denote midnight UTC.
    Intervals are inclusive at both boundaries.
    """

    if value is None:
        return None
    parsed: datetime
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if "T" not in text and " " not in text:
                d = date.fromisoformat(text)
                parsed = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
            else:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid ISO-8601 temporal value: {value!r}") from exc
    else:
        raise TypeError(f"Unsupported temporal value: {type(value).__name__}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def intervals_overlap(
    start_a: str | None,
    end_a: str | None,
    start_b: str | None,
    end_b: str | None,
) -> bool:
    if end_a is not None and start_b is not None and end_a < start_b:
        return False
    if end_b is not None and start_a is not None and end_b < start_a:
        return False
    return True


def _canonical_literal(literal: TypedLiteral) -> tuple[str, str, str | None, Any]:
    datatype = literal.resolved_datatype()
    value = literal.value
    if datatype == LiteralDatatype.INTEGER:
        if isinstance(value, bool):
            raise TypeError("Boolean values cannot be stored as integer literals")
        value = int(value)
    elif datatype == LiteralDatatype.FLOAT:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Float literals must be finite")
    elif datatype == LiteralDatatype.BOOLEAN:
        if isinstance(value, str):
            lowered = value.strip().casefold()
            if lowered not in {"true", "false"}:
                raise ValueError("Boolean string literals must be 'true' or 'false'")
            value = lowered == "true"
        else:
            value = bool(value)
    elif datatype == LiteralDatatype.DATE:
        if isinstance(value, datetime):
            value = value.date().isoformat()
        elif isinstance(value, date):
            value = value.isoformat()
        else:
            value = date.fromisoformat(str(value)).isoformat()
    elif datatype == LiteralDatatype.DATETIME:
        value = normalize_temporal(value)
    elif datatype in {LiteralDatatype.STRING, LiteralDatatype.URI}:
        value = str(value)
    elif datatype == LiteralDatatype.JSON:
        json.dumps(value, allow_nan=False)
    unit = literal.unit.strip() if literal.unit else None
    return _stable_json(value), datatype.value, unit, value


def _render_literal(value: Any, datatype: str, unit: str | None) -> str:
    if datatype == LiteralDatatype.STRING.value:
        rendered = json.dumps(value, ensure_ascii=False)
    else:
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
    return f"{rendered} {unit}" if unit else rendered


class RelationStore:
    """Typed subject-predicate-object claims over canonical entity identities.

    The relation store deliberately sits on top of the same claim ledger. A
    structured claim can receive evidence, participate in proof hyperedges,
    collapse when premises fail, and gate actions exactly like a text claim.
    """

    def __init__(self, ledger: "EpistemicLedger"):
        self.ledger = ledger

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------
    def add_entity(
        self,
        canonical_name: str,
        *,
        entity_type: str = "thing",
        aliases: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
        actor: str = "user",
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> str:
        if not canonical_name.strip():
            raise ValueError("canonical_name cannot be empty")
        entity_type = normalize_identifier(entity_type) or "thing"
        alias_list = list(aliases or [])
        normalized = normalize_name(canonical_name)
        row = self.ledger.db.conn.execute(
            "SELECT id FROM entities WHERE normalized_name = ? AND entity_type = ?",
            (normalized, entity_type),
        ).fetchone()
        if row is not None:
            entity_id = row["id"]
            for alias in alias_list:
                self.add_alias(entity_id, alias)
            return entity_id

        entity_id = _new_id("ent")
        now = _utcnow()
        self.ledger.db.conn.execute(
            """INSERT INTO entities
               (id, canonical_name, normalized_name, entity_type, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                entity_id,
                " ".join(canonical_name.strip().split()),
                normalized,
                entity_type,
                _stable_json(metadata or {}),
                now,
            ),
        )
        self._insert_alias(entity_id, canonical_name, entity_type, now)
        for alias in alias_list:
            self._insert_alias(entity_id, alias, entity_type, now)
        self.ledger._event(
            "entity",
            entity_id,
            "ENTITY_CREATED",
            {
                "canonical_name": canonical_name,
                "entity_type": entity_type,
                "aliases": alias_list,
            },
            actor,
            actor_role,
        )
        self.ledger.db.conn.commit()
        return entity_id

    def add_alias(self, entity_id: str, alias: str) -> None:
        entity = self.get_entity(entity_id)
        self._insert_alias(entity_id, alias, entity["entity_type"], _utcnow())
        self.ledger._event(
            "entity",
            entity_id,
            "ENTITY_ALIAS_ADDED",
            {"alias": alias},
            "user",
            ActorRole.HUMAN,
        )
        self.ledger.db.conn.commit()

    def _insert_alias(self, entity_id: str, alias: str, entity_type: str, created_at: str) -> None:
        if not alias.strip():
            raise ValueError("alias cannot be empty")
        normalized = normalize_name(alias)
        existing = self.ledger.db.conn.execute(
            """SELECT entity_id FROM entity_aliases
               WHERE alias_normalized = ? AND entity_type = ?""",
            (normalized, entity_type),
        ).fetchone()
        if existing is not None and existing["entity_id"] != entity_id:
            raise ValueError(
                f"Alias {alias!r} already resolves to another {entity_type} entity"
            )
        self.ledger.db.conn.execute(
            """INSERT OR IGNORE INTO entity_aliases
               (alias_normalized, alias_text, entity_type, entity_id, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (normalized, alias.strip(), entity_type, entity_id, created_at),
        )

    def get_entity(self, entity_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown entity: {entity_id}")
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        result["aliases"] = [
            item["alias_text"]
            for item in self.ledger.db.conn.execute(
                "SELECT alias_text FROM entity_aliases WHERE entity_id = ? ORDER BY alias_text",
                (entity_id,),
            ).fetchall()
        ]
        return result

    def resolve_entity(
        self,
        reference: str,
        *,
        entity_type: str | None = None,
        create: bool = False,
        aliases: Iterable[str] | None = None,
    ) -> str:
        direct = self.ledger.db.conn.execute(
            "SELECT id FROM entities WHERE id = ?", (reference,)
        ).fetchone()
        if direct is not None:
            entity = self.get_entity(reference)
            if entity_type and entity["entity_type"] != normalize_identifier(entity_type):
                raise TypeError(
                    f"Entity {reference} has type {entity['entity_type']}, expected {entity_type}"
                )
            return reference

        normalized = normalize_name(reference)
        params: list[Any] = [normalized]
        sql = "SELECT entity_id, entity_type FROM entity_aliases WHERE alias_normalized = ?"
        if entity_type:
            sql += " AND entity_type = ?"
            params.append(normalize_identifier(entity_type))
        rows = self.ledger.db.conn.execute(sql, params).fetchall()
        unique = {row["entity_id"] for row in rows}
        if len(unique) == 1:
            return next(iter(unique))
        if len(unique) > 1:
            raise ValueError(f"Ambiguous entity reference: {reference!r}; provide entity_type")
        if create:
            return self.add_entity(
                reference,
                entity_type=entity_type or "thing",
                aliases=aliases,
            )
        raise KeyError(f"Unknown entity: {reference}")

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------
    def add_predicate(
        self,
        canonical_name: str,
        *,
        domain_type: str | None = None,
        range_kind: PredicateRangeKind | str = PredicateRangeKind.EITHER,
        range_type: str | None = None,
        inverse_predicate: str | None = None,
        symmetric: bool = False,
        metadata: dict[str, Any] | None = None,
        actor: str = "user",
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> str:
        normalized = normalize_identifier(canonical_name)
        if not normalized:
            raise ValueError("predicate name cannot be empty")
        range_kind = PredicateRangeKind(range_kind)
        domain = normalize_identifier(domain_type) if domain_type else None
        range_type_norm = normalize_identifier(range_type) if range_type else None
        existing = self.ledger.db.conn.execute(
            "SELECT * FROM predicates WHERE normalized_name = ?", (normalized,)
        ).fetchone()
        if existing is not None:
            if domain and existing["domain_type"] not in {None, domain}:
                raise TypeError(
                    f"Predicate {normalized} domain is {existing['domain_type']}, not {domain}"
                )
            if existing["range_kind"] != range_kind.value:
                if existing["range_kind"] != PredicateRangeKind.EITHER.value:
                    raise TypeError(
                        f"Predicate {normalized} range kind is {existing['range_kind']}, not {range_kind.value}"
                    )
            if range_type_norm and existing["range_type"] not in {None, range_type_norm}:
                raise TypeError(
                    f"Predicate {normalized} range type is {existing['range_type']}, not {range_type_norm}"
                )
            return existing["id"]

        inverse_id = None
        if inverse_predicate:
            inverse_id = self.resolve_predicate(inverse_predicate)
        predicate_id = _new_id("prd")
        self.ledger.db.conn.execute(
            """INSERT INTO predicates
               (id, canonical_name, normalized_name, domain_type, range_kind,
                range_type, inverse_predicate_id, symmetric, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                predicate_id,
                canonical_name.strip(),
                normalized,
                domain,
                range_kind.value,
                range_type_norm,
                inverse_id,
                int(symmetric),
                _stable_json(metadata or {}),
                _utcnow(),
            ),
        )
        self.ledger._event(
            "predicate",
            predicate_id,
            "PREDICATE_CREATED",
            {
                "canonical_name": canonical_name,
                "normalized_name": normalized,
                "domain_type": domain,
                "range_kind": range_kind.value,
                "range_type": range_type_norm,
                "symmetric": symmetric,
            },
            actor,
            actor_role,
        )
        self.ledger.db.conn.commit()
        return predicate_id

    def resolve_predicate(self, reference: str) -> str:
        row = self.ledger.db.conn.execute(
            "SELECT id FROM predicates WHERE id = ?", (reference,)
        ).fetchone()
        if row is not None:
            return row["id"]
        normalized = normalize_identifier(reference)
        row = self.ledger.db.conn.execute(
            "SELECT id FROM predicates WHERE normalized_name = ?", (normalized,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown predicate: {reference}")
        return row["id"]

    def get_predicate(self, predicate_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM predicates WHERE id = ?", (predicate_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown predicate: {predicate_id}")
        result = dict(row)
        result["symmetric"] = bool(result["symmetric"])
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    # ------------------------------------------------------------------
    # Structured claims
    # ------------------------------------------------------------------
    def add_relation_claim(
        self,
        subject: str,
        predicate: str,
        object_value: str | TypedLiteral | Any,
        *,
        subject_type: str | None = None,
        object_kind: ObjectKind | str | None = None,
        object_type: str | None = None,
        predicate_range_kind: PredicateRangeKind | str | None = None,
        polarity: bool = True,
        valid_from: str | date | datetime | None = None,
        valid_to: str | date | datetime | None = None,
        qualifiers: dict[str, Any] | None = None,
        scope: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        create_missing: bool = True,
        actor: str = "user",
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> str:
        subject_id = self.resolve_entity(
            subject, entity_type=subject_type, create=create_missing
        )
        actual_subject_type = self.get_entity(subject_id)["entity_type"]

        predicate_id: str | None
        predicate_row: dict[str, Any] | None
        try:
            predicate_id = self.resolve_predicate(predicate)
            predicate_row = self.get_predicate(predicate_id)
        except KeyError:
            if not create_missing:
                raise
            predicate_id = None
            predicate_row = None

        kind = self._infer_object_kind(object_value, object_kind, predicate_row)
        object_entity_id: str | None = None
        literal_json: str | None = None
        literal_datatype: str | None = None
        literal_unit: str | None = None
        literal_value: Any = None
        if kind == ObjectKind.ENTITY:
            if not isinstance(object_value, str):
                raise TypeError("Entity objects must be referenced by name or ID")
            object_entity_id = self.resolve_entity(
                object_value,
                entity_type=object_type,
                create=create_missing,
            )
            actual_object_type = self.get_entity(object_entity_id)["entity_type"]
        else:
            literal = self._as_literal(object_value)
            literal_json, literal_datatype, literal_unit, literal_value = _canonical_literal(literal)
            actual_object_type = literal_datatype

        if predicate_id is None:
            range_kind = PredicateRangeKind(predicate_range_kind or kind.value)
            predicate_id = self.add_predicate(
                predicate,
                domain_type=actual_subject_type,
                range_kind=range_kind,
                range_type=actual_object_type,
                actor=actor,
                actor_role=actor_role,
            )
            predicate_row = self.get_predicate(predicate_id)

        assert predicate_row is not None
        self._validate_relation_types(
            subject_id,
            predicate_row,
            kind,
            object_entity_id,
            literal_datatype,
        )

        start = normalize_temporal(valid_from)
        end = normalize_temporal(valid_to)
        if start is not None and end is not None and start > end:
            raise ValueError("valid_from must be earlier than or equal to valid_to")
        qualifiers = qualifiers or {}
        qualifiers_json = _stable_json(qualifiers)
        object_token = (
            {"kind": "entity", "id": object_entity_id}
            if kind == ObjectKind.ENTITY
            else {
                "kind": "literal",
                "value": json.loads(literal_json or "null"),
                "datatype": literal_datatype,
                "unit": literal_unit,
            }
        )
        key_payload = {
            "subject": subject_id,
            "predicate": predicate_id,
            "object": object_token,
            "polarity": bool(polarity),
            "valid_from": start,
            "valid_to": end,
            "qualifiers": qualifiers,
        }
        canonical_key = hashlib.sha256(_stable_json(key_payload).encode("utf-8")).hexdigest()
        existing = self.ledger.db.conn.execute(
            "SELECT claim_id FROM relation_claims WHERE canonical_key = ?", (canonical_key,)
        ).fetchone()
        if existing is not None:
            return existing["claim_id"]

        subject_row = self.get_entity(subject_id)
        if kind == ObjectKind.ENTITY:
            object_display = self.get_entity(object_entity_id or "")["canonical_name"]
        else:
            object_display = _render_literal(
                literal_value, literal_datatype or LiteralDatatype.STRING.value, literal_unit
            )
        atom = (
            f"{subject_row['canonical_name']} "
            f"{predicate_row['normalized_name']} {object_display}"
        )
        canonical_text = atom if polarity else f"NOT ({atom})"

        claim_id = _new_id("clm")
        now = _utcnow()
        self.ledger.db.conn.execute(
            """INSERT INTO claims
               (id, canonical_text, claim_type, status, scope_json, metadata_json, created_at, updated_at)
               VALUES (?, ?, 'relation', ?, ?, ?, ?, ?)""",
            (
                claim_id,
                canonical_text,
                ClaimStatus.PROVISIONAL.value,
                _stable_json(scope or {}),
                _stable_json(metadata or {}),
                now,
                now,
            ),
        )
        self.ledger.db.conn.execute(
            """INSERT INTO relation_claims
               (claim_id, subject_entity_id, predicate_id, object_kind,
                object_entity_id, literal_json, literal_datatype, literal_unit,
                polarity, valid_from, valid_to, qualifiers_json, canonical_key, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                claim_id,
                subject_id,
                predicate_id,
                kind.value,
                object_entity_id,
                literal_json,
                literal_datatype,
                literal_unit,
                int(bool(polarity)),
                start,
                end,
                qualifiers_json,
                canonical_key,
                now,
            ),
        )
        self.ledger._event(
            "claim",
            claim_id,
            "RELATION_CLAIM_PROPOSED",
            {
                "subject_entity_id": subject_id,
                "predicate_id": predicate_id,
                "object": object_token,
                "polarity": bool(polarity),
                "valid_from": start,
                "valid_to": end,
                "qualifiers": qualifiers,
                "canonical_key": canonical_key,
            },
            actor,
            actor_role,
        )
        self._link_exact_negations(
            claim_id,
            subject_id,
            predicate_id,
            kind,
            object_entity_id,
            literal_json,
            literal_datatype,
            literal_unit,
            bool(polarity),
            start,
            end,
            qualifiers_json,
        )
        self.ledger.db.conn.commit()
        return claim_id

    def _infer_object_kind(
        self,
        object_value: Any,
        requested: ObjectKind | str | None,
        predicate: dict[str, Any] | None = None,
    ) -> ObjectKind:
        if requested is not None:
            return ObjectKind(requested)
        if isinstance(object_value, TypedLiteral) or not isinstance(object_value, str):
            return ObjectKind.LITERAL
        if predicate and predicate["range_kind"] == PredicateRangeKind.LITERAL.value:
            return ObjectKind.LITERAL
        return ObjectKind.ENTITY

    @staticmethod
    def _as_literal(value: Any) -> TypedLiteral:
        return value if isinstance(value, TypedLiteral) else TypedLiteral(value)

    def _validate_relation_types(
        self,
        subject_id: str,
        predicate: dict[str, Any],
        object_kind: ObjectKind,
        object_entity_id: str | None,
        literal_datatype: str | None,
    ) -> None:
        subject_type = self.get_entity(subject_id)["entity_type"]
        expected_domain = predicate["domain_type"]
        if expected_domain and expected_domain != "thing" and subject_type != expected_domain:
            raise TypeError(
                f"Predicate {predicate['normalized_name']} expects subject type "
                f"{expected_domain}, got {subject_type}"
            )
        range_kind = PredicateRangeKind(predicate["range_kind"])
        if range_kind != PredicateRangeKind.EITHER and range_kind.value != object_kind.value:
            raise TypeError(
                f"Predicate {predicate['normalized_name']} expects {range_kind.value} objects"
            )
        expected_range = predicate["range_type"]
        if not expected_range:
            return
        if object_kind == ObjectKind.ENTITY:
            actual = self.get_entity(object_entity_id or "")["entity_type"]
            if expected_range != "thing" and actual != expected_range:
                raise TypeError(
                    f"Predicate {predicate['normalized_name']} expects object type "
                    f"{expected_range}, got {actual}"
                )
        elif literal_datatype != expected_range:
            raise TypeError(
                f"Predicate {predicate['normalized_name']} expects literal datatype "
                f"{expected_range}, got {literal_datatype}"
            )

    def _link_exact_negations(
        self,
        claim_id: str,
        subject_id: str,
        predicate_id: str,
        object_kind: ObjectKind,
        object_entity_id: str | None,
        literal_json: str | None,
        literal_datatype: str | None,
        literal_unit: str | None,
        polarity: bool,
        valid_from: str | None,
        valid_to: str | None,
        qualifiers_json: str,
    ) -> None:
        rows = self.ledger.db.conn.execute(
            """SELECT * FROM relation_claims
               WHERE subject_entity_id = ? AND predicate_id = ?
                 AND object_kind = ? AND polarity = ?
                 AND qualifiers_json = ? AND claim_id != ?""",
            (
                subject_id,
                predicate_id,
                object_kind.value,
                int(not polarity),
                qualifiers_json,
                claim_id,
            ),
        ).fetchall()
        for row in rows:
            same_object = (
                row["object_entity_id"] == object_entity_id
                if object_kind == ObjectKind.ENTITY
                else row["literal_json"] == literal_json
                and row["literal_datatype"] == literal_datatype
                and row["literal_unit"] == literal_unit
            )
            if not same_object or not intervals_overlap(
                valid_from, valid_to, row["valid_from"], row["valid_to"]
            ):
                continue
            existing = self.ledger.db.conn.execute(
                """SELECT 1 FROM contradictions
                   WHERE active = 1 AND ((claim_a = ? AND claim_b = ?) OR
                                         (claim_a = ? AND claim_b = ?))""",
                (claim_id, row["claim_id"], row["claim_id"], claim_id),
            ).fetchone()
            if existing is None:
                self.ledger.add_contradiction(
                    claim_id,
                    row["claim_id"],
                    rationale="Exact structured negation over an overlapping validity interval",
                    actor="relation_policy",
                    actor_role=ActorRole.POLICY,
                )

    def get_relation_claim(self, claim_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            """SELECT rc.*, c.canonical_text, c.claim_type, c.status, c.scope_json, c.metadata_json,
                      s.canonical_name AS subject_name, s.entity_type AS subject_type,
                      p.canonical_name AS predicate_name, p.normalized_name AS predicate,
                      o.canonical_name AS object_name, o.entity_type AS object_type
               FROM relation_claims rc
               JOIN claims c ON c.id = rc.claim_id
               JOIN entities s ON s.id = rc.subject_entity_id
               JOIN predicates p ON p.id = rc.predicate_id
               LEFT JOIN entities o ON o.id = rc.object_entity_id
               WHERE rc.claim_id = ?""",
            (claim_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Claim {claim_id} is not a structured relation claim")
        result = dict(row)
        result["polarity"] = bool(result["polarity"])
        result["qualifiers"] = json.loads(result.pop("qualifiers_json"))
        result["scope"] = json.loads(result.pop("scope_json"))
        result["metadata"] = json.loads(result.pop("metadata_json"))
        if result["object_kind"] == ObjectKind.ENTITY.value:
            result["object"] = {
                "kind": "entity",
                "id": result["object_entity_id"],
                "name": result["object_name"],
                "entity_type": result["object_type"],
            }
        else:
            result["object"] = {
                "kind": "literal",
                "value": json.loads(result["literal_json"]),
                "datatype": result["literal_datatype"],
                "unit": result["literal_unit"],
            }
        return result

    def list_relation_claims(self) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT claim_id FROM relation_claims ORDER BY created_at"
        ).fetchall()
        return [self.get_relation_claim(row["claim_id"]) for row in rows]

    def find_relation_claims(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object_value: str | TypedLiteral | Any | None = None,
        object_kind: ObjectKind | str | None = None,
        subject_type: str | None = None,
        object_type: str | None = None,
        polarity: bool | None = None,
        valid_at: str | date | datetime | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if subject is not None:
            clauses.append("rc.subject_entity_id = ?")
            params.append(self.resolve_entity(subject, entity_type=subject_type))
        predicate_row: dict[str, Any] | None = None
        if predicate is not None:
            predicate_id = self.resolve_predicate(predicate)
            predicate_row = self.get_predicate(predicate_id)
            clauses.append("rc.predicate_id = ?")
            params.append(predicate_id)
        if polarity is not None:
            clauses.append("rc.polarity = ?")
            params.append(int(polarity))
        if object_value is not None:
            kind = self._infer_object_kind(object_value, object_kind, predicate_row)
            clauses.append("rc.object_kind = ?")
            params.append(kind.value)
            if kind == ObjectKind.ENTITY:
                clauses.append("rc.object_entity_id = ?")
                params.append(self.resolve_entity(str(object_value), entity_type=object_type))
            else:
                literal_json, datatype, unit, _ = _canonical_literal(self._as_literal(object_value))
                clauses.extend(
                    [
                        "rc.literal_json = ?",
                        "rc.literal_datatype = ?",
                        "rc.literal_unit IS ?",
                    ]
                )
                params.extend([literal_json, datatype, unit])
        sql = "SELECT rc.claim_id FROM relation_claims rc"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY rc.created_at"
        rows = self.ledger.db.conn.execute(sql, params).fetchall()
        results = [self.get_relation_claim(row["claim_id"]) for row in rows]
        point = normalize_temporal(valid_at)
        if point is not None:
            results = [
                item
                for item in results
                if (item["valid_from"] is None or item["valid_from"] <= point)
                and (item["valid_to"] is None or item["valid_to"] >= point)
            ]
        return results

    def negate_relation_claim(
        self,
        claim_id: str,
        *,
        actor: str = "user",
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> str:
        relation = self.get_relation_claim(claim_id)
        object_value: Any
        kind = ObjectKind(relation["object_kind"])
        if kind == ObjectKind.ENTITY:
            object_value = relation["object"]["id"]
            object_type = relation["object"]["entity_type"]
        else:
            object_value = TypedLiteral(
                relation["object"]["value"],
                relation["object"]["datatype"],
                relation["object"]["unit"],
            )
            object_type = "thing"
        return self.add_relation_claim(
            relation["subject_entity_id"],
            relation["predicate_id"],
            object_value,
            subject_type=relation["subject_type"],
            object_kind=kind,
            object_type=object_type,
            polarity=not relation["polarity"],
            valid_from=relation["valid_from"],
            valid_to=relation["valid_to"],
            qualifiers=relation["qualifiers"],
            create_missing=False,
            actor=actor,
            actor_role=actor_role,
        )
