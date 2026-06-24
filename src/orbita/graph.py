from __future__ import annotations

import hashlib
import html
import json
import shutil
import subprocess
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from .ledger import EpistemicLedger


GRAPH_SCHEMA_VERSION = "1.0"
DIFF_SCHEMA_VERSION = "1.0"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _json_load(value: str | None, default: Any) -> Any:
    if value in (None, ""):
        return default
    return json.loads(value)


def _edge_id(source: str, target: str, kind: str, discriminator: Any = None) -> str:
    payload = {"source": source, "target": target, "kind": kind, "d": discriminator}
    return f"edg_{_hash_json(payload)[:20]}"


def _node_signature(node: dict[str, Any]) -> dict[str, Any]:
    """Return only fields that represent the node's epistemic state."""
    keep = {
        "kind",
        "status",
        "support_state",
        "active",
        "outcome",
        "review_status",
        "label",
        "metadata",
    }
    return {key: node.get(key) for key in sorted(keep) if key in node}


def _edge_signature(edge: dict[str, Any]) -> dict[str, Any]:
    keep = {"source", "target", "kind", "active", "label", "metadata"}
    return {key: edge.get(key) for key in sorted(keep) if key in edge}


@dataclass(frozen=True, slots=True)
class GraphArtifact:
    id: str
    role: str
    path: str
    content_hash: str
    size_bytes: int
    media_type: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "path": self.path,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
        }


class EpistemicGraphRuntime:
    """Build, persist, compare, and render auditable epistemic graphs.

    A proof is represented as a node so that premises are AND-connected into
    that proof and separate proof nodes are OR-alternatives for the conclusion.
    This keeps the visual representation faithful to the support engine.
    """

    def __init__(self, ledger: "EpistemicLedger", artifact_root: str | Path | None = None):
        self.ledger = ledger
        if artifact_root is None:
            db_path = Path(getattr(ledger.db, "path", "orbita.db"))
            if str(db_path) == ":memory:":
                artifact_root = Path.cwd() / ".orbita_graph_artifacts"
            else:
                artifact_root = db_path.parent / "graph_artifacts"
        self.artifact_root = Path(artifact_root)

    # ------------------------------------------------------------------
    # Snapshot construction and persistence
    # ------------------------------------------------------------------
    def capture(
        self,
        *,
        name: str | None = None,
        root_claim_ids: Iterable[str] | None = None,
        include_descendants: bool = False,
        persist: bool = True,
        actor: str = "graph-runtime",
    ) -> dict[str, Any]:
        roots = list(dict.fromkeys(root_claim_ids or []))
        for claim_id in roots:
            self.ledger._require_claim(claim_id)

        claim_ids = self._claim_scope(roots, include_descendants=include_descendants)
        nodes, edges = self._build_graph(claim_ids)
        nodes = sorted(nodes, key=lambda item: (item["kind"], item["id"]))
        edges = sorted(edges, key=lambda item: (item["kind"], item["source"], item["target"], item["id"]))

        core = {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "name": name or "epistemic snapshot",
            "root_claim_ids": roots,
            "include_descendants": bool(include_descendants),
            "nodes": nodes,
            "edges": edges,
            "summary": self._summary(nodes, edges),
        }
        graph_hash = _hash_json(core)
        snapshot_id = _new_id("gph")
        snapshot = {
            "id": snapshot_id,
            "created_at": _utcnow(),
            **core,
            "graph_hash": graph_hash,
        }
        if persist:
            self.ledger.db.conn.execute(
                """INSERT INTO graph_snapshots
                   (id, name, root_claim_ids_json, include_descendants, graph_json,
                    graph_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot_id,
                    snapshot["name"],
                    _stable_json(roots),
                    int(include_descendants),
                    _stable_json(snapshot),
                    graph_hash,
                    snapshot["created_at"],
                ),
            )
            self.ledger._event(
                "graph_snapshot",
                snapshot_id,
                "GRAPH_SNAPSHOT_CAPTURED",
                {
                    "name": snapshot["name"],
                    "graph_hash": graph_hash,
                    "root_claim_ids": roots,
                    "summary": snapshot["summary"],
                },
                actor,
                self._tool_role(),
            )
            self.ledger.db.conn.commit()
        return snapshot

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT graph_json FROM graph_snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown graph snapshot: {snapshot_id}")
        return json.loads(row["graph_json"])

    def list_snapshots(self) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            """SELECT id, name, root_claim_ids_json, include_descendants,
                      graph_hash, created_at
               FROM graph_snapshots ORDER BY created_at"""
        ).fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "root_claim_ids": json.loads(row["root_claim_ids_json"]),
                "include_descendants": bool(row["include_descendants"]),
                "graph_hash": row["graph_hash"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def verify_snapshot(self, snapshot_id: str) -> bool:
        snapshot = self.get_snapshot(snapshot_id)
        core = {
            key: snapshot[key]
            for key in (
                "schema_version",
                "name",
                "root_claim_ids",
                "include_descendants",
                "nodes",
                "edges",
                "summary",
            )
        }
        calculated = _hash_json(core)
        row = self.ledger.db.conn.execute(
            "SELECT graph_hash FROM graph_snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
        return calculated == snapshot.get("graph_hash") == row["graph_hash"]

    # ------------------------------------------------------------------
    # Diff construction and persistence
    # ------------------------------------------------------------------
    def compare(
        self,
        before: str | dict[str, Any],
        after: str | dict[str, Any],
        *,
        name: str | None = None,
        persist: bool = True,
        actor: str = "graph-runtime",
    ) -> dict[str, Any]:
        before_graph = self.get_snapshot(before) if isinstance(before, str) else before
        after_graph = self.get_snapshot(after) if isinstance(after, str) else after
        before_nodes = {node["id"]: node for node in before_graph["nodes"]}
        after_nodes = {node["id"]: node for node in after_graph["nodes"]}
        before_edges = {edge["id"]: edge for edge in before_graph["edges"]}
        after_edges = {edge["id"]: edge for edge in after_graph["edges"]}

        added_nodes = [after_nodes[node_id] for node_id in sorted(after_nodes.keys() - before_nodes.keys())]
        removed_nodes = [before_nodes[node_id] for node_id in sorted(before_nodes.keys() - after_nodes.keys())]
        changed_nodes: list[dict[str, Any]] = []
        for node_id in sorted(before_nodes.keys() & after_nodes.keys()):
            left = _node_signature(before_nodes[node_id])
            right = _node_signature(after_nodes[node_id])
            if left != right:
                changed_nodes.append(
                    {
                        "id": node_id,
                        "kind": after_nodes[node_id]["kind"],
                        "label": after_nodes[node_id].get("label", node_id),
                        "before": left,
                        "after": right,
                        "transition": self._transition(before_nodes[node_id], after_nodes[node_id]),
                    }
                )

        added_edges = [after_edges[edge_id] for edge_id in sorted(after_edges.keys() - before_edges.keys())]
        removed_edges = [before_edges[edge_id] for edge_id in sorted(before_edges.keys() - after_edges.keys())]
        changed_edges: list[dict[str, Any]] = []
        for edge_id in sorted(before_edges.keys() & after_edges.keys()):
            left = _edge_signature(before_edges[edge_id])
            right = _edge_signature(after_edges[edge_id])
            if left != right:
                changed_edges.append({"id": edge_id, "before": left, "after": right})

        causal_paths = self._causal_paths(before_graph, changed_nodes)
        transitions = Counter(item["transition"] for item in changed_nodes)
        core = {
            "schema_version": DIFF_SCHEMA_VERSION,
            "name": name or "epistemic collapse diff",
            "before_snapshot_id": before_graph.get("id"),
            "after_snapshot_id": after_graph.get("id"),
            "before_graph_hash": before_graph.get("graph_hash"),
            "after_graph_hash": after_graph.get("graph_hash"),
            "added_nodes": added_nodes,
            "removed_nodes": removed_nodes,
            "changed_nodes": changed_nodes,
            "added_edges": added_edges,
            "removed_edges": removed_edges,
            "changed_edges": changed_edges,
            "causal_paths": causal_paths,
            "summary": {
                "added_nodes": len(added_nodes),
                "removed_nodes": len(removed_nodes),
                "changed_nodes": len(changed_nodes),
                "added_edges": len(added_edges),
                "removed_edges": len(removed_edges),
                "changed_edges": len(changed_edges),
                "transitions": dict(sorted(transitions.items())),
                "collapsed_claims": [
                    item["id"]
                    for item in changed_nodes
                    if item["kind"] == "claim" and item["transition"] == "collapsed"
                ],
                "recovered_claims": [
                    item["id"]
                    for item in changed_nodes
                    if item["kind"] == "claim" and item["transition"] == "recovered"
                ],
                "preserved_root_claims": self._preserved_roots(before_graph, after_graph),
            },
        }
        diff_hash = _hash_json(core)
        diff_id = _new_id("gdf")
        diff = {"id": diff_id, "created_at": _utcnow(), **core, "diff_hash": diff_hash}
        if persist:
            if not before_graph.get("id") or not after_graph.get("id"):
                raise ValueError("Persisted diffs require persisted before and after snapshots")
            self.ledger.db.conn.execute(
                """INSERT INTO graph_diffs
                   (id, name, before_snapshot_id, after_snapshot_id, diff_json,
                    diff_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    diff_id,
                    diff["name"],
                    before_graph["id"],
                    after_graph["id"],
                    _stable_json(diff),
                    diff_hash,
                    diff["created_at"],
                ),
            )
            self.ledger._event(
                "graph_diff",
                diff_id,
                "GRAPH_DIFF_CAPTURED",
                {
                    "name": diff["name"],
                    "before_snapshot_id": before_graph["id"],
                    "after_snapshot_id": after_graph["id"],
                    "diff_hash": diff_hash,
                    "summary": diff["summary"],
                },
                actor,
                self._tool_role(),
            )
            self.ledger.db.conn.commit()
        return diff

    def get_diff(self, diff_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT diff_json FROM graph_diffs WHERE id = ?", (diff_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown graph diff: {diff_id}")
        return json.loads(row["diff_json"])

    def list_diffs(self) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            """SELECT id, name, before_snapshot_id, after_snapshot_id,
                      diff_hash, created_at
               FROM graph_diffs ORDER BY created_at"""
        ).fetchall()
        return [dict(row) for row in rows]

    def verify_diff(self, diff_id: str) -> bool:
        diff = self.get_diff(diff_id)
        core = {
            key: diff[key]
            for key in (
                "schema_version",
                "name",
                "before_snapshot_id",
                "after_snapshot_id",
                "before_graph_hash",
                "after_graph_hash",
                "added_nodes",
                "removed_nodes",
                "changed_nodes",
                "added_edges",
                "removed_edges",
                "changed_edges",
                "causal_paths",
                "summary",
            )
        }
        calculated = _hash_json(core)
        row = self.ledger.db.conn.execute(
            "SELECT diff_hash FROM graph_diffs WHERE id = ?", (diff_id,)
        ).fetchone()
        return calculated == diff.get("diff_hash") == row["diff_hash"]

    # ------------------------------------------------------------------
    # Rendering and artifact integrity
    # ------------------------------------------------------------------
    def render_snapshot(
        self,
        snapshot: str | dict[str, Any],
        *,
        output_dir: str | Path | None = None,
        formats: Iterable[str] = ("json", "dot", "svg", "html"),
    ) -> list[dict[str, Any]]:
        graph = self.get_snapshot(snapshot) if isinstance(snapshot, str) else snapshot
        output = Path(output_dir) if output_dir is not None else self.artifact_root / graph["id"]
        output.mkdir(parents=True, exist_ok=True)
        requested = tuple(dict.fromkeys(formats))
        artifacts: list[GraphArtifact] = []
        stem = self._safe_stem(graph.get("name") or graph["id"])

        json_path = output / f"{stem}.json"
        dot_path = output / f"{stem}.dot"
        svg_path = output / f"{stem}.svg"
        html_path = output / f"{stem}.html"
        if "json" in requested:
            self._write_text(json_path, json.dumps(graph, indent=2, ensure_ascii=False))
            artifacts.append(self._artifact("snapshot_json", json_path, "application/json"))
        dot_text = self._snapshot_dot(graph)
        if "dot" in requested:
            self._write_text(dot_path, dot_text)
            artifacts.append(self._artifact("snapshot_dot", dot_path, "text/vnd.graphviz"))
        svg_text: str | None = None
        if "svg" in requested or "html" in requested:
            svg_text = self._render_dot(dot_text)
        if "svg" in requested:
            self._write_text(svg_path, svg_text or "")
            artifacts.append(self._artifact("snapshot_svg", svg_path, "image/svg+xml"))
        if "html" in requested:
            self._write_text(html_path, self._snapshot_html(graph, svg_text or ""))
            artifacts.append(self._artifact("snapshot_html", html_path, "text/html"))

        self._persist_artifacts(snapshot_id=graph.get("id"), diff_id=None, artifacts=artifacts)
        return [artifact.as_dict() for artifact in artifacts]

    def render_diff(
        self,
        diff: str | dict[str, Any],
        *,
        output_dir: str | Path | None = None,
        formats: Iterable[str] = ("json", "dot", "svg", "html"),
    ) -> list[dict[str, Any]]:
        change = self.get_diff(diff) if isinstance(diff, str) else diff
        after = self.get_snapshot(change["after_snapshot_id"])
        output = Path(output_dir) if output_dir is not None else self.artifact_root / change["id"]
        output.mkdir(parents=True, exist_ok=True)
        requested = tuple(dict.fromkeys(formats))
        artifacts: list[GraphArtifact] = []
        stem = self._safe_stem(change.get("name") or change["id"])

        json_path = output / f"{stem}.json"
        dot_path = output / f"{stem}.dot"
        svg_path = output / f"{stem}.svg"
        html_path = output / f"{stem}.html"
        if "json" in requested:
            self._write_text(json_path, json.dumps(change, indent=2, ensure_ascii=False))
            artifacts.append(self._artifact("diff_json", json_path, "application/json"))
        dot_text = self._diff_dot(change, after)
        if "dot" in requested:
            self._write_text(dot_path, dot_text)
            artifacts.append(self._artifact("diff_dot", dot_path, "text/vnd.graphviz"))
        svg_text: str | None = None
        if "svg" in requested or "html" in requested:
            svg_text = self._render_dot(dot_text)
        if "svg" in requested:
            self._write_text(svg_path, svg_text or "")
            artifacts.append(self._artifact("diff_svg", svg_path, "image/svg+xml"))
        if "html" in requested:
            self._write_text(html_path, self._diff_html(change, after, svg_text or ""))
            artifacts.append(self._artifact("diff_html", html_path, "text/html"))

        self._persist_artifacts(snapshot_id=None, diff_id=change.get("id"), artifacts=artifacts)
        return [artifact.as_dict() for artifact in artifacts]

    def list_artifacts(
        self, *, snapshot_id: str | None = None, diff_id: str | None = None
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if snapshot_id is not None:
            clauses.append("snapshot_id = ?")
            params.append(snapshot_id)
        if diff_id is not None:
            clauses.append("diff_id = ?")
            params.append(diff_id)
        sql = "SELECT * FROM graph_artifacts"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at"
        return [dict(row) for row in self.ledger.db.conn.execute(sql, params).fetchall()]

    def verify_artifacts(
        self, *, snapshot_id: str | None = None, diff_id: str | None = None
    ) -> bool:
        artifacts = self.list_artifacts(snapshot_id=snapshot_id, diff_id=diff_id)
        if not artifacts:
            return False
        for artifact in artifacts:
            path = Path(artifact["path"])
            if not path.is_file():
                return False
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != artifact["content_hash"] or path.stat().st_size != artifact["size_bytes"]:
                return False
        return True

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------
    def _claim_scope(self, roots: list[str], *, include_descendants: bool) -> list[str]:
        conn = self.ledger.db.conn
        if not roots:
            return [row["id"] for row in conn.execute("SELECT id FROM claims ORDER BY id").fetchall()]
        included = set(roots)
        queue = deque(roots)
        while queue:
            current = queue.popleft()
            premise_rows = conn.execute(
                """SELECT pp.premise_claim_id
                   FROM proofs p JOIN proof_premises pp ON pp.proof_id = p.id
                   WHERE p.conclusion_claim_id = ? AND p.active = 1""",
                (current,),
            ).fetchall()
            contradiction_rows = conn.execute(
                """SELECT CASE WHEN claim_a = ? THEN claim_b ELSE claim_a END AS other
                   FROM contradictions
                   WHERE active = 1 AND (claim_a = ? OR claim_b = ?)""",
                (current, current, current),
            ).fetchall()
            candidates = [row["premise_claim_id"] for row in premise_rows]
            candidates.extend(row["other"] for row in contradiction_rows)
            if include_descendants:
                candidates.extend(self.ledger.descendants_of_claim(current))
            for claim_id in candidates:
                if claim_id not in included:
                    included.add(claim_id)
                    queue.append(claim_id)
        return sorted(included)

    def _build_graph(self, claim_ids: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from .support import SupportEngine

        conn = self.ledger.db.conn
        reports = SupportEngine(self.ledger).evaluate_many(claim_ids)
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}

        for claim_id in claim_ids:
            claim = self.ledger.get_claim(claim_id)
            report = reports[claim_id]
            metadata: dict[str, Any] = {
                "claim_type": claim["claim_type"],
                "scope": claim["scope"],
                "reasons": report.reasons,
                "direct_support_sources": report.direct_support_sources,
                "direct_refute_sources": report.direct_refute_sources,
                "satisfied_proofs": report.satisfied_proofs,
                "broken_proofs": report.broken_proofs,
                "contradictions": report.contradictions,
            }
            if "relation" in claim:
                relation = claim["relation"]
                metadata["relation"] = {
                    "subject": relation["subject_name"],
                    "predicate": relation["predicate"],
                    "object": relation["object"],
                    "polarity": relation["polarity"],
                    "valid_from": relation["valid_from"],
                    "valid_to": relation["valid_to"],
                    "qualifiers": relation["qualifiers"],
                }
            nodes[claim_id] = {
                "id": claim_id,
                "kind": "claim",
                "label": claim["canonical_text"],
                "status": claim["status"],
                "support_state": report.state.value,
                "metadata": metadata,
            }

        placeholders = ",".join("?" for _ in claim_ids) or "NULL"
        att_rows = conn.execute(
            f"""SELECT a.id AS attestation_id, a.claim_id, a.stance, a.confidence,
                       e.*
                FROM attestations a JOIN evidence e ON e.id = a.evidence_id
                WHERE a.claim_id IN ({placeholders})
                ORDER BY e.id, a.claim_id""",
            claim_ids,
        ).fetchall()
        evidence_ids: set[str] = set()
        for row in att_rows:
            evidence_ids.add(row["id"])
            nodes.setdefault(
                row["id"],
                {
                    "id": row["id"],
                    "kind": "evidence",
                    "label": row["excerpt"][:120] or row["source_uri"],
                    "active": bool(row["active"]),
                    "metadata": {
                        "source_uri": row["source_uri"],
                        "source_kind": row["source_kind"],
                        "content_hash": row["content_hash"],
                        "independence_key": row["independence_key"],
                        "metadata": _json_load(row["metadata_json"], {}),
                    },
                },
            )
            source_kind = row["source_kind"]
            warranting = source_kind not in {
                kind.value for kind in self.ledger.policy.non_warrant_source_kinds
            }
            edge = {
                "id": _edge_id(row["id"], row["claim_id"], "attestation", row["attestation_id"]),
                "source": row["id"],
                "target": row["claim_id"],
                "kind": "attestation",
                "label": row["stance"] if warranting else "proposal · non-warranting",
                "active": bool(row["active"]),
                "metadata": {
                    "attestation_id": row["attestation_id"],
                    "stance": row["stance"],
                    "confidence": row["confidence"],
                    "source_kind": source_kind,
                    "warranting": warranting,
                },
            }
            edges[edge["id"]] = edge

        proof_rows = conn.execute(
            f"""SELECT * FROM proofs WHERE conclusion_claim_id IN ({placeholders}) ORDER BY id""",
            claim_ids,
        ).fetchall()
        proof_ids: set[str] = set()
        for proof in proof_rows:
            proof_ids.add(proof["id"])
            nodes[proof["id"]] = {
                "id": proof["id"],
                "kind": "proof",
                "label": proof["rule"],
                "active": bool(proof["active"]),
                "metadata": {
                    "logic": "AND within this proof; OR across proof nodes for the same conclusion",
                    "rule": proof["rule"],
                    "metadata": _json_load(proof["metadata_json"], {}),
                },
            }
            premise_rows = conn.execute(
                """SELECT premise_claim_id, position FROM proof_premises
                   WHERE proof_id = ? ORDER BY position""",
                (proof["id"],),
            ).fetchall()
            for premise in premise_rows:
                if premise["premise_claim_id"] not in nodes:
                    # Scope should already include premises, but retain a safe reference.
                    claim = self.ledger.get_claim(premise["premise_claim_id"])
                    nodes[premise["premise_claim_id"]] = {
                        "id": premise["premise_claim_id"],
                        "kind": "claim",
                        "label": claim["canonical_text"],
                        "status": claim["status"],
                        "support_state": "not_evaluated",
                        "metadata": {"out_of_scope_reference": True},
                    }
                edge = {
                    "id": _edge_id(
                        premise["premise_claim_id"],
                        proof["id"],
                        "proof_premise",
                        premise["position"],
                    ),
                    "source": premise["premise_claim_id"],
                    "target": proof["id"],
                    "kind": "proof_premise",
                    "label": f"premise {premise['position'] + 1}",
                    "active": bool(proof["active"]),
                    "metadata": {"semantics": "AND", "position": premise["position"]},
                }
                edges[edge["id"]] = edge
            edge = {
                "id": _edge_id(proof["id"], proof["conclusion_claim_id"], "proof_conclusion"),
                "source": proof["id"],
                "target": proof["conclusion_claim_id"],
                "kind": "proof_conclusion",
                "label": "derives",
                "active": bool(proof["active"]),
                "metadata": {"semantics": "OR alternative"},
            }
            edges[edge["id"]] = edge

        contradiction_rows = conn.execute(
            f"""SELECT * FROM contradictions
                WHERE claim_a IN ({placeholders}) OR claim_b IN ({placeholders})
                ORDER BY id""",
            [*claim_ids, *claim_ids],
        ).fetchall()
        for row in contradiction_rows:
            if row["claim_a"] not in nodes or row["claim_b"] not in nodes:
                continue
            edge = {
                "id": _edge_id(row["claim_a"], row["claim_b"], "contradiction", row["id"]),
                "source": row["claim_a"],
                "target": row["claim_b"],
                "kind": "contradiction",
                "label": "contradicts",
                "active": bool(row["active"]),
                "metadata": {"contradiction_id": row["id"], "rationale": row["rationale"], "bidirectional": True},
            }
            edges[edge["id"]] = edge

        self._add_analysis_nodes(nodes, edges, claim_ids, evidence_ids)
        self._add_proposal_nodes(nodes, edges, claim_ids, proof_ids)
        return list(nodes.values()), list(edges.values())

    def _add_analysis_nodes(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: dict[str, dict[str, Any]],
        claim_ids: list[str],
        evidence_ids: set[str],
    ) -> None:
        conn = self.ledger.db.conn
        placeholders = ",".join("?" for _ in claim_ids) or "NULL"
        rows = conn.execute(
            f"""SELECT DISTINCT ar.*
                FROM analysis_receipts ar
                LEFT JOIN analysis_claim_assessments aca ON aca.receipt_id = ar.id
                WHERE aca.claim_id IN ({placeholders})
                   OR ar.evidence_id IN ({','.join('?' for _ in evidence_ids) or 'NULL'})
                ORDER BY ar.id""",
            [*claim_ids, *sorted(evidence_ids)],
        ).fetchall()
        for row in rows:
            nodes[row["id"]] = {
                "id": row["id"],
                "kind": "analysis_receipt",
                "label": f"{row['analysis_type']} · {row['status']}",
                "status": row["status"],
                "metadata": {
                    "dataset_uri": row["dataset_uri"],
                    "dataset_hash": row["dataset_hash"],
                    "code_hash": row["code_hash"],
                    "receipt_hash": row["receipt_hash"],
                    "parent_receipt_id": row["parent_receipt_id"],
                    "comparison": _json_load(row["comparison_json"], {}),
                },
            }
            if row["evidence_id"] and row["evidence_id"] in nodes:
                edge = {
                    "id": _edge_id(row["id"], row["evidence_id"], "receipt_evidence"),
                    "source": row["id"],
                    "target": row["evidence_id"],
                    "kind": "receipt_evidence",
                    "label": "materializes evidence",
                    "active": True,
                    "metadata": {},
                }
                edges[edge["id"]] = edge
            assessment_rows = conn.execute(
                """SELECT * FROM analysis_claim_assessments
                   WHERE receipt_id = ? ORDER BY position""",
                (row["id"],),
            ).fetchall()
            for assessment in assessment_rows:
                if assessment["claim_id"] not in nodes:
                    continue
                edge = {
                    "id": _edge_id(
                        row["id"],
                        assessment["claim_id"],
                        "analysis_assessment",
                        assessment["id"],
                    ),
                    "source": row["id"],
                    "target": assessment["claim_id"],
                    "kind": "analysis_assessment",
                    "label": assessment["outcome"],
                    "active": True,
                    "metadata": {
                        "assessment_id": assessment["id"],
                        "metric_path": assessment["metric_path"],
                        "metric_value": _json_load(assessment["metric_value_json"], None),
                        "outcome": assessment["outcome"],
                        "confidence": assessment["confidence"],
                        "rationale": assessment["rationale"],
                    },
                }
                edges[edge["id"]] = edge

    def _add_proposal_nodes(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: dict[str, dict[str, Any]],
        claim_ids: list[str],
        proof_ids: set[str],
    ) -> None:
        conn = self.ledger.db.conn
        target_ids = set(claim_ids) | set(proof_ids)
        if not target_ids:
            return
        placeholders = ",".join("?" for _ in target_ids)
        rows = conn.execute(
            f"""SELECT pi.*, pb.provider, pb.model_name, pb.model_version,
                       pb.response_id, pb.status AS batch_status, pb.response_hash,
                       pb.system_prompt_hash, pb.user_prompt_hash, pb.created_at AS batch_created_at
                FROM proposal_items pi
                JOIN proposal_batches pb ON pb.id = pi.batch_id
                WHERE pi.durable_entity_id IN ({placeholders})
                ORDER BY pb.id, pi.position""",
            sorted(target_ids),
        ).fetchall()
        batch_ids: set[str] = set()
        for row in rows:
            batch_ids.add(row["batch_id"])
            batch_node_id = row["batch_id"]
            nodes.setdefault(
                batch_node_id,
                {
                    "id": batch_node_id,
                    "kind": "proposal_batch",
                    "label": f"{row['provider']}/{row['model_name']}",
                    "status": row["batch_status"],
                    "metadata": {
                        "provider": row["provider"],
                        "model_name": row["model_name"],
                        "model_version": row["model_version"],
                        "response_id": row["response_id"],
                        "response_hash": row["response_hash"],
                        "system_prompt_hash": row["system_prompt_hash"],
                        "user_prompt_hash": row["user_prompt_hash"],
                    },
                },
            )
            item_id = row["id"]
            nodes[item_id] = {
                "id": item_id,
                "kind": "proposal_item",
                "label": f"{row['item_type']}:{row['local_id']}",
                "status": row["status"],
                "review_status": row["status"],
                "metadata": {
                    "payload_hash": row["payload_hash"],
                    "rationale": row["rationale"],
                    "requires_human_review": bool(row["requires_human_review"]),
                    "review_reason": row["review_reason"],
                    "reviewed_by": row["reviewed_by"],
                    "durable_entity_type": row["durable_entity_type"],
                    "durable_entity_id": row["durable_entity_id"],
                },
            }
            membership = {
                "id": _edge_id(batch_node_id, item_id, "proposal_membership"),
                "source": batch_node_id,
                "target": item_id,
                "kind": "proposal_membership",
                "label": "contains",
                "active": True,
                "metadata": {"position": row["position"]},
            }
            edges[membership["id"]] = membership
            if row["durable_entity_id"] in nodes:
                materializes = {
                    "id": _edge_id(item_id, row["durable_entity_id"], "proposal_materialization"),
                    "source": item_id,
                    "target": row["durable_entity_id"],
                    "kind": "proposal_materialization",
                    "label": "proposes",
                    "active": row["status"] == "applied",
                    "metadata": {"durable_entity_type": row["durable_entity_type"]},
                }
                edges[materializes["id"]] = materializes

    # ------------------------------------------------------------------
    # Diff helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _transition(before: dict[str, Any], after: dict[str, Any]) -> str:
        if before.get("kind") != after.get("kind"):
            return "type_changed"
        if before.get("kind") == "claim":
            left = before.get("support_state")
            right = after.get("support_state")
            if left == right and before.get("status") != after.get("status"):
                return "status_changed"
            if left == right:
                return "metadata_changed"
            if left in {"supported", "challenged"} and right in {"unknown", "unsupported"}:
                return "collapsed"
            if left in {"unknown", "unsupported"} and right in {"supported", "challenged"}:
                return "recovered"
            if left == "supported" and right == "challenged":
                return "challenged"
            if left == "challenged" and right == "supported":
                return "resolved"
            return "support_changed"
        if before.get("kind") == "evidence" and before.get("active") != after.get("active"):
            return "revoked" if before.get("active") else "reactivated"
        if before.get("active") != after.get("active"):
            return "deactivated" if before.get("active") else "reactivated"
        if before.get("status") != after.get("status"):
            return "status_changed"
        return "metadata_changed"

    @staticmethod
    def _preserved_roots(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
        before_nodes = {node["id"]: node for node in before["nodes"]}
        after_nodes = {node["id"]: node for node in after["nodes"]}
        roots = set(before.get("root_claim_ids", [])) | set(after.get("root_claim_ids", []))
        return sorted(
            claim_id
            for claim_id in roots
            if claim_id in before_nodes
            and claim_id in after_nodes
            and before_nodes[claim_id].get("support_state") in {"supported", "challenged"}
            and after_nodes[claim_id].get("support_state") in {"supported", "challenged"}
        )

    def _causal_paths(
        self, before_graph: dict[str, Any], changed_nodes: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        changed_evidence = {
            item["id"]
            for item in changed_nodes
            if item["kind"] == "evidence" and item["transition"] in {"revoked", "deactivated"}
        }
        changed_claims = {
            item["id"]
            for item in changed_nodes
            if item["kind"] == "claim"
            and item["transition"] in {"collapsed", "challenged", "support_changed"}
        }
        if not changed_evidence or not changed_claims:
            return []
        adjacency: dict[str, list[tuple[str, str]]] = {}
        causal_kinds = {"attestation", "proof_premise", "proof_conclusion", "contradiction"}
        for edge in before_graph["edges"]:
            if edge["kind"] not in causal_kinds or not edge.get("active", True):
                continue
            adjacency.setdefault(edge["source"], []).append((edge["target"], edge["id"]))
            if edge["kind"] == "contradiction":
                adjacency.setdefault(edge["target"], []).append((edge["source"], edge["id"]))
        paths: list[dict[str, Any]] = []
        for source in sorted(changed_evidence):
            queue: deque[tuple[str, list[str], list[str]]] = deque([(source, [source], [])])
            visited = {source}
            while queue:
                node_id, node_path, edge_path = queue.popleft()
                if node_id in changed_claims:
                    paths.append(
                        {
                            "trigger_node_id": source,
                            "affected_claim_id": node_id,
                            "node_path": node_path,
                            "edge_path": edge_path,
                        }
                    )
                if len(node_path) > 20:
                    continue
                for target, edge_id in sorted(adjacency.get(node_id, [])):
                    if target in visited:
                        continue
                    visited.add(target)
                    queue.append((target, [*node_path, target], [*edge_path, edge_id]))
        return paths

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _summary(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
        node_kinds = Counter(node["kind"] for node in nodes)
        edge_kinds = Counter(edge["kind"] for edge in edges)
        support_states = Counter(
            node.get("support_state") for node in nodes if node["kind"] == "claim"
        )
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes_by_kind": dict(sorted(node_kinds.items())),
            "edges_by_kind": dict(sorted(edge_kinds.items())),
            "claims_by_support_state": dict(sorted(support_states.items())),
            "inactive_evidence": sorted(
                node["id"]
                for node in nodes
                if node["kind"] == "evidence" and not node.get("active", True)
            ),
        }

    @staticmethod
    def _safe_stem(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
        return cleaned.strip("_")[:80] or "graph"

    @staticmethod
    def _dot_escape(value: Any) -> str:
        text = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return text.replace("\n", "\\n")

    def _snapshot_dot(self, graph: dict[str, Any], overrides: dict[str, str] | None = None) -> str:
        overrides = overrides or {}
        lines = [
            "digraph Orbita {",
            '  graph [rankdir="LR", bgcolor="white", pad="0.25", nodesep="0.35", ranksep="0.7", fontname="Arial"];',
            '  node [shape="box", style="rounded,filled", fontname="Arial", fontsize="10", color="#475569", fontcolor="#0f172a", margin="0.12,0.08"];',
            '  edge [fontname="Arial", fontsize="8", color="#64748b", arrowsize="0.7"];',
            f'  label="{self._dot_escape(graph.get("name", "Orbita epistemic graph"))}";',
            '  labelloc="t";',
        ]
        for node in graph["nodes"]:
            attrs = self._node_dot_attrs(node)
            if node["id"] in overrides:
                attrs["color"] = overrides[node["id"]]
                attrs["penwidth"] = "3"
            attr_text = ", ".join(f'{key}="{self._dot_escape(value)}"' for key, value in attrs.items())
            lines.append(f'  "{self._dot_escape(node["id"])}" [{attr_text}];')
        for edge in graph["edges"]:
            attrs = self._edge_dot_attrs(edge)
            attr_text = ", ".join(f'{key}="{self._dot_escape(value)}"' for key, value in attrs.items())
            lines.append(
                f'  "{self._dot_escape(edge["source"])}" -> "{self._dot_escape(edge["target"])}" [{attr_text}];'
            )
        lines.append("}")
        return "\n".join(lines)

    def _diff_dot(self, diff: dict[str, Any], after: dict[str, Any]) -> str:
        override: dict[str, str] = {}
        transition_color = {
            "collapsed": "#dc2626",
            "recovered": "#16a34a",
            "challenged": "#d97706",
            "resolved": "#2563eb",
            "revoked": "#dc2626",
            "reactivated": "#16a34a",
            "status_changed": "#7c3aed",
            "support_changed": "#d97706",
        }
        for item in diff["changed_nodes"]:
            override[item["id"]] = transition_color.get(item["transition"], "#7c3aed")
        graph = dict(after)
        graph["name"] = diff.get("name", "Epistemic collapse diff")
        return self._snapshot_dot(graph, overrides=override)

    @staticmethod
    def _node_dot_attrs(node: dict[str, Any]) -> dict[str, str]:
        kind = node["kind"]
        label = node.get("label", node["id"])
        if len(label) > 80:
            label = label[:77] + "…"
        sub = node["id"]
        if kind == "claim":
            state = node.get("support_state", "unknown")
            fill = {
                "supported": "#dcfce7",
                "challenged": "#fef3c7",
                "unsupported": "#fee2e2",
                "unknown": "#e2e8f0",
                "not_evaluated": "#f1f5f9",
            }.get(state, "#e2e8f0")
            return {"label": f"{label}\n[{state}]\n{sub}", "fillcolor": fill, "shape": "box"}
        if kind == "evidence":
            fill = "#dbeafe" if node.get("active", True) else "#fecaca"
            state = "active" if node.get("active", True) else "revoked"
            return {"label": f"Evidence: {label}\n[{state}]\n{sub}", "fillcolor": fill, "shape": "note"}
        if kind == "proof":
            return {"label": f"Proof: {label}\n{sub}", "fillcolor": "#ede9fe", "shape": "diamond"}
        if kind == "analysis_receipt":
            return {"label": f"Receipt: {label}\n{sub}", "fillcolor": "#cffafe", "shape": "component"}
        if kind == "proposal_batch":
            return {"label": f"Model batch: {label}\n{sub}", "fillcolor": "#fce7f3", "shape": "folder"}
        if kind == "proposal_item":
            return {"label": f"Proposal: {label}\n{sub}", "fillcolor": "#fae8ff", "shape": "box"}
        return {"label": f"{label}\n{sub}", "fillcolor": "#f1f5f9", "shape": "box"}

    @staticmethod
    def _edge_dot_attrs(edge: dict[str, Any]) -> dict[str, str]:
        kind = edge["kind"]
        label = edge.get("label", kind)
        attrs = {"label": label}
        if kind == "attestation":
            metadata = edge.get("metadata", {})
            stance = metadata.get("stance", label)
            if not metadata.get("warranting", True):
                attrs.update(color="#64748b", style="dotted")
            else:
                attrs.update(color="#16a34a" if stance == "support" else "#dc2626")
        elif kind == "proof_premise":
            attrs.update(color="#7c3aed", style="bold")
        elif kind == "proof_conclusion":
            attrs.update(color="#4f46e5", style="bold")
        elif kind == "contradiction":
            attrs.update(color="#dc2626", style="dashed", dir="both")
        elif kind.startswith("proposal_"):
            attrs.update(color="#a21caf", style="dotted")
        elif kind.startswith("analysis_") or kind == "receipt_evidence":
            attrs.update(color="#0891b2")
        if not edge.get("active", True):
            attrs.update(style="dashed", color="#94a3b8")
        return attrs

    @staticmethod
    def _render_dot(dot_text: str) -> str:
        executable = shutil.which("dot")
        if executable is None:
            raise RuntimeError(
                "Graphviz 'dot' was not found. JSON and DOT export remain available; install Graphviz for SVG/HTML."
            )
        proc = subprocess.run(
            [executable, "-Tsvg"],
            input=dot_text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Graphviz rendering failed: {proc.stderr.decode('utf-8', 'replace')}")
        return proc.stdout.decode("utf-8")

    def _snapshot_html(self, graph: dict[str, Any], svg: str) -> str:
        title = html.escape(graph.get("name", "Orbita epistemic graph"))
        graph_json = json.dumps({node["id"]: node for node in graph["nodes"]}, ensure_ascii=False).replace("</", "<\\/")
        summary = html.escape(json.dumps(graph["summary"], indent=2, ensure_ascii=False))
        return self._html_shell(
            title,
            svg,
            graph_json,
            f"<h2>Snapshot summary</h2><pre>{summary}</pre>",
        )

    def _diff_html(self, diff: dict[str, Any], after: dict[str, Any], svg: str) -> str:
        title = html.escape(diff.get("name", "Orbita collapse diff"))
        nodes = {node["id"]: node for node in after["nodes"]}
        changed = {item["id"]: item for item in diff["changed_nodes"]}
        for node_id, item in changed.items():
            if node_id in nodes:
                nodes[node_id] = {**nodes[node_id], "diff": item}
        graph_json = json.dumps(nodes, ensure_ascii=False).replace("</", "<\\/")
        rows = []
        for item in diff["changed_nodes"]:
            rows.append(
                "<tr>"
                f"<td><code>{html.escape(item['id'])}</code></td>"
                f"<td>{html.escape(item['kind'])}</td>"
                f"<td>{html.escape(item['transition'])}</td>"
                f"<td>{html.escape(str(item['before'].get('support_state', item['before'].get('active', ''))))}</td>"
                f"<td>{html.escape(str(item['after'].get('support_state', item['after'].get('active', ''))))}</td>"
                "</tr>"
            )
        table = (
            "<h2>State transitions</h2>"
            "<table><thead><tr><th>ID</th><th>Kind</th><th>Transition</th><th>Before</th><th>After</th></tr></thead>"
            f"<tbody>{''.join(rows) or '<tr><td colspan=5>No state changes</td></tr>'}</tbody></table>"
            f"<h2>Diff summary</h2><pre>{html.escape(json.dumps(diff['summary'], indent=2))}</pre>"
        )
        return self._html_shell(title, svg, graph_json, table)

    @staticmethod
    def _html_shell(title: str, svg: str, node_json: str, extra: str) -> str:
        # Graphviz emits an XML declaration and an SVG doctype for standalone
        # files. Strip those wrappers before embedding the SVG in HTML.
        svg_start = svg.find("<svg")
        embedded_svg = svg[svg_start:] if svg_start >= 0 else svg
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0; background: #f8fafc; color: #0f172a; }}
header {{ padding: 18px 24px; background: #0f172a; color: white; }}
main {{ display: grid; grid-template-columns: minmax(0, 1fr) 360px; min-height: calc(100vh - 70px); }}
#canvas {{ overflow: auto; padding: 18px; background: white; }}
#canvas svg {{ min-width: 900px; height: auto; }}
aside {{ border-left: 1px solid #cbd5e1; padding: 18px; overflow: auto; background: #f8fafc; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #e2e8f0; padding: 12px; border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #cbd5e1; padding: 7px; text-align: left; vertical-align: top; }}
.node {{ cursor: pointer; }}
.node:hover polygon, .node:hover path, .node:hover ellipse {{ filter: brightness(.92); }}
code {{ font-size: 12px; }}
@media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} aside {{ border-left: 0; border-top: 1px solid #cbd5e1; }} }}
</style>
</head>
<body>
<header><h1>{title}</h1><div>Click a node to inspect its exact epistemic record.</div></header>
<main>
<section id="canvas">{embedded_svg}</section>
<aside>
<div id="details"><h2>Node details</h2><p>Select a node in the graph.</p></div>
{extra}
</aside>
</main>
<script>
const nodes = {node_json};
document.querySelectorAll('#canvas g.node').forEach(group => {{
  group.addEventListener('click', () => {{
    const title = group.querySelector('title');
    if (!title) return;
    const item = nodes[title.textContent];
    if (!item) return;
    document.getElementById('details').innerHTML = '<h2>Node details</h2><pre>' +
      JSON.stringify(item, null, 2).replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c])) + '</pre>';
  }});
}});
</script>
</body>
</html>"""

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def _artifact(role: str, path: Path, media_type: str) -> GraphArtifact:
        content = path.read_bytes()
        return GraphArtifact(
            id=_new_id("gar"),
            role=role,
            path=str(path.resolve()),
            content_hash=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            media_type=media_type,
        )

    def _persist_artifacts(
        self,
        *,
        snapshot_id: str | None,
        diff_id: str | None,
        artifacts: list[GraphArtifact],
    ) -> None:
        if snapshot_id is None and diff_id is None:
            return
        for artifact in artifacts:
            self.ledger.db.conn.execute(
                """INSERT INTO graph_artifacts
                   (id, snapshot_id, diff_id, role, path, content_hash,
                    size_bytes, media_type, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    artifact.id,
                    snapshot_id,
                    diff_id,
                    artifact.role,
                    artifact.path,
                    artifact.content_hash,
                    artifact.size_bytes,
                    artifact.media_type,
                    _utcnow(),
                ),
            )
        self.ledger.db.conn.commit()

    @staticmethod
    def _tool_role():
        from .models import ActorRole

        return ActorRole.TOOL
