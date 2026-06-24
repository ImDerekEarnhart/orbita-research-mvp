"""Graph data builder and visualization page for Orbita Research MVP."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------

def build_graph_data(case_id: str, conn: Any) -> dict[str, Any]:
    """Return vis.js-compatible nodes + edges from a case's belief graph."""

    claim_rows = conn.execute(
        """SELECT cc.claim_id, cc.finding_type, cc.run_id AS case_run_id,
                  c.canonical_text, c.status, c.claim_type, c.created_at
           FROM case_claims cc
           JOIN claims c ON c.id = cc.claim_id
           WHERE cc.case_id = ?
           ORDER BY cc.created_at""",
        (case_id,),
    ).fetchall()

    claim_ids = [r["claim_id"] for r in claim_rows]

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_id_set: set[str] = set()

    def add_node(n: dict[str, Any]) -> None:
        if n["id"] not in node_id_set:
            node_id_set.add(n["id"])
            nodes.append(n)

    # --- Claim nodes ---
    for r in claim_rows:
        text = r["canonical_text"]
        label = (text[:55] + "…") if len(text) > 55 else text
        add_node({
            "id": r["claim_id"],
            "label": label,
            "type": "claim",
            "status": r["status"],
            "finding_type": r["finding_type"],
            "full_text": text,
            "claim_type": r["claim_type"],
            "created_at": r["created_at"],
        })

    # --- Source file nodes ---
    file_rows = conn.execute(
        "SELECT id, original_name, artifact_kind, stored_path, created_at FROM case_files WHERE case_id = ?",
        (case_id,),
    ).fetchall()
    file_stored_paths: dict[str, str] = {}
    for r in file_rows:
        add_node({
            "id": r["id"],
            "label": r["original_name"],
            "type": "source",
            "artifact_kind": r["artifact_kind"],
            "created_at": r["created_at"],
        })
        if r["stored_path"]:
            file_stored_paths[r["id"]] = r["stored_path"].replace("\\", "/")

    # --- Analysis run nodes ---
    run_rows = conn.execute(
        "SELECT id, status, started_at, completed_at FROM case_runs WHERE case_id = ? ORDER BY started_at",
        (case_id,),
    ).fetchall()
    run_id_set: set[str] = {r["id"] for r in run_rows}
    for i, r in enumerate(run_rows, 1):
        add_node({
            "id": r["id"],
            "label": f"Analysis Run #{i}",
            "type": "analysis_run",
            "status": r["status"],
            "started_at": r["started_at"],
            "completed_at": r["completed_at"],
        })

    # --- Run → claim edges (generates) ---
    for r in claim_rows:
        if r["case_run_id"] and r["case_run_id"] in run_id_set:
            edges.append({
                "id": f"gen_{r['case_run_id']}_{r['claim_id']}",
                "from": r["case_run_id"],
                "to": r["claim_id"],
                "type": "generates",
                "label": "generates",
            })

    case_row = conn.execute(
        "SELECT id, name, goal, status, created_at FROM research_cases WHERE id = ?",
        (case_id,),
    ).fetchone()
    meta = {
        "case_id": case_id,
        "case_name": case_row["name"] if case_row else case_id,
        "claim_count": len(claim_rows),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if not claim_ids:
        return {"nodes": nodes, "edges": edges, "meta": meta}

    ph = ",".join("?" * len(claim_ids))

    # --- Evidence nodes + attestation edges ---
    att_rows = conn.execute(
        f"""SELECT e.id, e.source_uri, e.source_kind, e.excerpt, e.active, e.created_at,
                   a.stance, a.claim_id, a.confidence
            FROM attestations a
            JOIN evidence e ON e.id = a.evidence_id
            WHERE a.claim_id IN ({ph})""",
        claim_ids,
    ).fetchall()

    evidence_seen: dict[str, dict[str, Any]] = {}
    for r in att_rows:
        ev_id = r["id"]
        if ev_id not in evidence_seen:
            excerpt = r["excerpt"] or ""
            kind_short = (r["source_kind"] or "").replace("dataset_analysis_receipt", "analysis").replace("dataset", "data")
            label = kind_short + ": " + (excerpt[:38] + "…" if len(excerpt) > 38 else excerpt)
            ev_node: dict[str, Any] = {
                "id": ev_id,
                "label": label,
                "type": "evidence",
                "source_kind": r["source_kind"],
                "source_uri": r["source_uri"],
                "excerpt": excerpt,
                "active": bool(r["active"]),
                "created_at": r["created_at"],
            }
            add_node(ev_node)
            evidence_seen[ev_id] = ev_node

        edge_type = "supports" if r["stance"] == "support" else "refutes"
        edges.append({
            "id": f"att_{ev_id}_{r['claim_id']}",
            "from": ev_id,
            "to": r["claim_id"],
            "type": edge_type,
            "label": edge_type,
            "confidence": r["confidence"],
        })

    # --- File → evidence edges (source_uri match) ---
    for ev_id, ev_node in evidence_seen.items():
        uri = (ev_node.get("source_uri") or "").replace("\\", "/")
        for file_id, stored in file_stored_paths.items():
            if stored and stored in uri:
                edges.append({
                    "id": f"src_ev_{file_id}_{ev_id}",
                    "from": file_id,
                    "to": ev_id,
                    "type": "source_of",
                    "label": "source",
                })
                break

    # --- Derivation (proof) edges ---
    proof_rows = conn.execute(
        f"""SELECT p.id, p.conclusion_claim_id, pp.premise_claim_id, p.rule
            FROM proofs p
            JOIN proof_premises pp ON pp.proof_id = p.id
            WHERE p.conclusion_claim_id IN ({ph}) AND pp.premise_claim_id IN ({ph}) AND p.active = 1""",
        claim_ids + claim_ids,
    ).fetchall()
    for r in proof_rows:
        edges.append({
            "id": f"proof_{r['id']}_{r['premise_claim_id']}",
            "from": r["premise_claim_id"],
            "to": r["conclusion_claim_id"],
            "type": "derived_from",
            "label": "derived from",
            "rule": r["rule"],
        })

    # --- Contradiction edges ---
    contra_rows = conn.execute(
        f"""SELECT id, claim_a, claim_b, rationale, active FROM contradictions
            WHERE (claim_a IN ({ph}) OR claim_b IN ({ph})) AND active = 1""",
        claim_ids + claim_ids,
    ).fetchall()
    for r in contra_rows:
        edges.append({
            "id": f"contra_{r['id']}",
            "from": r["claim_a"],
            "to": r["claim_b"],
            "type": "contradicts",
            "label": "contradicts",
            "rationale": r["rationale"],
        })

    # --- Supersession edges ---
    super_rows = conn.execute(
        f"""SELECT id, older_claim_id, newer_claim_id, rationale, active
            FROM claim_supersessions
            WHERE (older_claim_id IN ({ph}) OR newer_claim_id IN ({ph})) AND active = 1""",
        claim_ids + claim_ids,
    ).fetchall()
    for r in super_rows:
        if r["newer_claim_id"] not in node_id_set:
            nc = conn.execute(
                "SELECT id, canonical_text, status, claim_type, created_at FROM claims WHERE id = ?",
                (r["newer_claim_id"],),
            ).fetchone()
            if nc:
                text = nc["canonical_text"]
                add_node({
                    "id": nc["id"],
                    "label": (text[:55] + "…") if len(text) > 55 else text,
                    "type": "claim",
                    "status": nc["status"],
                    "finding_type": "superseded",
                    "full_text": text,
                    "claim_type": nc["claim_type"],
                    "created_at": nc["created_at"],
                })
        edges.append({
            "id": f"super_{r['id']}",
            "from": r["newer_claim_id"],
            "to": r["older_claim_id"],
            "type": "supersedes",
            "label": "supersedes",
            "rationale": r["rationale"],
        })

    # --- Reexamination task nodes ---
    reex_rows = conn.execute(
        f"""SELECT id, claim_id, trigger_type, impact, reason, status, priority, created_at
            FROM reexamination_queue WHERE claim_id IN ({ph})""",
        claim_ids,
    ).fetchall()
    for r in reex_rows:
        short_impact = (r["impact"][:28] + "…") if len(r["impact"]) > 28 else r["impact"]
        add_node({
            "id": r["id"],
            "label": f"Re-examine: {short_impact}",
            "type": "reexamination",
            "claim_id": r["claim_id"],
            "trigger_type": r["trigger_type"],
            "impact": r["impact"],
            "reason": r["reason"],
            "status": r["status"],
            "priority": r["priority"],
            "created_at": r["created_at"],
        })
        edges.append({
            "id": f"reex_{r['id']}",
            "from": r["id"],
            "to": r["claim_id"],
            "type": "flags",
            "label": "flags",
        })

    return {"nodes": nodes, "edges": edges, "meta": meta}


def build_events(case_id: str, conn: Any, since: str = "") -> dict[str, Any]:
    """Return the event feed for a case, filtered by a since-timestamp cursor."""
    all_events: list[dict[str, Any]] = []

    case_row = conn.execute(
        "SELECT id, name, status, created_at FROM research_cases WHERE id = ?", (case_id,)
    ).fetchone()
    if case_row:
        all_events.append({
            "id": f"syn_case_{case_id}",
            "timestamp": case_row["created_at"],
            "type": "case_created",
            "entity_type": "case",
            "entity_id": case_id,
            "actor": "user",
            "actor_role": "human",
            "summary": f"Case ‘{case_row['name']}’ created",
        })

    for r in conn.execute(
        "SELECT id, original_name, created_at FROM case_files WHERE case_id = ?", (case_id,)
    ).fetchall():
        all_events.append({
            "id": f"syn_file_{r['id']}",
            "timestamp": r["created_at"],
            "type": "file_uploaded",
            "entity_type": "file",
            "entity_id": r["id"],
            "actor": "user",
            "actor_role": "human",
            "summary": f"File ‘{r['original_name']}’ uploaded",
        })

    plan_ids: list[str] = []
    for r in conn.execute(
        "SELECT id, version, status, compiler, created_at FROM analysis_plans WHERE case_id = ? ORDER BY version",
        (case_id,),
    ).fetchall():
        plan_ids.append(r["id"])
        all_events.append({
            "id": f"syn_plan_{r['id']}",
            "timestamp": r["created_at"],
            "type": "plan_compiled",
            "entity_type": "plan",
            "entity_id": r["id"],
            "actor": r["compiler"] or "compiler",
            "actor_role": "tool",
            "summary": f"Analysis plan v{r['version']} compiled ({r['status']})",
        })

    run_ids: list[str] = []
    for i, r in enumerate(
        conn.execute(
            "SELECT id, status, started_at, completed_at FROM case_runs WHERE case_id = ? ORDER BY started_at",
            (case_id,),
        ).fetchall(),
        1,
    ):
        run_ids.append(r["id"])
        all_events.append({
            "id": f"syn_run_s_{r['id']}",
            "timestamp": r["started_at"],
            "type": "run_started",
            "entity_type": "run",
            "entity_id": r["id"],
            "actor": "orbita-engine",
            "actor_role": "tool",
            "summary": f"Analysis Run #{i} started",
        })
        if r["completed_at"]:
            all_events.append({
                "id": f"syn_run_e_{r['id']}",
                "timestamp": r["completed_at"],
                "type": "run_completed",
                "entity_type": "run",
                "entity_id": r["id"],
                "actor": "orbita-engine",
                "actor_role": "tool",
                "summary": f"Analysis Run #{i} {r['status']}",
            })

    claim_ids = [r["claim_id"] for r in conn.execute(
        "SELECT claim_id FROM case_claims WHERE case_id = ?", (case_id,)
    ).fetchall()]

    all_entity_ids = [case_id] + run_ids + plan_ids + claim_ids
    if all_entity_ids:
        ph = ",".join("?" * len(all_entity_ids))
        for r in conn.execute(
            f"""SELECT id, entity_type, entity_id, event_type, payload_json, actor, actor_role, created_at
                FROM events WHERE entity_id IN ({ph})
                ORDER BY created_at DESC LIMIT 500""",
            all_entity_ids,
        ).fetchall():
            try:
                payload = json.loads(r["payload_json"])
            except Exception:
                payload = {}
            summary = _event_summary(r["event_type"], r["entity_type"], r["entity_id"], payload)
            all_events.append({
                "id": f"db_{r['id']}",
                "timestamp": r["created_at"],
                "type": r["event_type"],
                "entity_type": r["entity_type"],
                "entity_id": r["entity_id"],
                "actor": r["actor"],
                "actor_role": r["actor_role"],
                "summary": summary,
            })

    if since:
        all_events = [e for e in all_events if e["timestamp"] > since]

    all_events.sort(key=lambda e: e["timestamp"], reverse=True)

    seen_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for e in all_events:
        if e["id"] not in seen_ids:
            seen_ids.add(e["id"])
            deduped.append(e)

    deduped = deduped[:100]
    cursor = deduped[0]["timestamp"] if deduped else since
    return {"events": deduped, "cursor": cursor}


def _event_summary(event_type: str, entity_type: str, entity_id: str, payload: dict[str, Any]) -> str:
    text = payload.get("canonical_text", "")
    short = (text[:55] + "…") if len(text) > 55 else text
    eid = entity_id[:18] + "…" if len(entity_id) > 20 else entity_id
    return {
        "CLAIM_CREATED": f"Claim created: ‘{short}’",
        "CLAIM_STATUS_UPDATED": f"Status → {payload.get('new_status','?')}: ‘{short}’",
        "CLAIM_COMMITTED": f"Claim committed: ‘{short}’",
        "CHECK_RECORDED": f"Check ‘{payload.get('name','?')}’ {'passed' if payload.get('passed') else 'failed'} on {eid}",
        "EVIDENCE_ADDED": f"Evidence attached ({payload.get('source_kind','?')}) to {eid}",
        "EVIDENCE_REVOKED": f"Evidence revoked from {eid}",
        "CONTRADICTION_ADDED": f"Contradiction recorded between two claims",
        "PROOF_ADDED": f"Derivation recorded for {eid}",
        "CLAIM_SUPERSEDES": f"Claim supersedes {payload.get('older_claim_id','?')[:16]}…",
        "REEXAMINATION_QUEUED": f"Re-examination queued: {payload.get('reason','')}",
        "REEXAMINATION_RESOLVED": f"Re-examination resolved for {eid}",
    }.get(event_type, f"{entity_type} {event_type.replace('_',' ').lower()}: {eid}")


# ---------------------------------------------------------------------------
# Graph visualization HTML page
# ---------------------------------------------------------------------------

GRAPH_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Orbita · Belief Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
:root{--bg:#0b1020;--panel:#121a2c;--panel2:#0e1525;--border:#2a3753;--text:#e9eef8;--dim:#7a8faf;--accent:#65d6ff;--accent2:#9ee7ff}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden;font-size:13px}
#hdr{background:var(--panel);border-bottom:1px solid var(--border);padding:9px 16px;display:flex;align-items:center;gap:10px;flex-shrink:0;min-height:44px}
#hdr-logo{font-size:14px;font-weight:700;color:var(--accent);letter-spacing:.05em;white-space:nowrap}
#hdr-logo span{color:#3b82f6}
#case-sel{background:var(--bg);border:1px solid var(--border);color:var(--text);padding:4px 10px;border-radius:6px;font-size:12px;min-width:200px;cursor:pointer}
#case-sel:focus{outline:none;border-color:var(--accent)}
#graph-info{font-size:11px;color:var(--dim);white-space:nowrap}
#status{margin-left:auto;font-size:11px;padding:3px 10px;border-radius:12px;background:#0e2010;color:#4ade80;border:1px solid #16a34a44;white-space:nowrap;transition:all .3s}
#status.offline{background:#200e0e;color:#f87171;border-color:#dc262644}
#status.connecting{background:#1e1a08;color:#fbbf24;border-color:#d9770644}
#main{display:flex;flex:1;overflow:hidden}
#feed{width:255px;min-width:255px;background:var(--panel2);border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;transition:width .25s}
#feed-hdr{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--dim);padding:8px 12px;border-bottom:1px solid var(--border);flex-shrink:0}
#feed-list{flex:1;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.ev{padding:7px 12px;border-bottom:1px solid #151e33;cursor:default}
.ev:hover{background:#141d30}
.ev-time{color:var(--dim);font-size:10px;margin-bottom:2px}
.ev-summary{color:var(--text);font-size:11px;line-height:1.45}
.ev-actor{color:var(--accent);font-size:10px;margin-top:2px;opacity:.75}
.ev-new{animation:flash-in .6s ease-out}
@keyframes flash-in{from{background:#1a2e1a}to{background:transparent}}
#graph-area{flex:1;position:relative;overflow:hidden;min-width:0}
#network{width:100%;height:100%}
#legend{position:absolute;bottom:10px;left:10px;background:rgba(18,26,44,.93);border:1px solid var(--border);border-radius:8px;padding:9px 12px;font-size:10px;display:flex;gap:14px;flex-wrap:wrap;max-width:480px;pointer-events:none}
.lg-grp h4{color:var(--dim);font-size:9px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}
.lg-item{display:flex;align-items:center;gap:4px;margin:2px 0;color:var(--text)}
.ld{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.ls{width:9px;height:9px;border-radius:2px;flex-shrink:0}
#loading{position:absolute;inset:0;background:var(--bg);display:flex;align-items:center;justify-content:center;flex-direction:column;gap:10px;z-index:20}
#loading.hidden{display:none}
.spin{width:28px;height:28px;border:2.5px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
#no-case{position:absolute;inset:0;display:none;align-items:center;justify-content:center;flex-direction:column;gap:10px;text-align:center;color:var(--dim);z-index:5}
#no-case.show{display:flex}
#no-case h2{font-size:22px;color:var(--text)}
#drawer{width:0;min-width:0;background:var(--panel);border-left:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;transition:width .22s}
#drawer.open{width:340px;min-width:340px}
#drw-hdr{padding:11px 12px;border-bottom:1px solid var(--border);flex-shrink:0;min-width:340px}
#drw-title{font-size:12px;font-weight:600;color:var(--text);line-height:1.45;word-break:break-word;padding-right:22px;position:relative}
#drw-close{position:absolute;top:0;right:0;background:none;border:none;color:var(--dim);cursor:pointer;font-size:20px;line-height:1;padding:0}
#drw-close:hover{color:var(--text)}
#drw-badge{font-size:10px;padding:2px 7px;border-radius:10px;display:inline-block;margin-top:5px}
.tabs{display:flex;border-bottom:1px solid var(--border);flex-shrink:0;min-width:340px}
.tb{flex:1;padding:7px 2px;background:none;border:none;color:var(--dim);cursor:pointer;font-size:11px;border-bottom:2px solid transparent;transition:all .15s;font-family:inherit}
.tb.on{color:var(--accent);border-bottom-color:var(--accent)}
.tb:hover:not(.on){color:var(--text)}
#drw-body{flex:1;overflow-y:auto;padding:11px 13px;min-width:340px;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.dr{margin-bottom:10px}
.dl{color:var(--dim);font-size:9px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px}
.dv{color:var(--text);line-height:1.5;font-size:12px}
.chk{display:flex;align-items:center;gap:6px;padding:5px 8px;border-radius:4px;margin:3px 0;background:var(--bg);font-size:11px}
.pass{color:#4ade80}.fail{color:#f87171}
.ev-box{padding:6px 8px;border-radius:4px;background:var(--bg);margin:4px 0;font-size:11px}
.tl-item{padding:5px 8px;border-radius:4px;margin:3px 0;background:var(--bg);font-size:11px}
.tl-meta{color:var(--dim);font-size:10px;margin-top:2px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
</style>
</head>
<body>
<div id="hdr">
  <div id="hdr-logo">ORBITA <span>·</span> Belief Graph</div>
  <select id="case-sel" onchange="switchCase(this.value)"><option value="">Select case…</option></select>
  <span id="graph-info"></span>
  <div id="status" class="connecting">● Connecting…</div>
</div>

<div id="main">
  <div id="feed">
    <div id="feed-hdr">Event Feed</div>
    <div id="feed-list"></div>
  </div>

  <div id="graph-area">
    <div id="loading"><div class="spin"></div><span style="color:var(--dim);font-size:12px">Loading belief graph…</span></div>
    <div id="no-case"><h2>Orbita Belief Graph</h2><p>Select a case above to visualize its epistemic memory.</p></div>
    <div id="network"></div>
    <div id="legend">
      <div class="lg-grp">
        <h4>Nodes</h4>
        <div class="lg-item"><div class="ld" style="background:#22c55e"></div>Claim – supported</div>
        <div class="lg-item"><div class="ld" style="background:#3b82f6"></div>Claim – committed</div>
        <div class="lg-item"><div class="ld" style="background:#f59e0b"></div>Claim – challenged</div>
        <div class="lg-item"><div class="ld" style="background:#ef4444"></div>Claim – refuted</div>
        <div class="lg-item"><div class="ld" style="background:#8b5cf6"></div>Evidence</div>
        <div class="lg-item"><div class="ld" style="background:#0ea5e9"></div>Analysis Run</div>
        <div class="lg-item"><div class="ld" style="background:#14b8a6"></div>Source File</div>
        <div class="lg-item"><div class="ld" style="background:#f97316"></div>Re-examination</div>
      </div>
      <div class="lg-grp">
        <h4>Edges</h4>
        <div class="lg-item"><span style="color:#22c55e;font-size:14px">—</span>&nbsp;supports</div>
        <div class="lg-item"><span style="color:#ef4444;font-size:14px">—</span>&nbsp;refutes</div>
        <div class="lg-item"><span style="color:#8b5cf6;font-size:14px">—</span>&nbsp;derived from</div>
        <div class="lg-item"><span style="color:#f59e0b;font-size:14px">—</span>&nbsp;supersedes</div>
        <div class="lg-item"><span style="color:#0ea5e9;font-size:14px">—</span>&nbsp;generates / tested</div>
        <div class="lg-item"><span style="color:#14b8a6;font-size:14px">—</span>&nbsp;source</div>
        <div class="lg-item"><span style="color:#f97316;font-size:14px">—</span>&nbsp;flags</div>
      </div>
    </div>
  </div>

  <div id="drawer">
    <div id="drw-hdr">
      <div id="drw-title">
        <span id="drw-title-text">Detail</span>
        <button id="drw-close" onclick="closeDrawer()">×</button>
      </div>
      <span id="drw-badge"></span>
    </div>
    <div class="tabs">
      <button class="tb on" data-tab="overview" onclick="switchTab('overview')">Overview</button>
      <button class="tb" data-tab="history" onclick="switchTab('history')">History</button>
      <button class="tb" data-tab="evidence" onclick="switchTab('evidence')">Evidence</button>
      <button class="tb" data-tab="impact" onclick="switchTab('impact')">Impact</button>
    </div>
    <div id="drw-body"></div>
  </div>
</div>

<script>
// ---- State ----
var caseId = null, network = null, nodesDS = null, edgesDS = null;
var selNodeId = null, selNodeData = null, curTab = 'overview';
var evCursor = '', graphTimer = null, evTimer = null, graphData = null;

// ---- Node/edge styling ----
var NODE_CLR = {
  claim_supported:  {background:'#22c55e',border:'#16a34a'},
  claim_committed:  {background:'#3b82f6',border:'#2563eb'},
  claim_challenged: {background:'#f59e0b',border:'#d97706'},
  claim_refuted:    {background:'#ef4444',border:'#dc2626'},
  claim_superseded: {background:'#6b21a8',border:'#581c87'},
  claim_pending:    {background:'#4b5568',border:'#374151'},
  claim_:           {background:'#4b5568',border:'#374151'},
  evidence:         {background:'#8b5cf6',border:'#7c3aed'},
  evidence_rev:     {background:'#3b2b6b',border:'#2b1b5b'},
  analysis_run:     {background:'#0ea5e9',border:'#0284c7'},
  source:           {background:'#14b8a6',border:'#0d9488'},
  reexamination:    {background:'#f97316',border:'#ea580c'},
};
var EDGE_CLR = {
  supports:     '#22c55e', refutes:      '#ef4444',
  derived_from: '#8b5cf6', supersedes:   '#f59e0b',
  contradicts:  '#ef4444', tested_by:    '#0ea5e9',
  generates:    '#0ea5e9', flags:        '#f97316',
  source_of:    '#14b8a6',
};
var STATUS_CLR = {
  supported:  {bg:'#14532d',fg:'#4ade80',brd:'#16a34a'},
  committed:  {bg:'#1e3a5f',fg:'#60a5fa',brd:'#2563eb'},
  challenged: {bg:'#451a03',fg:'#fcd34d',brd:'#d97706'},
  refuted:    {bg:'#450a0a',fg:'#f87171',brd:'#dc2626'},
  superseded: {bg:'#2e1065',fg:'#c084fc',brd:'#7c3aed'},
  pending:    {bg:'#1f2937',fg:'#9ca3af',brd:'#4b5563'},
};

// ---- Init vis.js ----
function initNetwork() {
  nodesDS = new vis.DataSet([]);
  edgesDS = new vis.DataSet([]);
  var opts = {
    nodes:{font:{color:'#e9eef8',size:11,face:'Inter,Segoe UI,Arial,sans-serif'},borderWidth:2,
           shadow:{enabled:true,color:'rgba(0,0,0,.45)',size:6,x:0,y:2}},
    edges:{width:1.5,selectionWidth:2.5,font:{color:'#7a8faf',size:10,background:'#0b1020'},
           smooth:{enabled:true,type:'curvedCW',roundness:.12}},
    physics:{solver:'forceAtlas2Based',forceAtlas2Based:{gravitationalConstant:-90,centralGravity:.012,
           springLength:130,springConstant:.05,damping:.55,avoidOverlap:.8},
           stabilization:{iterations:150,updateInterval:50}},
    interaction:{hover:true,tooltipDelay:150},
    layout:{improvedLayout:false},
  };
  network = new vis.Network(document.getElementById('network'),{nodes:nodesDS,edges:edgesDS},opts);
  network.on('click',function(p){
    if(p.nodes.length>0) openDrawer(p.nodes[0]);
    else closeDrawer();
  });
  network.on('stabilizationIterationsDone',function(){
    network.setOptions({physics:{stabilization:false}});
  });
}

// ---- Vis node/edge builders ----
function visNode(n) {
  var ck = n.type==='claim' ? 'claim_'+(n.status||'') : n.type==='evidence' ? (n.active===false?'evidence_rev':'evidence') : n.type;
  var clr = NODE_CLR[ck] || NODE_CLR['claim_'];
  var shapes = {claim:'ellipse',evidence:'box',analysis_run:'diamond',source:'triangleDown',reexamination:'star'};
  var lbl = (n.label||n.id||'');
  if(lbl.length>34) lbl=lbl.substring(0,32)+'…';
  return {id:n.id,label:lbl,shape:shapes[n.type]||'ellipse',color:clr,
          size:n.type==='analysis_run'?22:n.type==='source'?18:16,
          title:n.full_text||n.label||n.id,_raw:n};
}
function visEdge(e) {
  var c = EDGE_CLR[e.type]||'#4b5568';
  var dashed = e.type==='contradicts'||e.type==='supersedes';
  var bidir = e.type==='contradicts';
  return {id:e.id,from:e.from,to:e.to,label:e.label||e.type,
          color:{color:c,highlight:c,hover:c},dashes:dashed,
          arrows:{to:{enabled:true,scaleFactor:.55},from:{enabled:bidir,scaleFactor:.55}},_raw:e};
}

// ---- Data loading ----
async function loadCases() {
  try {
    var r = await fetch('/cases');
    if(!r.ok) throw 0;
    var d = await r.json();
    var sel = document.getElementById('case-sel');
    var prev = sel.value;
    var cases = d.cases||[];
    sel.innerHTML = '<option value="">Select case…</option>';
    cases.forEach(function(c){
      var o = document.createElement('option');
      o.value=c.id; o.textContent=c.name+' (• '+c.status+')';
      if(c.id===caseId) o.selected=true;
      sel.appendChild(o);
    });
    var params = new URLSearchParams(window.location.search);
    var urlCase = params.get('case_id');
    if(urlCase && caseId!==urlCase){sel.value=urlCase;switchCase(urlCase);}
    else if(!caseId && cases.length===1){sel.value=cases[0].id;switchCase(cases[0].id);}
    else if(!caseId){document.getElementById('loading').className='hidden';if(!cases.length)document.getElementById('no-case').className='show';}
    setStatus('live');
  } catch(e){console.error('loadCases:',e);setStatus('offline');document.getElementById('loading').className='hidden';}
}

function switchCase(id) {
  if(!id){document.getElementById('no-case').className='show';document.getElementById('loading').className='hidden';return;}
  document.getElementById('no-case').className='';
  caseId=id; evCursor='';
  var url=new URL(window.location); url.searchParams.set('case_id',id); window.history.replaceState({},'',url);
  stopPolling();
  document.getElementById('loading').className='';
  document.getElementById('feed-list').innerHTML='';
  loadGraph(); loadEvents(); startPolling();
}

async function loadGraph() {
  if(!caseId) return;
  try {
    var r = await fetch('/cases/'+caseId+'/graph');
    if(!r.ok) throw 0;
    graphData = await r.json();
    applyGraph(graphData);
    setStatus('live');
    document.getElementById('loading').className='hidden';
    var m=graphData.meta||{};
    document.getElementById('graph-info').textContent = (m.claim_count||0)+' claims · '+graphData.nodes.length+' nodes · '+graphData.edges.length+' edges';
  } catch(e){
    console.error('loadGraph:',e);setStatus('offline');
    document.getElementById('loading').className='hidden';
    if(!graphData){var nc=document.getElementById('no-case');nc.className='show';nc.innerHTML='<h2 style="color:var(--text);font-size:18px">Graph Unavailable</h2><p style="max-width:290px;margin-top:6px">Could not reach <code>/cases/{id}/graph</code>. The service may still be deploying, or the case ID may not exist in this database.<br><br><a href="javascript:void(0)" onclick="document.getElementById(\'no-case\').className=\'\';document.getElementById(\'loading\').className=\'\';loadGraph()" style="color:var(--accent)">Retry →</a></p>';}
  }
}

function applyGraph(d) {
  var vn=d.nodes.map(visNode), ve=d.edges.map(visEdge);
  var exN=new Set(nodesDS.getIds()), exE=new Set(edgesDS.getIds());
  var addN=[],updN=[],addE=[],updE=[];
  vn.forEach(function(n){(exN.has(n.id)?updN:addN).push(n);});
  ve.forEach(function(e){(exE.has(e.id)?updE:addE).push(e);});
  nodesDS.add(addN); nodesDS.update(updN); edgesDS.add(addE); edgesDS.update(updE);
  var newN=new Set(vn.map(function(n){return n.id;}));
  var newE=new Set(ve.map(function(e){return e.id;}));
  nodesDS.remove([...exN].filter(function(id){return !newN.has(id);}));
  edgesDS.remove([...exE].filter(function(id){return !newE.has(id);}));
  if(selNodeId && selNodeData){
    var updated=d.nodes.find(function(n){return n.id===selNodeId;});
    if(updated){selNodeData=updated;if(curTab==='overview')renderOverview(updated);}
  }
}

async function loadEvents() {
  if(!caseId) return;
  try {
    var r = await fetch('/cases/'+caseId+'/events?since='+encodeURIComponent(evCursor));
    if(!r.ok) return;
    var d = await r.json();
    if(d.events && d.events.length) { prependEvents(d.events); evCursor=d.cursor; }
  } catch(e){}
}

function prependEvents(evs) {
  var list=document.getElementById('feed-list');
  evs.forEach(function(ev){
    var div=document.createElement('div'); div.className='ev ev-new';
    var ts=''; try{ts=new Date(ev.timestamp).toLocaleTimeString();}catch(e){}
    div.innerHTML='<div class="ev-time">'+ts+'</div><div class="ev-summary">'+esc(ev.summary)+'</div><div class="ev-actor">'+esc(ev.actor)+'</div>';
    list.insertBefore(div,list.firstChild);
  });
  while(list.children.length>120) list.removeChild(list.lastChild);
}

// ---- Polling ----
function startPolling(){
  graphTimer=setInterval(loadGraph,5000);
  evTimer=setInterval(loadEvents,3000);
}
function stopPolling(){clearInterval(graphTimer);clearInterval(evTimer);}

// ---- Detail drawer ----
function openDrawer(nodeId) {
  var n = graphData && graphData.nodes.find(function(n){return n.id===nodeId;});
  if(!n) return;
  selNodeId=nodeId; selNodeData=n;
  document.getElementById('drawer').classList.add('open');
  document.getElementById('drw-title-text').textContent=n.full_text||n.label||nodeId;
  var sc = STATUS_CLR[n.status]||STATUS_CLR.pending;
  var badge=document.getElementById('drw-badge');
  badge.textContent=(n.type||'')+(n.status?' · '+n.status:'');
  badge.style.cssText='background:'+sc.bg+';color:'+sc.fg+';border:1px solid '+sc.brd+';font-size:10px;padding:2px 7px;border-radius:10px;display:inline-block;margin-top:5px';
  switchTab('overview');
}

function closeDrawer() {
  document.getElementById('drawer').classList.remove('open');
  selNodeId=null; selNodeData=null;
  if(network) network.unselectAll();
}

function switchTab(tab) {
  curTab=tab;
  document.querySelectorAll('.tb').forEach(function(b){b.classList.toggle('on',b.dataset.tab===tab);});
  if(!selNodeData) return;
  if(tab==='overview') renderOverview(selNodeData);
  else if(tab==='history') renderHistory(selNodeData);
  else if(tab==='evidence') renderEvidence(selNodeData);
  else if(tab==='impact') renderImpact(selNodeData);
}

function renderOverview(n) {
  var h='';
  if(n.type==='claim'){
    h+=row('Status','<span style="color:'+((STATUS_CLR[n.status]||STATUS_CLR.pending).fg)+'">'+esc(n.status||'—')+'</span>');
    h+=row('Finding type',esc(n.finding_type||'—'));
    h+=row('Claim type',esc(n.claim_type||'—'));
    h+=row('Statement','<span style="line-height:1.55">'+esc(n.full_text||'')+'</span>');
    h+=row('Created',fmt(n.created_at));
    h+='<div style="margin-top:8px"><a href="/claims/'+n.id+'/history" target="_blank">Full history →</a>&nbsp;&nbsp;<a href="/claims/'+n.id+'/impact" target="_blank">Impact →</a></div>';
  } else if(n.type==='evidence'){
    h+=row('Kind',esc(n.source_kind||'—'));
    h+=row('Active',n.active===false?'<span style="color:#f87171">revoked</span>':'yes');
    h+=row('Source','<span style="word-break:break-all;font-size:10px">'+esc(n.source_uri||'')+'</span>');
    h+=row('Excerpt','<em style="line-height:1.55">'+esc(n.excerpt||'')+'</em>');
    h+=row('Created',fmt(n.created_at));
  } else if(n.type==='analysis_run'){
    h+=row('Status',esc(n.status||'—'));
    h+=row('Started',fmt(n.started_at));
    h+=row('Completed',fmt(n.completed_at));
  } else if(n.type==='source'){
    h+=row('File',esc(n.label||'—'));
    h+=row('Kind',esc(n.artifact_kind||'—'));
    h+=row('Created',fmt(n.created_at));
  } else if(n.type==='reexamination'){
    h+=row('Status',esc(n.status||'—'));
    h+=row('Priority',esc(n.priority||'—'));
    h+=row('Impact',esc(n.impact||'—'));
    h+=row('Reason',esc(n.reason||'—'));
    h+=row('Created',fmt(n.created_at));
  } else {
    h='<pre style="font-size:10px;color:var(--dim);white-space:pre-wrap;word-break:break-all">'+esc(JSON.stringify(n,null,2))+'</pre>';
  }
  setBody(h);
}

async function renderHistory(n) {
  if(n.type!=='claim'){setBody('<p style="color:var(--dim)">History only available for claims.</p>');return;}
  setBody('<p style="color:var(--dim)">Loading…</p>');
  try {
    var r=await fetch('/claims/'+n.id+'/history');
    var d=await r.json();
    var tl=d.timeline||[];
    if(!tl.length){setBody('<p style="color:var(--dim)">No history events recorded.</p>');return;}
    var h='';
    tl.forEach(function(item){
      var icon = item.kind==='check' ? (item.event_type==='CHECK_PASSED'?'<span class="pass">✓</span>':'<span class="fail">✗</span>') : '○';
      var detail=item.detail||{};
      var score = detail.score!=null ? ' <span style="color:var(--dim)">score: '+parseFloat(detail.score).toFixed(4)+'</span>' : '';
      h+='<div class="tl-item">'+icon+' <strong>'+esc(item.event_type.replace(/_/g,' ').toLowerCase())+'</strong>'+score+'<div class="tl-meta">'+fmt(item.created_at)+' · '+esc(item.actor||'')+'</div></div>';
    });
    setBody(h);
  } catch(e){setBody('<p style="color:#f87171">Failed to load history.</p>');}
}

async function renderEvidence(n) {
  if(n.type!=='claim'){setBody('<p style="color:var(--dim)">Evidence only available for claims.</p>');return;}
  setBody('<p style="color:var(--dim)">Loading…</p>');
  try {
    var r=await fetch('/claims/'+n.id+'/history');
    var d=await r.json();
    var ev=d.evidence&&d.evidence[n.id]||[];
    if(!ev.length){setBody('<p style="color:var(--dim)">No evidence attached.</p>');return;}
    var h='';
    ev.forEach(function(e){
      var sc=e.stance==='support'?'#4ade80':'#f87171';
      var rev=e.active===false?'<span style="color:#f87171"> (revoked)</span>':'';
      h+='<div class="ev-box"><div style="display:flex;align-items:center;gap:6px"><span style="color:'+sc+'">'+esc(e.stance)+'</span>'+rev+'<span style="color:var(--dim)">'+esc(e.source_kind||'')+'</span></div>';
      h+='<div style="color:var(--dim);font-size:10px;margin-top:3px;word-break:break-all">'+esc(e.source_uri||'')+'</div>';
      if(e.excerpt) h+='<div style="margin-top:4px;font-style:italic">'+esc(e.excerpt)+'</div>';
      h+='</div>';
    });
    setBody(h);
  } catch(e){setBody('<p style="color:#f87171">Failed to load evidence.</p>');}
}

async function renderImpact(n) {
  if(n.type!=='claim'){setBody('<p style="color:var(--dim)">Impact only available for claims.</p>');return;}
  setBody('<p style="color:var(--dim)">Loading…</p>');
  try {
    var r=await fetch('/claims/'+n.id+'/impact');
    var d=await r.json();
    var deps=d.dependents||[];
    var reex=d.open_reexamination||[];
    var h='';
    if(deps.length){
      h+='<div class="dl">Downstream dependents ('+deps.length+')</div>';
      deps.forEach(function(id){
        var gn=graphData&&graphData.nodes.find(function(x){return x.id===id;});
        var label=gn?(gn.full_text||gn.label):id;
        h+='<div class="chk" style="cursor:pointer" onclick="jumpTo('+JSON.stringify(id)+')">▶&nbsp;<span>'+esc(label)+'</span></div>';
      });
    } else {
      h='<p style="color:var(--dim)">No downstream dependents.</p>';
    }
    if(reex.length){
      h+='<div class="dl" style="margin-top:10px">Open re-examinations</div>';
      reex.forEach(function(item){
        h+='<div class="ev-box"><strong>'+esc(item.impact)+'</strong><div style="color:var(--dim)">'+esc(item.reason)+'</div></div>';
      });
    }
    setBody(h);
  } catch(e){setBody('<p style="color:#f87171">Failed to load impact.</p>');}
}

function jumpTo(nodeId) {
  if(!network) return;
  network.focus(nodeId,{scale:1.3,animation:{duration:500,easingFunction:'easeInOutCubic'}});
  network.selectNodes([nodeId]);
  openDrawer(nodeId);
}

// ---- Utils ----
function row(lbl,val){return '<div class="dr"><div class="dl">'+lbl+'</div><div class="dv">'+val+'</div></div>';}
function setBody(h){document.getElementById('drw-body').innerHTML=h;}
function esc(s){if(!s)return '';return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function fmt(iso){if(!iso)return '—';try{return new Date(iso).toLocaleString();}catch(e){return iso;}}
function setStatus(s){
  var el=document.getElementById('status');
  if(s==='live'){el.textContent='● Live';el.className='';}
  else if(s==='offline'){el.textContent='● Offline';el.className='offline';}
  else {el.textContent='● Connecting…';el.className='connecting';}
}

// ---- Boot ----
initNetwork();
loadCases();
setInterval(loadCases,30000);
</script>
</body>
</html>"""
