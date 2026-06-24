# Orbita Research MVP — Agent Guide

This is the Orbita epistemic research system. It ingests tabular data, generates mathematical
hypotheses, actively tries to falsify each one with three attacks, and commits surviving
discoveries to a persistent belief graph with full evidence provenance.

## Live Deployment

```
Base URL:   https://orbita-research-mvp-production.up.railway.app
Graph UI:   https://orbita-research-mvp-production.up.railway.app/graph
API Docs:   https://orbita-research-mvp-production.up.railway.app/docs
```

Persistent storage lives on a Railway volume at `/data/orbita_mvp.db`.

## The Discovery Workflow

Every research task follows this sequence:

```
1. POST /cases              → create a case, get case_id
2. POST /cases/{id}/files   → upload one or more CSV/data files
3. POST /cases/{id}/compile → generate analysis plan (candidate hypotheses)
4. POST /cases/{id}/run     → run discovery engine, falsify candidates, commit survivors
5. GET  /graph?case_id={id} → view the live belief graph
```

## Critical Constraints

**Goal must be blank for open discovery.** There is no LLM API key configured.
Setting a non-empty goal triggers guided mode which requires an LLM and will
produce 0 candidates. Always use `"goal": ""`.

**Log-transform numeric columns before upload** if you expect power-law relationships.
The discovery engine finds LINEAR associations between columns. A power law
`y = a * x^b` becomes linear only in log space: `log(y) = log(a) + b * log(x)`.
Include both raw and log-transformed columns in the CSV to get both linear
screening (raw) and power-law discovery (log). Name log columns `log_*`.

**auto_approve must be true** to run without a separate approval step:
`POST /cases/{id}/run` with body `{"auto_approve": true}`.

## API Reference

### Cases

```
POST   /cases                          Create case
       body: {"name": str, "goal": ""}
       returns: {id, name, status, ...}

GET    /cases                          List all cases
GET    /cases/{id}                     Get case detail (includes files, plans, runs)
GET    /cases/{id}/hint                AI-readable summary for external agents
```

### Files

```
POST   /cases/{id}/files               Upload a dataset
       multipart form-data: file=<upload>
       returns: {id, original_name, artifact_kind, parse_status, ...}
```

### Analysis Plans

```
POST   /cases/{id}/compile             Generate analysis plan from uploaded files
       body: {"max_candidates": 60}    (optional, default 60)
       returns: plan object with candidates array

GET    /cases/{id}/plans/{plan_id}     Get a specific plan
POST   /cases/{id}/plans/{plan_id}/approve
POST   /plans/{plan_id}/revise        Submit an AI-revised plan
POST   /cases/{id}/plans               Submit an externally generated plan
```

### Running Discovery

```
POST   /cases/{id}/run
       body: {"auto_approve": true, "plan_id": null}
       returns: run result with findings, surviving claims, report paths

GET    /cases/{id}/report              HTML research dossier (open in browser)
GET    /cases/{id}/download/markdown   Markdown report
GET    /cases/{id}/download/json       Machine-readable result
```

### Belief Graph

```
GET    /cases/{id}/graph               Graph nodes + edges for vis.js
       returns: {nodes: [...], edges: [...], meta: {...}}

GET    /cases/{id}/events              Event feed
       query: ?since=ISO_TIMESTAMP

GET    /cases/{id}/claims              All claims linked to this case
GET    /claims/{id}/history            Full provenance history of a claim
GET    /claims/{id}/impact             Downstream dependents of a claim
```

### Belief Memory (manual operations)

```
POST   /claims/{id}/evidence           Attach external evidence to a claim
POST   /claims/{id}/derive             Record a logical derivation between claims
POST   /claims/{id}/supersede          Replace a claim with an updated version
POST   /contradictions                 Mark two claims as contradicting each other
POST   /evidence/{id}/revoke           Revoke a piece of evidence
GET    /reexamination                  Queue of claims flagged for re-examination
POST   /reexamination/{id}/resolve     Resolve a re-examination item
```

## Falsification Pipeline

Each candidate hypothesis is attacked with three checks:
- **baseline**: must beat a no-op model by a margin (default 0.05)
- **held_out**: must generalize to rows withheld during candidate generation
- **cross_seed**: must be stable across 9 different random data splits

A claim is `supported` only if all three pass. `refuted` if any kill it.
On the /graph page: green = supported, red = refuted.

## Belief Graph Nodes

| Type | Shape | Color | Meaning |
|------|-------|-------|---------|
| claim | ellipse | green/red/amber/blue | A mathematical hypothesis |
| evidence | box | purple | Statistical evidence backing a claim |
| analysis_run | diamond | sky blue | One execution of the discovery engine |
| source | triangle | teal | An uploaded data file |
| reexamination | star | orange | A claim flagged for re-examination |

## Example Agent Tasks

### "Run a discovery on this CSV"
```
1. POST /cases  {"name": "...", "goal": ""}
2. POST /cases/{id}/files  (upload the CSV)
3. POST /cases/{id}/compile
4. POST /cases/{id}/run  {"auto_approve": true}
5. Open /graph?case_id={id} to show results
6. GET /cases/{id}/claims to list what was discovered
```

### "What did the last run find?"
```
GET /cases  → find the case
GET /cases/{id}/claims  → list surviving claims and their statuses
GET /cases/{id}/report  → full HTML dossier
```

### "Show the evidence behind a specific claim"
```
GET /claims/{claim_id}/history  → full provenance: events, checks, evidence, contradictions
```

### "Add my own evidence to a claim"
```
POST /claims/{claim_id}/evidence
body: {
  "source_uri": "https://...",
  "excerpt": "...",
  "source_kind": "literature",
  "independence_key": "paper:doi:...",
  "stance": "support",
  "confidence": 0.9
}
```

### "Mark two claims as contradicting each other"
```
POST /contradictions
body: {"claim_a": "claim_xxx", "claim_b": "claim_yyy", "rationale": "..."}
```

## Data Preparation Tips

For power-law discovery (most scientific domains):
```python
import pandas as pd, numpy as np
df = pd.read_csv("data.csv")
for col in ["x", "y", "z"]:          # numeric columns
    df[f"log_{col}"] = np.log10(df[col].clip(lower=1e-10))
df.to_csv("data_with_logs.csv", index=False)
```

Include both raw and log columns. The engine will screen all pairs.
Raw-column candidates usually get refuted (nonlinear in linear space).
Log-column candidates survive if a real power law exists.

## Local Development

```bash
pip install -e .
orbita-research-api          # starts on port 8010
# or
orbita-mvp --help            # CLI interface
```

Database path: `ORBITA_MVP_DB` env var (default `./orbita_mvp.db`)
Workspace path: `ORBITA_MVP_WORKSPACE` env var (default `./orbita_workspace`)
