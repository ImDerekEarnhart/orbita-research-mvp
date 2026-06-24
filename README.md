# Orbita Research MVP v0.1.0

Orbita Research MVP is a local service that turns uploaded research material into an explicit analysis plan, runs governed discovery and falsification checks, imports the resulting claims into persistent epistemic memory, and produces an expert-readable research dossier.

It combines:

- the Orbita Epistemic Runtime v1.5;
- the Orbita Discovery Kit engine;
- a new intake and profiling layer;
- Open Discovery when the research goal is blank;
- a reviewable and hash-bound analysis plan;
- persistent belief history and explicit supersession;
- forward dependency-collapse analysis;
- a non-destructive re-examination queue;
- a local browser interface and REST API;
- Markdown, HTML, and JSON research reports.

## The intended workflow

```text
Files + optional natural-language goal
              ↓
Safe ingestion and deterministic profiling
              ↓
Research compiler proposes a frozen plan
              ↓
Human or authorized AI reviews the plan
              ↓
Locked scout/confirmation discovery run
              ↓
Judge + baseline + held-out + cross-seed falsifiers
              ↓
Persistent claims, evidence, checks, proofs, contradictions
              ↓
Belief history + supersession + dependency collapse
              ↓
Expert-readable research dossier
```

A blank goal activates **Open Discovery**. The current v0.1 compiler searches parsed tables for candidate linear associations and group differences, while also reporting data-quality issues and inferred artifact guards.

## Install on Windows

Extract the ZIP, open PowerShell in the extracted folder, and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

Start the service:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_mvp.ps1
```

Leave that PowerShell window open. In another PowerShell window, open the interface:

```powershell
Start-Process "http://127.0.0.1:8010/"
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8010/docs
```

The service uses port `8010` so it can run alongside the earlier discovery API on port `8000`.

## Browser workflow

1. Create a case.
2. Enter a goal or leave it blank.
3. Upload a table and optional supporting documents.
4. Click **Compile research plan**.
5. Review the inferred roles, assumptions, candidates, and thresholds.
6. Click **Approve plan**.
7. Click **Run governed discovery**.
8. Open the generated research dossier.

## Supported uploads

First-class parsing in v0.1:

- CSV and TSV;
- Excel workbooks;
- JSON and JSONL record collections;
- Parquet when the local pandas installation has a Parquet engine;
- PDF text extraction;
- DOCX;
- TXT, Markdown, Python, R, and TeX text;
- Jupyter notebooks as JSON/text context;
- ZIP archives with path and size protections.

Every original upload is preserved and hashed. Unsupported files are preserved and marked unsupported rather than guessed at.

### Important v0.1 boundary

Automated hypothesis generation currently runs on **tabular datasets**. PDFs, documents, notebooks, and text files are preserved as provenance and source context, but they do not yet autonomously control statistical analysis or silently override the table.

## Open Discovery protections

When no goal is supplied, the service does not simply mine the entire table and call the largest correlation a discovery.

It:

1. creates a deterministic scout partition;
2. generates candidates only from that scout partition;
3. freezes the candidate list in the approved plan;
4. scores candidates on a locked confirmation partition;
5. repeats confirmation checks across bootstrap seeds;
6. records all failed candidates as well as survivors;
7. reports how many candidates were proposed and tested.

The default candidate status is therefore evidence within the current finite dataset, not universal proof or external replication.

## Person or AI operation

A person can use the browser. An external AI can use the same API.

### 1. Create a case

```powershell
$case = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8010/cases" `
  -ContentType "application/json" `
  -Body '{"name":"Immune dataset","goal":""}'
```

### 2. Upload a file

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8010/cases/$($case.id)/files" `
  -F "file=@C:\path\to\data.csv"
```

### 3. Let Orbita compile the plan

```powershell
$plan = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8010/cases/$($case.id)/compile" `
  -ContentType "application/json" `
  -Body '{"max_candidates":60}'
```

### 4. Approve and run

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8010/plans/$($plan.id)/approve" `
  -ContentType "application/json" `
  -Body '{"reviewer":"Derek"}'

$run = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8010/cases/$($case.id)/run" `
  -ContentType "application/json" `
  -Body (ConvertTo-Json @{ plan_id = $plan.id })
```

### 5. Open the report

```powershell
Start-Process "http://127.0.0.1:8010/cases/$($case.id)/report"
```

## External AI research compiler

An AI can inspect a case without direct filesystem access:

```text
GET /cases/{case_id}/context
```

It can then submit a complete plan:

```text
POST /cases/{case_id}/plans
```

The service validates that the selected dataset belongs to the case, candidate IDs are unique, and required plan fields exist. The plan must still be approved before execution unless the caller explicitly requests auto-approval.

A candidate currently supports these tested kinds:

```json
{
  "id": "linear:marker_a:response",
  "statement": "marker_a and response show a stable positive linear association.",
  "kind": "linear_association",
  "predictor": "marker_a",
  "outcome": "response",
  "expected_direction": "positive",
  "parents": []
}
```

or:

```json
{
  "id": "group:treatment:response",
  "statement": "response differs systematically across levels of treatment.",
  "kind": "group_difference",
  "group": "treatment",
  "outcome": "response",
  "parents": []
}
```

`parents` may name other candidate IDs in the same approved plan. After the run, Orbita records those links as explicit proof/derivation edges. Premises within one derivation are AND-connected; separate derivations are OR alternatives.

## Understanding the output

For the current table domain, the score is a held-out predictive `R²`-style score:

- `1.0` means nearly all confirmation-set variance was reproduced;
- `0.0` means no improvement over the mean baseline;
- negative values mean worse than the baseline.

The score is not a probability that the claim is true and is not a p-value.

A finding contains:

- the frozen candidate statement and payload;
- judge status and held-out score;
- baseline comparison;
- held-out falsifier result;
- cross-seed median and spread;
- final status;
- hash-chain links and SHA-256 receipt.

## Belief graph memory

Every imported finding receives a durable claim ID. The system permanently records:

- the source run and evidence receipt;
- support or refutation stance;
- each check passed or failed;
- claim status changes;
- derivation parents;
- contradictions;
- supersession links;
- dependency impacts;
- re-examination tasks.

### Reconstruct a claim’s complete history

```text
GET /claims/{claim_id}/history
```

The response includes:

- the full supersession chain;
- append-only claim events;
- evidence and source independence keys;
- checks that promoted or weakened the claim;
- contradictions;
- proofs and premises;
- descendants that depend on it;
- open re-examination tasks.

### Supersede a claim

```text
POST /claims/{claim_id}/supersede
```

Body:

```json
{
  "new_statement": "The relation holds only in untreated samples.",
  "rationale": "The replication failed in the treated subgroup."
}
```

The old claim remains in storage with status `superseded`. The new claim receives its own identity. The collapse analyzer walks forward through all dependent derivations.

### Dependency-collapse classifications

- `weakened_but_still_supported` — one route was lost but another complete route remains;
- `challenged` — support and active contradiction coexist;
- `unsupported_must_reexamine` — no valid support route remains;
- `recovered` — support was restored;
- `support_changed` — another meaningful transition occurred.

Nothing is automatically deleted. Impacted claims enter:

```text
GET /reexamination
```

A researcher can resolve a task through:

```text
POST /reexamination/{queue_id}/resolve
```

## Generated case artifacts

Each completed run creates a directory containing:

```text
discovery_ledger.jsonl
report/research_dossier.md
report/research_dossier.html
report/research_dossier.json
report/approved_plan.json
report/engine_result.json
```

The SQLite database also stores claims, evidence, attestations, proofs, contradictions, events, graph snapshots, supersessions, checks, case records, and the re-examination queue.

## Command-line demo

After installation:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1
```

Or directly:

```powershell
.\.venv\Scripts\orbita-mvp.exe `
  --db .\demo.db `
  --workspace .\demo_workspace `
  demo .\examples\marker_response.csv `
  --name "Open discovery demo"
```

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
```

## Honest limitations

This is a functional local MVP, not a universal autonomous scientist.

- Automatic analysis is currently limited to table profiling, linear associations, and group differences.
- It does not yet perform domain-specific causal inference, survival analysis, mixed-effects modeling, graph theorem proving, image analysis, or arbitrary time-series analysis.
- Supporting documents are context and provenance in v0.1 rather than a fully autonomous claim-extraction and evidence-adjudication loop.
- There is no built-in hosted LLM provider. An AI can use the public case/context/plan API, or a future provider adapter can be added.
- Authentication, multi-user isolation, encrypted object storage, quotas, and production deployment controls are not included in the local MVP.
- A researcher must still review the unit of analysis, missing-value semantics, derived columns, confounders, and domain interpretation.

The architecture is intentionally ready for additional domain plug-ins and an LLM research-compiler adapter without allowing model prose to bypass deterministic checks or persistent provenance.

## Reviewable plan revisions

The browser plan editor and API never overwrite an existing plan. Saving edits creates a new immutable plan version with a new SHA-256 hash:

```text
POST /plans/{plan_id}/revise
```

The revised plan must then be approved before execution. This preserves the original machine-generated proposal and the human- or AI-reviewed version.

## Direct belief-memory API

The combined MVP exposes the underlying belief lifecycle in addition to normal research cases:

```text
GET  /cases/{case_id}/claims
GET  /claims/{claim_id}/history
GET  /claims/{claim_id}/impact
POST /claims/{claim_id}/evidence
POST /claims/{claim_id}/derive
POST /claims/{claim_id}/supersede
POST /contradictions
POST /evidence/{evidence_id}/revoke
GET  /reexamination
POST /reexamination/{queue_id}/resolve
```

Adding evidence, derivations, contradictions, supersessions, or evidence revocations automatically recomputes the affected support closure. Meaningful changes are appended as events and placed in the re-examination queue. Existing claims and old plan versions are not deleted.

### Contradiction example

```json
{
  "claim_a": "clm_...",
  "claim_b": "clm_...",
  "rationale": "The two supported statements cannot both hold under the same scope."
}
```

Submit that body to `POST /contradictions`. The response includes dependency impacts, and each claim history retains the contradiction link and rationale.

## Local-service concurrency boundary

The bundled SQLite database is configured for the single-user local FastAPI service, with WAL mode and a busy timeout. Run one Orbita server process against a database file. A production multi-user deployment should use per-request database sessions or a server database such as PostgreSQL, plus authentication and tenant isolation.
