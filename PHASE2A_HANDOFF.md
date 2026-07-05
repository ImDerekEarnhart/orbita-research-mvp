# Phase 2A — Memory Graph Foundation (Backend)

**Branch:** `saas/phase2a-memory-graphs` (base: `staging/phase1-hardened` @ eda2a81)
Not pushed, not deployed, not merged.

## Files changed
- `src/orbita_mvp/storage.py` — additive `_migrate()`: case_claims gains nullable
  graph_id / origin_json / epistemic_status (+ graph index); new empty
  counterexamples table (id, claim_id, case_id, graph_id, run_id, dataset_id,
  world/measurements/failure JSON, found_by, minimal_known, created_at).
  New: stamp_run_claims (post-import provenance stamp), graph_claims (scoped query).
- `src/orbita_mvp/service.py` — run_case accepts graph_id; after import, one
  UPDATE stamps this run's claims with graph_id + origin_json
  {dataset_ids, engine_version, plan_hash, operators: []}. Engine call sites untouched.
- `src/orbita_mvp/api.py` — RunRequest.graph_id; GET /graphs/{graph_id}/claims
  (Basic-auth like all non-public routes; excludes legacy NULL-graph rows).
- `tests/test_graph_scoping.py`

## Behavior notes
- Legacy claims keep graph_id NULL — unaffected, still served by case queries,
  never returned by graph-scoped queries.
- epistemic_status is passive (written nowhere) — reserved for Phase 2B statuses.
- counterexamples: no writes anywhere in 2A (Phase 2B counterexample memory).
- operators list in origin_json stays [] until operator registry executes (2D).

## Tests (all pass)
- `test_graph_scoping.py` — 6/6: stamped claims returned, cross-graph excluded,
  legacy NULL rows unaffected+excluded, 401 without/with wrong Basic auth,
  counterexamples table exists+empty, new columns present.
- Regression: test_data_lifecycle + test_backend_direct_access_hardening →
  16/16 combined. py_compile clean.

## Out of scope (later phases)
Observation ledger, counterexample writes/screening, operator execution,
gauntlet stages, curriculum queues, chat, external retrieval, deployment.
