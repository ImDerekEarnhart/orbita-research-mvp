# Phase 2B — Observation Ledger + Counterexample Memory (Backend)

**Branch:** `saas/phase2b-observation-counterexamples` (base: `saas/phase2a-memory-graphs` @ 5ca8633)
Not merged to main/master. Production untouched. Staging retry pending for the populated case-delete fix.

## Files changed
- `src/orbita_mvp/observations.py` (new) — append-only per-case `observations.jsonl`
  under the case workspace; hash-chained entries (content_hash + previous_hash);
  no update/delete functions; `verify_chain()` detects tampering.
- `src/orbita_mvp/receipts.py` (new) — falsifier receipt convention:
  stage/status/killed/reason/metric + epistemic_effect ∈
  {supports, refutes, challenges, none, unknown}. A passed check is
  "supports" only when the finding committed; otherwise "none" — no kill ≠ proof.
- `src/orbita_mvp/storage.py` — `record_counterexample` (insert-only),
  `graph_counterexamples`, `case_counterexamples`, `graph_memory_summary`
  (verdict counts, counterexample counts, observation counts, dataset
  supports/refutes/challenges relations); counterexample indexes; `delete_case`
  now removes case-owned counterexamples and owned claim-memory rows before
  deleting case-owned claims/case row.
- `src/orbita_mvp/service.py` — observations at dataset import, run start,
  run receipts, run completed/failed; `_import_result(graph_id=…)` writes one
  counterexample per killed candidate (claim/case/graph/run/dataset linked,
  found_by = killing falsifier, failure_json = receipts, world_json = candidate
  statement+payload+dataset sha, minimal_known=false) and stamps
  `falsifier_receipts` into finding_detail.
- `src/orbita_mvp/api.py` — `/graphs/{id}/claims` now returns per-claim
  counterexample_count + `summary`; new `GET /graphs/{id}/summary` and
  `GET /graphs/{id}/counterexamples` (Basic auth; read-only, no mutation routes).
  Basic Auth now catches only malformed auth decoding, so downstream delete/storage
  errors are not mislabeled as 401 Unauthorized.
- `tests/test_phase2b_memory.py` (new)

## Behavior notes
- Ledger lives inside the case workspace → existing `delete_case` rmtree removes it.
- Counterexample effect: kill on `falsified_candidate` → refutes; any other
  kill (not_supported / inconclusive / functional_form / no-incremental-value /
  regime_dependent) → challenges.
- Legacy NULL-graph rows never appear in graph-scoped queries or summaries.
- No writes ever mutate claims/case_claims from the counterexample path.
- Pre-deploy review found a lifecycle bug where deleted-case counterexamples
  remained queryable by case_id/graph_id. Fixed in this branch; regression test
  proves `delete_case()` removes those rows and the observation workspace.
- Staging review found a populated Phase 2B delete blocker: real runs kept
  `mvp_claim_keys`/`claim_checks` rows for case-owned claims, causing a foreign-key
  failure when deleting those claims. The old auth middleware swallowed that
  downstream exception as a fake 401. Fixed with ordered memory-table cleanup and
  narrower auth exception handling.

## Tests (all pass)
- `test_phase2b_memory.py` — 13/13: ledger entries+fields, append-only+hash
  chain+tamper detection, delete cascade, counterexample writes on kills,
  no counterexample for committed claims, cross-graph exclusion, auth,
  effect mapping (incl. "passed ≠ supports"), receipts in finding_detail,
  claims summary + counts, per-graph summary scoping, and counterexample cleanup
  on case deletion. New API-level regression deletes a fully populated run with
  7 counterexamples through valid Basic Auth and proves graph/case memory is gone.
- Regression subset: data lifecycle + graph scoping + direct-access hardening
  16/16. py_compile clean.

## Out of scope (later phases)
Operator execution/benchmarking, self-improvement queues, chat translator,
external retrieval, shared/org graphs, deployment.
