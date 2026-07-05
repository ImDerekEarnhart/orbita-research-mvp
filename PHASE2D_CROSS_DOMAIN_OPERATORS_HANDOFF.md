# Phase 2D-A - Cross-Domain Operator Proposals (Backend)

Branch: `saas/phase2d-cross-domain-operators`
Base: Phase 2B staging-green backend `83af16268158231dd3a80f9a623ff5f729510242`

## Backend Code Changes
None for this bounded phase.

## Existing Backend APIs Reused
- `GET /graphs/{graph_id}/claims`
- `GET /graphs/{graph_id}/summary`
- `GET /graphs/{graph_id}/counterexamples`

These remain Basic Auth protected and graph-id scoped. Frontend `guardGraph` enforces user ownership before reading backend graph memory.

## Storage
No backend SQLite schema change. Operator proposals are stored in the frontend Postgres `operator_proposals` table because proposals are user/graph review objects, not backend-executed operators.

## Operator Semantics
Candidate-only. No claim statuses are changed. No operator execution or benchmark promotion is implemented.

## Tests
Backend tests unchanged from Phase 2B. Frontend tests cover proposal generation, ownership, storage, and graph routes.

## Staging
Backend redeploy is not required unless an unrelated deployment process needs both services aligned. Production untouched.

## Out of Scope
Chat translator, operator execution, benchmark promotion, external retrieval, blockchain/crypto, production launch.
