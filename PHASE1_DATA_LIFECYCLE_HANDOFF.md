# Phase 1 Data Lifecycle Handoff

Branches:
- Frontend: `saas/phase1-data-lifecycle-hardening`
- Backend: `saas/phase1-data-lifecycle-hardening`

## Files Changed
- Frontend: `server.js`, `lib/dataLifecycle.js`, `tests/data_lifecycle.test.js`, `tests/test_data_lifecycle_http.js`
- Backend: `src/orbita_mvp/api.py`, `src/orbita_mvp/service.py`, `src/orbita_mvp/storage.py`, `tests/test_data_lifecycle.py`

## Backend Delete Behavior
- Added Basic-auth-protected `DELETE /cases/{case_id}`.
- Existing case: removes case DB rows and safely removes `workspace/cases/{case_id}` artifacts.
- Missing case: `404` with safe generic detail.
- `/health` remains public.

## Frontend Delete Behavior
- `DELETE /api/orbita/cases/:caseId` still requires User A/User B ownership guard.
- Frontend now calls backend delete first.
- Frontend ownership/job/resource rows are cleaned only after backend deletion succeeds.
- Backend delete failure returns safe partial-failure response instead of false success.

## Account Delete Behavior
- `POST /api/user/delete` verifies password, deletes all owned backend cases first, then cleans frontend rows/anonymizes user.
- If any backend case deletion fails, account deletion returns a safe partial-failure response.
- Admin user deletion follows the same backend-first behavior.

## recordResource Behavior
- Upload/compile/run no longer swallow `recordResource` failures.
- User gets a safe `500` if ownership/resource recording fails.
- Run queue path removes the queued job/resource record on local failure.
- Backend actions that already succeeded may leave backend artifacts; this is reported instead of hidden.

## Export Status
- Added simple `GET /api/user/export`.
- Exports only owned case/resource metadata plus case/report/download/graph links.
- Does not include other users' data or raw backend artifacts.

## Tests Added
- Backend local delete/auth/artifact-preservation tests.
- Frontend lifecycle helper unit tests.
- Live staging HTTP lifecycle tests using generated users and real sessions.

## Local Results
- Backend: `python -m pytest tests/test_data_lifecycle.py tests/test_backend_direct_access_hardening.py tests/test_api.py tests/test_tamper_endpoint_security.py` -> 17 passed.
- Frontend: `node --check server.js` -> passed.
- Frontend: `node --test tests/auth.test.js tests/data_lifecycle.test.js tests/upload_safety.test.js` -> 39 passed.

## Staging Results
- Backend staging deploy: `7869c6a4-0ec3-417d-a629-29d6132a24ff` -> SUCCESS.
- Frontend staging deploy: `d0f1487b-15c2-4f7f-84ce-cf21d65998a6` -> SUCCESS.
- `node --test tests/test_data_lifecycle_http.js` -> 4 passed.
- `node --test tests/test_upload_safety_http.js` -> 6 passed.
- Direct backend DELETE probe: unauth `401`, authed missing case `404`, health `200`.

## Commits
- Backend code: `c4139ad`
- Frontend code: `7bb3969`
- Frontend live HTTP tests: `d97c18e`

## Remaining Launch Blockers
- Railway private networking/internal backend access.
- Credential rotation before production.
- Disable/restrict backend docs in production.
- Production env setup.
- `safeusi.com` connection.
- Final production launch checklist.

## Exact Next Step
Run production-infra hardening: private backend URL, credential rotation, backend docs restriction, production env setup, then production deploy/domain checklist.
