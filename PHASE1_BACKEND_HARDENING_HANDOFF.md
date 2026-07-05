# Phase 1 Backend Direct-Access Hardening Handoff

Branch: `saas/phase1-backend-direct-access-hardening`

## Files Changed
- `src/orbita_mvp/api.py`
- `src/orbita_mvp/upload_safety.py`
- `tests/test_backend_direct_access_hardening.py`
- `tests/test_api.py`
- `tests/test_tamper_endpoint_security.py`

## Basic Auth Behavior
- `/health`, `/healthz`, and `/ui/*` remain public.
- Sensitive routes now fail closed when Basic Auth credentials are missing.
- Unauthenticated local/test mode requires an explicit local/test environment value.
- Wrong Basic Auth returns `401`; correct Basic Auth reaches intended routes.

## Upload Rules
- Allows CSV only, with final `.csv` extension.
- Max file size: 50 MiB.
- Rejects path filenames, absolute paths, leading dots, `..`, colon, slash, backslash, controls, unsupported characters.
- Rejects executable/script/archive extensions and double-extension traps.
- Rejects dangerous MIME types; MIME is not trusted alone.
- Rejects PE, ELF, ZIP/JAR/XLSX, gzip, PDF, PNG/JPEG, RAR/7z, shebang, HTML/script starts, NUL bytes, and binary controls.
- Invalid upload returns `400`; oversized upload returns `413`.

## Tests Added
- Basic Auth missing/wrong/correct coverage.
- `/health` public coverage.
- Valid tiny CSV accepted.
- Unsafe filename matrix rejected.
- Unsafe MIME/content sniffing rejected.
- Oversized upload rejected.

## Local Results
- `python -m pytest tests/test_backend_direct_access_hardening.py` -> 6 passed.
- `python -m pytest tests/test_api.py tests/test_tamper_endpoint_security.py` -> 7 passed.
- `python -m py_compile src/orbita_mvp/api.py src/orbita_mvp/upload_safety.py tests/test_backend_direct_access_hardening.py tests/test_api.py tests/test_tamper_endpoint_security.py` -> passed.

## Staging Results
- Pending backend staging deploy and direct backend probes.

## Remaining Risks
- Railway private networking/internal backend access is still pending.
- Backend Basic credentials should be rotated before production.
- Backend `/docs` should be disabled or restricted in production.
- Direct backend cleanup/delete endpoint remains unavailable.
- Silent `recordResource` swallow remains a frontend/backend integration gap.
- Backend-side deletion/export and production hardening checklist remain pending.

## Exact Next Step
Commit, deploy this backend branch to staging only, then verify direct backend unsafe uploads are rejected while valid CSV and Basic Auth behavior still work.
