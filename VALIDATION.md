# MVP validation record

Validated in the build environment on 2026-06-23.

## Integrated sources

- Orbita Epistemic Runtime v1.5 archive SHA-256:
  `34dd3a69c0ab78ac0183fb0e7e96c5ee158a292d605f756624019d49ad413757`
- Orbita Discovery Kit v0.2.0 archive SHA-256:
  `b60aec5e0cb47f294bb86bd9eb641f2ead2e200de89725f67dafbc277213c6fd`

The `src/orbita` runtime source was copied from the supplied v1.5 archive. The database connection was configured for the single-process FastAPI service by enabling cross-thread access, WAL mode, and a 30-second busy timeout. The `src/orbita_discovery` source matches the supplied Discovery Kit source.

## Tests passed

- 6 integrated MVP and API tests:
  - blank-goal Open Discovery from upload through report;
  - browser/API case workflow;
  - immutable plan revision and re-hashing;
  - supersession history and re-examination;
  - alternate derivation preserving support;
  - contradiction propagation.
- 11 upstream epistemic graph, event/policy, and hypergraph tests.
- Discovery Kit smoke test.
- Live local Uvicorn smoke check:
  - `/health` returned 200;
  - `/` returned the browser interface.
- End-to-end demo generated a discovery ledger, persistent claims, graph snapshot, and Markdown/HTML/JSON dossier.

## Scientific boundary

Passing software tests means the mechanisms execute as specified. It does not validate any domain finding produced from a user's data. Every generated finding remains conditional on input quality, approved assumptions, selected tests, and independent replication.
