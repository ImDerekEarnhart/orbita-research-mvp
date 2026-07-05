"""Phase 2B observation ledger — append-only, per-case, hash-chained.

Each case workspace gets an ``observations.jsonl`` file. Entries are only ever
appended (this module deliberately exposes no update or delete operation), and
each entry carries a content hash plus the previous entry's hash so any
after-the-fact edit is detectable. The file lives inside the case workspace,
so the existing case-deletion cascade (``CaseStore.delete_case`` →
``shutil.rmtree(case_dir)``) removes it with everything else.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LEDGER_FILENAME = "observations.jsonl"

# Kinds recorded in Phase 2B. Free-form strings are accepted; these are the
# ones the service writes.
KIND_DATASET_IMPORTED = "dataset_imported"
KIND_RUN_STARTED = "run_started"
KIND_RUN_COMPLETED = "run_completed"
KIND_RUN_FAILED = "run_failed"
KIND_RUN_RECEIPTS = "run_receipts"


def engine_version() -> str:
    return os.getenv("GIT_COMMIT_SHA", os.getenv("RAILWAY_GIT_COMMIT_SHA", "unknown"))


def _ledger_path(case_dir: Path) -> Path:
    return Path(case_dir) / LEDGER_FILENAME


def _canonical(entry: dict[str, Any]) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _last_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                last = line
    if last is None:
        return None
    try:
        return json.loads(last).get("content_hash")
    except json.JSONDecodeError:
        return None


def record_observation(
    case_dir: Path,
    *,
    case_id: str,
    source: str,
    kind: str,
    graph_id: str | None = None,
    dataset_ids: list[str] | None = None,
    run_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one observation to the case ledger and return the stored entry."""
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    path = _ledger_path(case_dir)
    entry: dict[str, Any] = {
        "observation_id": f"obs_{uuid.uuid4().hex[:16]}",
        "case_id": case_id,
        "graph_id": graph_id,
        "dataset_ids": list(dataset_ids or []),
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "kind": kind,
        "engine_version": engine_version(),
        "payload": payload or {},
        "previous_hash": _last_hash(path),
    }
    entry["content_hash"] = hashlib.sha256(_canonical(entry).encode("utf-8")).hexdigest()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_canonical(entry) + "\n")
    return entry


def read_observations(case_dir: Path) -> list[dict[str, Any]]:
    path = _ledger_path(Path(case_dir))
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def observation_count(case_dir: Path) -> int:
    return len(read_observations(case_dir))


def verify_chain(case_dir: Path) -> bool:
    """True when every entry's previous_hash matches its predecessor's content_hash."""
    entries = read_observations(Path(case_dir))
    prev: str | None = None
    for entry in entries:
        if entry.get("previous_hash") != prev:
            return False
        recorded = entry.get("content_hash")
        stripped = {k: v for k, v in entry.items() if k != "content_hash"}
        if hashlib.sha256(_canonical(stripped).encode("utf-8")).hexdigest() != recorded:
            return False
        prev = recorded
    return True
