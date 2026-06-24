"""Core domain-agnostic governed discovery and falsification loop."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

STATUS = ["refuted", "unknown", "provisional", "challenged", "supported"]


@dataclass
class Candidate:
    id: str
    statement: str
    payload: dict[str, Any] = field(default_factory=dict)
    parents: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    status: str
    score: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUS:
            raise ValueError(f"Unknown verdict status: {self.status}")


@dataclass
class Falsification:
    name: str
    killed: bool
    metric: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Finding:
    candidate: Candidate
    verdict: Verdict
    falsifications: list[Falsification]
    final_status: str
    survived: list[str]
    sha256: str = ""
    prev: str = ""
    ts: float = 0.0


@runtime_checkable
class Domain(Protocol):
    name: str
    def propose(self) -> Iterable[Candidate]: ...
    def evidence_for(self, c: Candidate) -> Any: ...


@runtime_checkable
class Judge(Protocol):
    name: str
    def judge(self, c: Candidate, evidence: Any, domain: Domain) -> Verdict: ...


@runtime_checkable
class Falsifier(Protocol):
    name: str
    def attempt(self, c: Candidate, evidence: Any, domain: Domain) -> Falsification: ...


def _finding_body(f: Finding) -> dict[str, Any]:
    return {
        "candidate": asdict(f.candidate),
        "verdict": asdict(f.verdict),
        "falsifications": [asdict(x) for x in f.falsifications],
        "final_status": f.final_status,
        "survived": f.survived,
        "prev": f.prev,
    }


class Ledger:
    """Append-only hash-chained JSONL record for one run."""

    def __init__(self, path: str | Path | None = None, truncate: bool = True):
        self.path = Path(path) if path else None
        self.entries: list[Finding] = []
        self._prev = "GENESIS"
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if truncate:
                self.path.write_text("", encoding="utf-8")

    def append(self, f: Finding) -> Finding:
        f.ts = time.time()
        f.prev = self._prev
        body = _finding_body(f)
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        f.sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self._prev = f.sha256
        self.entries.append(f)
        if self.path:
            record = {**body, "sha256": f.sha256, "ts": round(f.ts, 3)}
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        return f


def resolve_status(verdict: Verdict, falsifications: list[Falsification]) -> tuple[str, list[str]]:
    survived = [x.name for x in falsifications if not x.killed]
    if any(x.killed for x in falsifications):
        return "refuted", survived
    return verdict.status, survived


class Engine:
    def __init__(self, judge: Judge, falsifiers: list[Falsifier], ledger: Ledger | None = None):
        self.judge = judge
        self.falsifiers = falsifiers
        self.ledger = ledger or Ledger()

    def run(self, domain: Domain, progress: Callable[[str], None] | None = None) -> Ledger:
        for candidate in domain.propose():
            evidence = domain.evidence_for(candidate)
            verdict = self.judge.judge(candidate, evidence, domain)
            attacks = [f.attempt(candidate, evidence, domain) for f in self.falsifiers]
            final_status, survived_attacks = resolve_status(verdict, attacks)
            finding = Finding(candidate, verdict, attacks, final_status, survived_attacks)
            self.ledger.append(finding)
            if progress:
                tag = "KILLED" if final_status == "refuted" else final_status.upper()
                progress(f"{candidate.id:<28} {tag:<11} score={verdict.score:+.3f}")
        return self.ledger


def survivors(ledger: Ledger) -> list[Finding]:
    accepted = {"supported", "challenged", "provisional"}
    out = [
        f for f in ledger.entries
        if f.final_status in accepted and all(not x.killed for x in f.falsifications)
    ]
    return sorted(out, key=lambda f: f.verdict.score, reverse=True)


def finding_to_dict(f: Finding) -> dict[str, Any]:
    return {**_finding_body(f), "sha256": f.sha256, "ts": f.ts}


def verify_ledger(path: str | Path) -> tuple[bool, list[str]]:
    """Recompute every content hash and previous-link in a JSONL ledger."""
    errors: list[str] = []
    previous = "GENESIS"
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        if record.get("prev") != previous:
            errors.append(f"line {line_no}: prev hash mismatch")
        body = {k: record[k] for k in (
            "candidate", "verdict", "falsifications", "final_status", "survived", "prev"
        )}
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if record.get("sha256") != actual:
            errors.append(f"line {line_no}: content hash mismatch")
        previous = record.get("sha256", "")
    return not errors, errors
