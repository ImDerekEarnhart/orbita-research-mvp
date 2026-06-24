"""Shared service functions used by the CLI and REST API."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .core import Engine, Ledger, finding_to_dict, survivors
from .falsifiers import BaselineFalsifier, CrossSeedFalsifier, HeldOutFalsifier
from .judges import GatedJudge, OptimisticJudge
from .registry import create_domain


def execute_run(
    *,
    domain_name: str,
    domain_config: dict[str, Any] | None = None,
    judge_name: str = "governed",
    commit_at: float = 0.5,
    baseline_margin: float = 0.05,
    falsifier_config: dict[str, Any] | None = None,
    output_dir: str | Path = "runs",
    run_id: str | None = None,
    progress=None,
) -> dict[str, Any]:
    run_id = run_id or uuid.uuid4().hex[:12]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / f"{run_id}.jsonl"

    domain = create_domain(domain_name, domain_config)
    if judge_name == "governed":
        judge = GatedJudge(commit_at=commit_at, baseline_margin=baseline_margin)
    elif judge_name == "naive":
        judge = OptimisticJudge(commit_at=commit_at)
    else:
        raise ValueError("judge must be 'governed' or 'naive'")

    fc = falsifier_config or {}
    falsifiers = [
        BaselineFalsifier(margin=float(fc.get("baseline_margin", 0.05))),
        HeldOutFalsifier(min_score=float(fc.get("held_out_min", 0.3))),
        CrossSeedFalsifier(
            seeds=int(fc.get("cross_seed_count", 7)),
            min_median=float(fc.get("cross_seed_min", 0.3)),
            max_spread=fc.get("cross_seed_max_spread"),
        ),
    ]
    engine = Engine(judge, falsifiers, Ledger(ledger_path))
    engine.run(domain, progress=progress)
    survivor_rows = survivors(engine.ledger)
    return {
        "run_id": run_id,
        "domain": domain_name,
        "judge": judge_name,
        "ledger_path": str(ledger_path.resolve()),
        "candidate_count": len(engine.ledger.entries),
        "survivor_count": len(survivor_rows),
        "survivor_ids": [x.candidate.id for x in survivor_rows],
        "findings": [finding_to_dict(x) for x in engine.ledger.entries],
    }
