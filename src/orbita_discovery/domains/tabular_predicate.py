"""Declarative domain for testing Boolean hypotheses against JSON-like records."""
from __future__ import annotations

import random
from typing import Any

from ..core import Candidate
from ..safeexpr import UnsafeExpression, evaluate


class TabularPredicateDomain:
    name = "tabular_predicate"

    def __init__(
        self,
        records: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        baseline: float = 0.5,
        train_fraction: float = 0.7,
    ):
        if not records:
            raise ValueError("records must not be empty")
        if not candidates:
            raise ValueError("candidates must not be empty")
        if not 0.1 <= train_fraction <= 0.9:
            raise ValueError("train_fraction must be between 0.1 and 0.9")
        self.records = records
        self.specs = candidates
        self.baseline = float(baseline)
        self.train_fraction = train_fraction
        # Parse and test each expression once for early, clear failure.
        for spec in self.specs:
            if not all(k in spec for k in ("id", "statement", "expression")):
                raise ValueError("candidate requires id, statement, expression")
            try:
                evaluate(str(spec["expression"]), self.records[0])
            except Exception as exc:
                raise ValueError(f"Invalid expression for {spec.get('id')}: {exc}") from exc

    def propose(self):
        for spec in self.specs:
            yield Candidate(
                id=str(spec["id"]),
                statement=str(spec["statement"]),
                payload={"expression": str(spec["expression"])},
            )

    def evidence_for(self, c: Candidate) -> Any:
        return self.records

    def splits(self, evidence, seed: int):
        idx = list(range(len(evidence)))
        random.Random(seed).shuffle(idx)
        cut = max(1, min(len(idx) - 1, int(self.train_fraction * len(idx))))
        return [evidence[i] for i in idx[:cut]], [evidence[i] for i in idx[cut:]]

    def refit(self, c: Candidate, train) -> str:
        return str(c.payload["expression"])

    def score(self, c: Candidate, model: str, test) -> float:
        if not test:
            return 0.0
        successes = 0
        for record in test:
            try:
                successes += bool(evaluate(model, record))
            except (UnsafeExpression, KeyError, TypeError, ValueError, ZeroDivisionError):
                successes += 0
        return successes / len(test)

    def baseline_score(self, test) -> float:
        return self.baseline
