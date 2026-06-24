"""Deterministic governed and optimistic judges."""
from __future__ import annotations

from typing import Any

from .core import Candidate, Domain, Verdict
from .falsifiers import FittableDomain


class GatedJudge:
    name = "governed"

    def __init__(self, commit_at: float = 0.5, baseline_margin: float = 0.05):
        self.commit_at = commit_at
        self.baseline_margin = baseline_margin

    def judge(self, c: Candidate, evidence: Any, domain: Domain) -> Verdict:
        if not isinstance(domain, FittableDomain):
            return Verdict("unknown", 0.0, {"skipped": "domain not fittable"})
        train, test = domain.splits(evidence, seed=1)
        model = domain.refit(c, train)
        held = domain.score(c, model, test)
        baseline = domain.baseline_score(test)
        if held >= self.commit_at and held - baseline >= self.baseline_margin:
            status = "supported"
        elif held >= self.commit_at * 0.6:
            status = "provisional"
        else:
            status = "unknown"
        return Verdict(status, held, {"held_out": held, "baseline": baseline})


class OptimisticJudge:
    name = "naive"

    def __init__(self, commit_at: float = 0.5):
        self.commit_at = commit_at

    def judge(self, c: Candidate, evidence: Any, domain: Domain) -> Verdict:
        if not isinstance(domain, FittableDomain):
            return Verdict("unknown", 0.0, {"skipped": "domain not fittable"})
        model = domain.refit(c, evidence)
        score = domain.score(c, model, evidence)
        return Verdict("supported" if score >= self.commit_at else "unknown", score, {"in_sample": score})
