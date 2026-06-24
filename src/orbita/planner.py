from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import RiskLevel, SupportState
from .support import SupportEngine


RISK_COST = {
    RiskLevel.LOW: 0.5,
    RiskLevel.MEDIUM: 4.0,
    RiskLevel.HIGH: 12.0,
}


@dataclass(slots=True)
class CandidateStep:
    intent: str
    action_type: str
    args: dict[str, Any]
    required_claims: list[str] = field(default_factory=list)
    obligations: list[dict[str, Any]] = field(default_factory=list)
    risk: RiskLevel = RiskLevel.LOW


@dataclass(slots=True)
class CandidatePlan:
    name: str
    goal: str
    steps: list[CandidateStep]


@dataclass(slots=True)
class PlanScore:
    name: str
    total: float
    unsupported_claims: list[str]
    challenged_claims: list[str]
    missing_obligations: list[int]
    risk_cost: float
    explanation: list[str]


class EpistemicPlanSelector:
    """Select the plan with the lowest measurable epistemic debt and action risk."""

    def __init__(self, support_engine: SupportEngine):
        self.support_engine = support_engine

    def score(self, plan: CandidatePlan) -> PlanScore:
        unsupported: list[str] = []
        challenged: list[str] = []
        missing_obligations: list[int] = []
        risk_cost = 0.0
        explanation: list[str] = []

        for idx, step in enumerate(plan.steps):
            risk_cost += RISK_COST[step.risk]
            if not step.obligations:
                missing_obligations.append(idx)
            for claim_id in step.required_claims:
                state = self.support_engine.evaluate(claim_id).state
                if state in {SupportState.UNKNOWN, SupportState.UNSUPPORTED}:
                    unsupported.append(claim_id)
                elif state == SupportState.CHALLENGED:
                    challenged.append(claim_id)

        total = (
            10.0 * len(set(unsupported))
            + 6.0 * len(set(challenged))
            + 3.0 * len(missing_obligations)
            + risk_cost
        )
        if unsupported:
            explanation.append(f"{len(set(unsupported))} required claim(s) lack support")
        if challenged:
            explanation.append(f"{len(set(challenged))} required claim(s) are challenged")
        if missing_obligations:
            explanation.append(f"{len(missing_obligations)} step(s) have no verification obligations")
        explanation.append(f"Risk cost: {risk_cost:.1f}")

        return PlanScore(
            name=plan.name,
            total=total,
            unsupported_claims=sorted(set(unsupported)),
            challenged_claims=sorted(set(challenged)),
            missing_obligations=missing_obligations,
            risk_cost=risk_cost,
            explanation=explanation,
        )

    def choose(self, plans: list[CandidatePlan]) -> tuple[CandidatePlan, list[PlanScore]]:
        if not plans:
            raise ValueError("At least one candidate plan is required")
        scored = [(plan, self.score(plan)) for plan in plans]
        scored.sort(key=lambda pair: (pair[1].total, len(pair[0].steps), pair[0].name))
        return scored[0][0], [score for _, score in scored]
