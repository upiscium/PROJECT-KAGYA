"""Plan and decision coordination boundary."""

from typing import Callable

from kagya.decision import DecisionRecord, DecisionStatus, DecisionStore
from kagya.planning import Plan, PlanCandidate, PlanStore


class PlanDecisionCoordinator:
    def __init__(
        self,
        plans: PlanStore,
        decisions: DecisionStore,
        *,
        persist: Callable[[], None],
    ) -> None:
        self._plans = plans
        self._decisions = decisions
        self._persist = persist

    def create_plan(self, candidate: PlanCandidate, *, actor_id: str) -> Plan:
        plan = self._plans.create(candidate, actor_id=actor_id)
        self._persist()
        return plan

    def list_decisions(
        self, status: DecisionStatus | None = None
    ) -> list[DecisionRecord]:
        return self._decisions.list_records(status)
