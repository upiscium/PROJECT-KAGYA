"""Motivation and goal coordination boundary."""

from typing import Callable

from kagya.motivation import Goal, GoalManager, GoalStatus, MotivationDynamics


class MotivationGoalCoordinator:
    def __init__(
        self,
        goals: GoalManager,
        motivations: MotivationDynamics,
        *,
        persist: Callable[[], None],
    ) -> None:
        self._goals = goals
        self._motivations = motivations
        self._persist = persist

    def resolve_goal_motivation(self, goal: Goal, status: GoalStatus) -> None:
        if status in {GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED}:
            self._motivations.resolve_goal(
                goal.goal_id, success=status == GoalStatus.COMPLETED
            )
        self._persist()

    def list_active_goal_ids(self) -> tuple[str, ...]:
        return tuple(goal.goal_id for goal in self._goals.list_goals(GoalStatus.ACTIVE))
