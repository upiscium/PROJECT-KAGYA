"""Domain coordinators used by the main-loop compatibility facade."""

from kagya.runtime.coordinators.action import ActionCoordinator
from kagya.runtime.coordinators.chat import (
    ChatOrchestrationCoordinator,
    ChatResult,
    ChatRollbackScope,
    ChatStage,
    ChatTransactionCallbacks,
    ChatTransactionTrace,
)
from kagya.runtime.coordinators.experience import (
    ExperienceIntegrationCoordinator,
    ExperienceIntegrationResult,
)
from kagya.runtime.coordinators.identity import IdentityNarrativeCoordinator
from kagya.runtime.coordinators.motivation import MotivationGoalCoordinator
from kagya.runtime.coordinators.persistence import PersistenceCoordinator
from kagya.runtime.coordinators.planning import PlanDecisionCoordinator

__all__ = [
    "ActionCoordinator",
    "ChatOrchestrationCoordinator",
    "ChatResult",
    "ChatRollbackScope",
    "ChatStage",
    "ChatTransactionCallbacks",
    "ChatTransactionTrace",
    "ExperienceIntegrationCoordinator",
    "ExperienceIntegrationResult",
    "IdentityNarrativeCoordinator",
    "MotivationGoalCoordinator",
    "PersistenceCoordinator",
    "PlanDecisionCoordinator",
]
