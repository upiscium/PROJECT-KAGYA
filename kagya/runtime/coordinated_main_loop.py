"""Dependency wiring for the coordinated runtime main loop."""

from datetime import UTC, datetime
from typing import Any, Callable, Protocol

from kagya.agency import (
    AgencyAttributionStore,
)
from kagya.body import EmotionEngineAllostasis, EmotionState
from kagya.attention import (
    AttentionSystem,
)
from kagya.belief import (
    BeliefStore,
)
from kagya.cognition import (
    CognitiveAppraiser,
    SurprisalCalculator,
    ValueConflictDefinition,
    ValueState,
    ValueSystem,
)
from kagya.config import Settings
from kagya.counterfactual import (
    CounterfactualStore,
)
from kagya.decision import (
    DecisionStore,
)
from kagya.identity import (
    EndorsementStatus,
    NarrativeSelf,
    OriginActor,
    OriginInputKind,
    SelfModel,
    new_identity_origin,
)
from kagya.experience import (
    ExperienceStore,
)
from kagya.feedback import (
    FeedbackStore,
)
from kagya.memory import DualMemorySystem
from kagya.metacognition import Metacognition
from kagya.models import ModelProvider
from kagya.motivation import (
    CommitmentStore,
    GoalManager,
    MotivationDynamics,
)
from kagya.persona import (
    ConsciousAgent,
    PromptBuilder,
    ResponsePostprocessor,
)
from kagya.planning import (
    PlanStore,
)
from kagya.relationship import RelationshipStore
from kagya.runtime.session_state import SessionState
from kagya.runtime.agent_state import PersistentAgentState
from kagya.runtime.working_memory import (
    WorkingMemory,
)
from kagya.runtime.context import ContextRegistry
from kagya.runtime.coordinators import (
    ActionCoordinator,
    ChatOrchestrationCoordinator,
    ChatResult,
    ExperienceIntegrationCoordinator,
    IdentityNarrativeCoordinator,
    MotivationGoalCoordinator,
    PersistenceCoordinator,
    PlanDecisionCoordinator,
)


class OperationalObserver(Protocol):
    def counter(self, name: str, amount: float = 1.0, **labels: str) -> None: ...

    def gauge(self, name: str, value: float, **labels: str) -> None: ...

    def observe(self, name: str, value: float, **labels: str) -> None: ...


class _MainLoopImplementation(
    PersistenceCoordinator,
    ExperienceIntegrationCoordinator,
    MotivationGoalCoordinator,
    PlanDecisionCoordinator,
    IdentityNarrativeCoordinator,
    ChatOrchestrationCoordinator[ChatResult],
    ActionCoordinator,
):
    """Construct runtime dependencies and expose coordinator-owned behavior."""

    def __init__(
        self,
        settings: Settings,
        provider: ModelProvider,
        memory_system: DualMemorySystem,
        *,
        session_state: SessionState | None = None,
        emotion_engine: EmotionEngineAllostasis | None = None,
        prompt_builder: PromptBuilder | None = None,
        agent: ConsciousAgent | None = None,
        postprocessor: ResponsePostprocessor | None = None,
        adapter_id: str | None = None,
        adapter_hash: str | None = None,
        activation_sequence: int | None = None,
        persistent_state: PersistentAgentState | None = None,
        working_memory: WorkingMemory | None = None,
        context_registry: ContextRegistry | None = None,
        appraiser: CognitiveAppraiser | None = None,
        telemetry: OperationalObserver | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.memory_system = memory_system
        self.session_state = session_state or SessionState()
        self.surprisal_calculator = SurprisalCalculator(
            provider,
            initial_baseline=settings.emotion.baseline_surprisal,
            initial_scale=settings.appraisal.initial_loss_scale,
            minimum_scale=settings.appraisal.minimum_loss_scale,
        )
        self.emotion_engine = emotion_engine or EmotionEngineAllostasis(
            EmotionState(optimal_loss=settings.emotion.baseline_surprisal),
            adaptation_rate=settings.emotion.decay_rate,
            response_rate=settings.emotion.appraisal_response_rate,
            resting_valence=settings.emotion.resting_valence,
            resting_arousal=settings.emotion.resting_arousal,
            valence_recovery_rate=settings.emotion.valence_recovery_rate,
            arousal_recovery_rate=settings.emotion.arousal_recovery_rate,
        )
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.agent = agent or ConsciousAgent(provider)
        self.postprocessor = postprocessor or ResponsePostprocessor()
        self.adapter_id = adapter_id
        self.adapter_hash = adapter_hash
        self.activation_sequence = activation_sequence
        self.persistent_state = persistent_state or PersistentAgentState()
        self.outbox: Any | None = None
        self.working_memory = working_memory or WorkingMemory(
            item_capacity=settings.working_memory.item_capacity,
            token_capacity=settings.working_memory.token_capacity,
        )
        self.context_registry = context_registry or ContextRegistry()
        self.appraiser = appraiser or CognitiveAppraiser()
        self.telemetry = telemetry
        self.value_system = ValueSystem(
            seeds=[
                ValueState(
                    value_id=seed.value_id,
                    name=seed.name,
                    weight=seed.weight,
                    confidence=seed.confidence,
                    stability=seed.stability,
                    source=seed.source,
                    origin=seed.origin,
                    last_updated_at=datetime.now(UTC).isoformat(),
                    allowed_update_rate=seed.allowed_update_rate,
                    origin_provenance=new_identity_origin(
                        OriginActor.INHERITED,
                        OriginInputKind.CONFIG_SEED,
                        source_ref=f"config:{seed.value_id}",
                        confidence=seed.confidence,
                        endorsement=EndorsementStatus.UNCERTAIN,
                    ),
                )
                for seed in settings.values.seeds
            ],
            conflicts=[
                ValueConflictDefinition(
                    left_value_id=conflict.left_value_id,
                    right_value_id=conflict.right_value_id,
                    name=conflict.name,
                )
                for conflict in settings.values.conflicts
            ],
            max_update_per_event=settings.values.max_update_per_event,
            max_total_update_per_event=settings.values.max_total_update_per_event,
        )
        self.goal_manager = GoalManager()
        self.plan_store = PlanStore()
        self.commitment_store = CommitmentStore()
        motivation_clock = clock or (lambda: datetime.now(UTC))
        self._motivation_clock = motivation_clock
        self.motivation_dynamics = MotivationDynamics(clock=motivation_clock)
        self.attention_system = AttentionSystem(
            capacity=min(3, self.working_memory.item_capacity),
            high_arousal_cap=1,
        )
        self.decision_store = DecisionStore()
        self.self_model = SelfModel()
        self.experience_store = ExperienceStore()
        self.relationship_store = RelationshipStore()
        self.narrative_self = NarrativeSelf()
        self.belief_store = BeliefStore()
        self.feedback_store = FeedbackStore()
        self.metacognition = Metacognition()
        self._action_execution: Any | None = None
        self.agency_attribution_store: AgencyAttributionStore
        self.counterfactual_store: CounterfactualStore
        self.persistence_coordinator = PersistenceCoordinator()
        self.experience_coordinator = ExperienceIntegrationCoordinator(
            self.experience_store,
            self.relationship_store,
            self.narrative_self,
            self.motivation_dynamics,
            persist_experience=self._persist_experience_state,
            persist_narrative=self._persist_narrative_self_state,
            persist_motivation=self._persist_motivation_state,
        )
        self.motivation_coordinator = MotivationGoalCoordinator(
            self.goal_manager,
            self.motivation_dynamics,
            persist=self._persist_motivation_state,
            clock=motivation_clock,
        )
        self.plan_decision_coordinator = PlanDecisionCoordinator(
            self.plan_store,
            self.decision_store,
            persist=self._persist_motivation_state,
        )
        self.identity_coordinator = IdentityNarrativeCoordinator(
            self.self_model,
            self.narrative_self,
            persist_self_model=self._persist_self_model_state,
            persist_narrative=self._persist_narrative_self_state,
        )
        self.chat_coordinator: ChatOrchestrationCoordinator[ChatResult] = (
            ChatOrchestrationCoordinator(self._chat_transaction)
        )
        self.action_coordinator = ActionCoordinator(self._action_execution)
        self.default_context_id: str | None = None
        self.restore_appraisal_state()
        self.restore_value_state()
        self.restore_motivation_state()
        self.restore_decision_state()
        self.restore_agency_attribution_state()
        self.restore_counterfactual_state()
        self.restore_self_model_state()
        self.restore_experience_state()
        self.restore_narrative_self_state()
        self.restore_belief_state()
        self.restore_attention_state()
        self.restore_feedback_state()
        self.restore_metacognition_state()

    def _metric(self, method: str, name: str, value: float, **labels: str) -> None:
        if self.telemetry is None:
            return
        try:
            getattr(self.telemetry, method)(name, value, **labels)
        except Exception:
            # Operational telemetry cannot alter cognition or event outcomes.
            return
