"""Integrated runtime main loop."""

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
import time
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid4, uuid5

from kagya.body import EmotionEngineAllostasis, EmotionState, EmotionUpdate
from kagya.attention import (
    AttentionCandidate,
    AttentionFocus,
    AttentionSource,
    AttentionSystem,
)
from kagya.belief import (
    BeliefEvidence,
    BeliefRecord,
    BeliefStore,
    EpistemicStatus,
    Proposition,
)
from kagya.cognition import (
    ActionScore,
    AppraisalResult,
    AppraisalSignals,
    CognitiveAppraiser,
    LossMeasurement,
    SurprisalCalculator,
    ValueConflictDefinition,
    ValueEvidence,
    ValueState,
    ValueSystem,
    ValueUpdateKind,
    ValueUpdateRecord,
)
from kagya.config import Settings
from kagya.decision import (
    ActionCandidate,
    ActionType,
    DecisionDatasetGenerator,
    DecisionDatasetRecord,
    DecisionRecord,
    DecisionStatus,
    DecisionStore,
    PredictedOutcome,
    parse_candidate_output,
    schema_candidate_prompt,
)
from kagya.identity import (
    EndorsementStatus,
    AutobiographicalEpisode,
    ContinuityLink,
    EpistemicUncertainty,
    IdentityOrigin,
    IdentityRevisionProposal,
    KnownLimitation,
    IdentityClaim,
    IdentityClaimKind,
    IdentityClaimStatus,
    FutureSelfProjection,
    NarrativeChapter,
    NarrativeSelf,
    OriginActor,
    OriginInputKind,
    SelfModel,
    SelfModelState,
    new_identity_origin,
)
from kagya.experience import (
    ExperienceAppraisal,
    ExperienceRecord,
    ExperienceStore,
    build_chat_experience,
)
from kagya.feedback import (
    FeedbackPropagation,
    FeedbackProvenance,
    FeedbackRecord,
    FeedbackRevision,
    FeedbackSignal,
    FeedbackStatus,
    FeedbackStore,
    FeedbackTarget,
    FeedbackTargetType,
    TrainingDisposition,
    ValueEvidenceProposal,
    feedback_fingerprint,
    normalize_signals,
)
from kagya.memory import DualMemorySystem, MemoryContext, MemoryLifecycleStatus
from kagya.memory import ValidationStatus
from kagya.memory.quality import assess_generation_health
from kagya.metacognition import CognitiveQuality, Metacognition
from kagya.models import ModelProvider
from kagya.motivation import (
    ACCEPTED_COMMITMENT_STATUSES,
    Commitment,
    CommitmentFulfillability,
    CommitmentLifecycleAction,
    CommitmentStatus,
    CommitmentStore,
    Goal,
    GoalDecision,
    GoalDecisionAction,
    GoalDecisionInput,
    GoalManager,
    GoalStatus,
    GoalType,
    IntrinsicGoalAction,
    IntrinsicGoalDeliberation,
    IntrinsicGoalStatus,
    MotivationDynamics,
    MotivationEpisode,
    MotivationKind,
    MotivationRecord,
    MotivationSource,
    MotivationStatus,
)
from kagya.persona import (
    ConsciousAgent,
    PromptBuilder,
    PublicSubjectSummary,
    ResponsePostprocessor,
)
from kagya.planning import (
    PLAN_STATE_KEY,
    EvidenceReference,
    Plan,
    PlanCandidate,
    PlanCondition,
    PlanStatus,
    PlanStore,
    ExpectedObservation,
    StepDefinition,
    StepStatus,
    VerificationPolicy,
)
from kagya.relationship import RelationshipState, RelationshipStore
from kagya.runtime.session_state import SessionState
from kagya.runtime.agent_state import PersistentAgentState
from kagya.runtime.agent_runtime import current_agent_event
from kagya.runtime.working_memory import (
    RetentionReason,
    WorkingMemory,
    WorkingMemoryItem,
    WorkingMemoryKind,
    WorkingMemoryView,
    working_memory_item,
)
from kagya.runtime.context import ContextFrame, ContextRegistry, InterlocutorModel
from kagya.runtime.coordinators import (
    ActionCoordinator,
    ChatOrchestrationCoordinator,
    ExperienceIntegrationCoordinator,
    IdentityNarrativeCoordinator,
    MotivationGoalCoordinator,
    PersistenceCoordinator,
    PlanDecisionCoordinator,
)


@dataclass(frozen=True)
class ChatResult:
    episode_id: str
    experience_id: str
    response: str
    hidden_thought: str
    loss: float | None
    valence: float
    arousal: float
    optimal_loss: float
    model_id: str
    adapter_id: str | None
    adapter_hash: str | None
    activation_sequence: int | None
    fallback_used: bool
    prompt: str
    memory_context: MemoryContext
    working_memory_view: WorkingMemoryView
    context_id: str
    loss_measurement: LossMeasurement
    appraisal: AppraisalResult
    emotion_update: EmotionUpdate


class OperationalObserver(Protocol):
    def counter(self, name: str, amount: float = 1.0, **labels: str) -> None: ...

    def gauge(self, name: str, value: float, **labels: str) -> None: ...

    def observe(self, name: str, value: float, **labels: str) -> None: ...


class _MainLoopImplementation:
    """Connect prediction error, emotion, memory, generation, and storage."""

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
        self.motivation_dynamics = MotivationDynamics()
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
        self.restore_self_model_state()
        self.restore_experience_state()
        self.restore_narrative_self_state()
        self.restore_belief_state()
        self.restore_attention_state()
        self.restore_feedback_state()
        self.restore_metacognition_state()

    @property
    def action_execution(self) -> Any | None:
        return self._action_execution

    @action_execution.setter
    def action_execution(self, execution: Any | None) -> None:
        self._action_execution = execution
        if hasattr(self, "action_coordinator"):
            self.action_coordinator.bind(execution)

    def chat(
        self,
        user_input: str,
        debug: bool = False,
        *,
        attachments: list[dict[str, Any]] | None = None,
        context_id: str | None = None,
        source_channel: str = "runtime.chat",
        source_session_id: str | None = None,
        interlocutor_key: str | None = None,
        create_context: bool = False,
        origin_actor: OriginActor = OriginActor.USER,
    ) -> ChatResult:
        return self.chat_coordinator.chat(
            user_input,
            debug,
            attachments=attachments,
            context_id=context_id,
            source_channel=source_channel,
            source_session_id=source_session_id,
            interlocutor_key=interlocutor_key,
            create_context=create_context,
            origin_actor=origin_actor,
        )

    def _chat_transaction(
        self,
        user_input: str,
        debug: bool = False,
        *,
        attachments: list[dict[str, Any]] | None = None,
        context_id: str | None = None,
        source_channel: str = "runtime.chat",
        source_session_id: str | None = None,
        interlocutor_key: str | None = None,
        create_context: bool = False,
        origin_actor: OriginActor = OriginActor.USER,
    ) -> ChatResult:
        previous_context_frames = self.context_registry.frames
        previous_interlocutors = self.context_registry.interlocutors
        previous_default_context_id = self.default_context_id
        previous_calibration = self.surprisal_calculator.export_history()
        previous_experience_state = self.experience_store.to_json()
        previous_relationship_state = self.relationship_store.to_json()
        previous_motivation_state = self.motivation_dynamics.to_json()
        previous_narrative_state = self.narrative_self.to_json()
        current_context = self._resolve_context(
            context_id,
            source_channel=source_channel,
            source_session_id=source_session_id,
            interlocutor_key=interlocutor_key,
            create_context=create_context,
        )
        previous_working_memory = self.working_memory.items
        previous_emotion_state = self.emotion_engine.state
        try:
            working_memory_count = len(self.working_memory.items)
            self.working_memory.advance()
            evicted = max(0, working_memory_count - len(self.working_memory.items))
            if evicted:
                self._metric(
                    "counter",
                    "kagya_working_memory_evictions_total",
                    float(evicted),
                    reason="decay",
                )
            self._sync_belief_working_memory(current_context.context_id)

            def compatibility(source_id: str | None) -> tuple[float, str]:
                return self.context_registry.compatibility(source_id, current_context)

            context_view = self.working_memory.select(
                resolver=self._resolve_working_memory,
                context_compatibility=compatibility,
                attention_refs=self.attention_system.focused_working_memory_refs(),
            )
            context_text = context_view.context_text()
            model_key = (
                f"{self.settings.model.provider}:{self.settings.model.primary_id}:"
                f"{self.adapter_id or 'base'}"
            )
            loss_measurement = self.surprisal_calculator.measure(
                context_text, user_input, model_key=model_key
            )
        except Exception:
            self.working_memory.restore(previous_working_memory)
            self.context_registry.restore(
                previous_context_frames, previous_interlocutors
            )
            self.default_context_id = previous_default_context_id
            self.surprisal_calculator.restore_history(previous_calibration)
            raise
        try:
            relationship_influence = self.relationship_store.influence(
                current_context.participant_ids
            )
            appraisal = self.appraiser.assess(
                loss_measurement,
                AppraisalSignals(
                    controllability=self.settings.appraisal.default_controllability
                    if loss_measurement.valid
                    else 0.5,
                    certainty=min(
                        self.settings.appraisal.default_certainty
                        if loss_measurement.valid
                        else 0.2,
                        relationship_influence.certainty,
                    )
                    if relationship_influence.relationship_refs
                    else (
                        self.settings.appraisal.default_certainty
                        if loss_measurement.valid
                        else 0.2
                    ),
                    threat=relationship_influence.threat,
                    social_relevance=max(
                        0.5 if current_context.participant_ids else 0.0,
                        relationship_influence.social_relevance,
                    ),
                    effort_cost=min(
                        1.0,
                        len(user_input.encode("utf-8"))
                        / self.settings.working_memory.token_capacity,
                    ),
                ),
            )
            emotion_update = self.emotion_engine.update_from_appraisal(appraisal)
            emotion_state = emotion_update.state
            retrieval_started = time.perf_counter()
            try:
                memory_context = self.memory_system.retrieve_context(
                    user_input,
                    current_context_id=current_context.context_id,
                    context_compatibility=compatibility,
                )
            finally:
                self._metric(
                    "observe",
                    "kagya_memory_retrieval_duration_seconds",
                    time.perf_counter() - retrieval_started,
                )
            for tier, records in (
                ("episodic", memory_context.db1_results),
                ("semantic", memory_context.db2_results),
            ):
                for record in records:
                    self._metric(
                        "observe",
                        "kagya_memory_relevance",
                        record.semantic_relevance,
                        tier=tier,
                    )
            event = current_agent_event()
            self._admit_runtime_context(
                memory_context, emotion_state, event, current_context
            )
            working_memory_view = self.working_memory.select(
                resolver=self._resolve_working_memory,
                context_compatibility=compatibility,
                attention_refs=self.attention_system.focused_working_memory_refs(),
            )
            prompt = self.prompt_builder.build(
                user_input,
                emotion_state,
                working_memory_view,
                current_context=current_context,
                attachments=attachments or [],
                subject_summary=self._public_subject_summary(
                    current_context.context_id, current_context.participant_ids
                ),
            )
            generation_started = time.perf_counter()
            try:
                raw_response = self.agent.generate(
                    prompt, attachments=attachments or []
                )
            except Exception as exc:
                if _fallback_used(self.provider):
                    raise RuntimeError("Fallback model generation failed") from exc
                raw_response = _generate_fallback(self.provider, prompt)
            processed_response = self.postprocessor.process(raw_response)
            if not processed_response.visible_response.strip():
                if _fallback_used(self.provider):
                    raise RuntimeError(
                        "Fallback model produced an empty visible response"
                    )
                raw_response = _generate_fallback(self.provider, prompt)
                processed_response = self.postprocessor.process(raw_response)
                if not processed_response.visible_response.strip():
                    raise RuntimeError(
                        "Fallback model produced an empty visible response"
                    )
            model_id = str(
                getattr(self.provider, "last_model_id", self.settings.model.primary_id)
            )
            fallback_used = bool(getattr(self.provider, "last_fallback_used", False))
            generation_duration = max(time.perf_counter() - generation_started, 1e-9)
            metric_labels = {
                "provider": self.settings.model.provider.lower(),
                "fallback": str(fallback_used).lower(),
            }
            self._metric(
                "observe",
                "kagya_generation_duration_seconds",
                generation_duration,
                **metric_labels,
            )
            approximate_tokens = max(
                1.0, len(processed_response.visible_response.encode("utf-8")) / 4.0
            )
            self._metric(
                "observe",
                "kagya_generation_tokens_per_second",
                approximate_tokens / generation_duration,
                **metric_labels,
            )
            if fallback_used:
                self._metric(
                    "counter",
                    "kagya_provider_fallback_total",
                    1.0,
                    provider=self.settings.model.provider.lower(),
                )
            generation_health = assess_generation_health(
                processed_response.visible_response,
                loss=loss_measurement.raw_loss
                if loss_measurement.raw_loss is not None
                else float("nan"),
                fallback_used=fallback_used,
            )
            if not generation_health.healthy:
                self._metric(
                    "counter",
                    "kagya_memory_quarantine_total",
                    1.0,
                    reason="health_check",
                )
            episode_id = self.memory_system.save_episodic(
                user_input,
                processed_response.visible_response,
                loss=loss_measurement.raw_loss
                if loss_measurement.raw_loss is not None
                else 0.0,
                emotion_valence=emotion_state.valence,
                emotion_arousal=emotion_state.arousal,
                generation_health=generation_health,
                source_event_id=None if event is None else event.event_id,
                source="runtime.chat" if event is None else event.source,
                processing_sequence=None
                if event is None
                else event.processing_sequence,
                causation_id=None if event is None else event.causation_id,
                correlation_id=None if event is None else event.correlation_id,
                context_id=current_context.context_id,
                source_channel=current_context.source_channel,
                source_session_id=current_context.source_session_id,
                provider=self.settings.model.provider,
                model_id=model_id,
                model_revision=str(getattr(self.provider, "model_revision", "unknown")),
                adapter_id=None if fallback_used else self.adapter_id,
                validation_status=ValidationStatus.UNVERIFIED,
                stage_external=event is not None,
            )
            experience_input = build_chat_experience(
                source_event_id=None if event is None else event.event_id,
                source_event_sequence=None
                if event is None
                else event.processing_sequence,
                episode_id=episode_id,
                identity_origin=new_identity_origin(
                    origin_actor,
                    OriginInputKind.OBSERVATION,
                    source_ref=f"context:{current_context.context_id}",
                    event_id=None if event is None else event.event_id,
                    event_sequence=None if event is None else event.processing_sequence,
                ),
                context_id=current_context.context_id,
                interlocutor_ids=current_context.participant_ids,
                appraisal=appraisal,
                valence=emotion_state.valence,
                arousal=emotion_state.arousal,
                prediction_error=(
                    loss_measurement.mean_token_loss if loss_measurement.valid else None
                ),
                value_revision_refs={
                    value.value_id: value.revision
                    for value in self.value_system.list_values()
                },
                active_goal_refs=tuple(
                    goal.goal_id
                    for goal in self.goal_manager.list_goals(GoalStatus.ACTIVE)
                ),
                self_model_revision=self.self_model.state.revision,
            )
            integration = self.experience_coordinator.integrate(
                experience_input,
                active_commitment_refs=tuple(
                    f"commitment:{item.commitment_id}"
                    for item in self.commitment_store.list_commitments()
                    if item.status in ACCEPTED_COMMITMENT_STATUSES
                ),
                event_id=None if event is None else event.event_id,
                event_sequence=None if event is None else event.processing_sequence,
            )
            experience = integration.experience
            if integration.narrative_episode is not None:
                self._sync_self_references()
                self._persist_self_model_state()
            self.memory_system.link_experience(
                episode_id,
                experience_id=experience.experience_id,
                subjective_salience=experience.subjective_salience,
                autobiographical_importance=experience.autobiographical_importance,
            )
            self.context_registry.record_shared_history(
                current_context.participant_ids, f"episode:{episode_id}"
            )
            self._persist_appraisal_state()
            self.working_memory.admit(
                working_memory_item(
                    item_id=f"episode:{episode_id}",
                    kind=WorkingMemoryKind.CONVERSATION,
                    reference=f"episode:{episode_id}",
                    source_event_id=None if event is None else event.event_id,
                    source_event_sequence=None
                    if event is None
                    else event.processing_sequence,
                    context_id=current_context.context_id,
                    source="runtime.chat" if event is None else event.source,
                    source_channel=current_context.source_channel,
                    source_session_id=current_context.source_session_id,
                    activation=0.5 + 0.5 * experience.subjective_salience,
                    salience=experience.subjective_salience,
                    retention_reason=RetentionReason.RECENT_CONTEXT,
                )
            )
            self._metric(
                "gauge",
                "kagya_working_memory_items",
                float(len(self.working_memory.items)),
            )
            self._metric(
                "gauge",
                "kagya_attention_focus_items",
                float(len(working_memory_view.selected)),
            )
            self.refresh_attention(compete=True)
        except Exception:
            self.emotion_engine.state = previous_emotion_state
            self.working_memory.restore(previous_working_memory)
            self.context_registry.restore(
                previous_context_frames, previous_interlocutors
            )
            self.default_context_id = previous_default_context_id
            self.surprisal_calculator.restore_history(previous_calibration)
            self._persist_appraisal_state()
            self.experience_store.restore(previous_experience_state)
            self._persist_experience_state()
            self.relationship_store.restore(previous_relationship_state)
            self.motivation_dynamics.restore(previous_motivation_state)
            self._persist_motivation_state()
            self.narrative_self.restore(previous_narrative_state)
            self._persist_narrative_self_state()
            self._sync_self_references()
            self._persist_self_model_state()
            raise
        return ChatResult(
            episode_id=episode_id,
            experience_id=experience.experience_id,
            response=processed_response.visible_response,
            hidden_thought=processed_response.hidden_thought if debug else "",
            loss=loss_measurement.raw_loss,
            valence=emotion_state.valence,
            arousal=emotion_state.arousal,
            optimal_loss=emotion_state.optimal_loss,
            model_id=model_id,
            adapter_id=None if fallback_used else self.adapter_id,
            adapter_hash=None if fallback_used else self.adapter_hash,
            activation_sequence=(None if fallback_used else self.activation_sequence),
            fallback_used=fallback_used,
            prompt=prompt,
            memory_context=memory_context,
            working_memory_view=working_memory_view,
            context_id=current_context.context_id,
            loss_measurement=loss_measurement,
            appraisal=appraisal,
            emotion_update=emotion_update,
        )

    def _metric(self, method: str, name: str, value: float, **labels: str) -> None:
        if self.telemetry is None:
            return
        try:
            getattr(self.telemetry, method)(name, value, **labels)
        except Exception:
            # Operational telemetry cannot alter cognition or event outcomes.
            return

    def advance_time(self, elapsed_seconds: float) -> EmotionUpdate:
        update = self.emotion_engine.advance_time(elapsed_seconds)
        self.working_memory.admit(
            working_memory_item(
                item_id="emotion:current",
                kind=WorkingMemoryKind.EMOTION,
                content=(
                    f"Current valence={update.state.valence:.3f}, "
                    f"arousal={update.state.arousal:.3f}"
                ),
                activation=1.0,
                salience=max(0.5, update.state.arousal),
                retention_reason=RetentionReason.CURRENT_EMOTION,
                source="runtime.emotion_tick",
            )
        )
        return update

    def restore_appraisal_state(self) -> None:
        self.surprisal_calculator.restore_history(
            self.persistent_state.extensions.get("appraisal_calibration")
        )

    def _persist_appraisal_state(self) -> None:
        self.persistence_coordinator.persist(
            self.persistent_state.extensions,
            "appraisal_calibration",
            self.surprisal_calculator.export_history,
        )

    def restore_experience_state(self) -> None:
        self.persistence_coordinator.restore(
            self.persistent_state.extensions,
            "experiences",
            self.experience_store.restore,
            self._persist_experience_state,
        )

    def _persist_experience_state(self) -> None:
        self.persistence_coordinator.persist(
            self.persistent_state.extensions,
            "experiences",
            self.experience_store.to_json,
        )

    def restore_narrative_self_state(self) -> None:
        self.narrative_self.restore(
            self.persistent_state.identity_extensions.get("narrative_self")
        )
        self._sync_self_references()
        self._persist_narrative_self_state()

    def _persist_narrative_self_state(self) -> None:
        self.persistent_state.identity_extensions["narrative_self"] = (
            self.narrative_self.to_json()
        )

    def restore_attention_state(self) -> None:
        self.persistence_coordinator.restore(
            self.persistent_state.extensions,
            "attention",
            self.attention_system.restore,
            self._persist_attention_state,
        )

    def _persist_attention_state(self) -> None:
        self.persistence_coordinator.persist(
            self.persistent_state.extensions,
            "attention",
            self.attention_system.to_json,
        )

    def refresh_attention(self, *, compete: bool = False) -> AttentionFocus:
        """Refresh structured candidates from authoritative internal stores."""
        experience_candidate_ids: set[str] = set()
        for experience in self.experience_store.list_records():
            experience_candidate_ids.add(f"experience:{experience.experience_id}")
            memory_refs = experience.result_refs.get("memory", ())
            self.attention_system.observe(
                candidate_id=f"experience:{experience.experience_id}",
                target_ref=f"experience:{experience.experience_id}",
                source=AttentionSource.EXPERIENCE,
                source_refs=(f"experience:{experience.experience_id}",),
                working_memory_ref=memory_refs[0] if memory_refs else None,
                salience=experience.subjective_salience,
                novelty=(experience.appraisal.novelty or 0.0)
                if experience.appraisal.novelty_valid
                else 0.0,
                value_relevance=experience.self_relevance,
                arousal=experience.appraisal.arousal,
                persistence=experience.autobiographical_importance,
            )
        self.attention_system.synchronize_source(
            AttentionSource.EXPERIENCE, experience_candidate_ids
        )
        active_motivations = [
            item
            for item in self.motivation_dynamics.list_records()
            if item.status == MotivationStatus.ACTIVE
        ]
        motivation_candidate_ids = {
            f"motivation:{item.motivation_id}" for item in active_motivations
        }
        for motivation in active_motivations:
            self.attention_system.observe(
                candidate_id=f"motivation:{motivation.motivation_id}",
                target_ref=motivation.target_ref,
                source=AttentionSource.MOTIVATION,
                source_refs=tuple(
                    dict.fromkeys(
                        (
                            f"motivation:{motivation.motivation_id}",
                            *(
                                f"experience:{item}"
                                for item in motivation.related_experience_ids
                            ),
                        )
                    )
                ),
                drive=motivation.strength,
                urgency=motivation.strength * (1.0 - motivation.satiation),
                persistence=motivation.persistence,
            )
        self.attention_system.synchronize_source(
            AttentionSource.MOTIVATION, motivation_candidate_ids
        )
        active_commitments = {
            item.related_goal_id: item
            for item in self.commitment_store.list_commitments()
            if item.status in ACCEPTED_COMMITMENT_STATUSES
        }
        active_goals = [
            goal
            for goal in self.goal_manager.list_goals()
            if goal.status
            in {GoalStatus.ACTIVE, GoalStatus.CANDIDATE, GoalStatus.SUSPENDED}
        ]
        goal_candidate_ids = {f"goal:{goal.goal_id}" for goal in active_goals}
        for goal in active_goals:
            commitment = active_commitments.get(goal.goal_id)
            value_relevance = min(
                1.0,
                max((abs(item) for item in goal.value_effects.values()), default=0.0),
            )
            self.attention_system.observe(
                candidate_id=f"goal:{goal.goal_id}",
                target_ref=f"goal:{goal.goal_id}",
                source=AttentionSource.GOAL,
                source_refs=tuple(
                    ref
                    for ref in (
                        f"goal:{goal.goal_id}",
                        None
                        if goal.origin_value_id is None
                        else f"value:{goal.origin_value_id}",
                    )
                    if ref is not None
                ),
                working_memory_ref=f"goal:{goal.goal_id}",
                urgency=goal.urgency,
                value_relevance=value_relevance,
                commitment_cost=0.8 if commitment is not None else 0.0,
                persistence=max(
                    goal.priority, 0.8 if goal.status == GoalStatus.ACTIVE else 0.0
                ),
            )
        self.attention_system.synchronize_source(
            AttentionSource.GOAL, goal_candidate_ids
        )
        active_commitment_records = self.commitment_store.list_commitments()
        active_commitment_records = [
            item
            for item in active_commitment_records
            if item.status in ACCEPTED_COMMITMENT_STATUSES
        ]
        commitment_candidate_ids = {
            f"commitment:{item.commitment_id}" for item in active_commitment_records
        }
        for commitment in active_commitment_records:
            if commitment.related_goal_id is None:
                continue
            self.attention_system.observe(
                candidate_id=f"commitment:{commitment.commitment_id}",
                target_ref=f"commitment:{commitment.commitment_id}",
                source=AttentionSource.COMMITMENT,
                source_refs=(
                    f"commitment:{commitment.commitment_id}",
                    f"goal:{commitment.related_goal_id}",
                ),
                working_memory_ref=f"commitment:{commitment.commitment_id}",
                urgency=self.goal_manager.get(commitment.related_goal_id).urgency,
                commitment_cost=1.0,
                persistence=1.0,
            )
        self.attention_system.synchronize_source(
            AttentionSource.COMMITMENT, commitment_candidate_ids
        )
        event = current_agent_event()
        if compete:
            self.attention_system.compete(
                event_id=None if event is None else event.event_id,
                event_sequence=None if event is None else event.processing_sequence,
            )
        self._persist_attention_state()
        return self.attention_system.focus

    def refocus_attention(
        self,
        candidate_ids: tuple[str, ...],
        *,
        reason_code: str,
        provenance_refs: tuple[str, ...],
    ) -> AttentionFocus:
        event = current_agent_event()
        focus = self.attention_system.refocus(
            candidate_ids,
            reason_code=reason_code,
            provenance_refs=provenance_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_attention_state()
        return focus

    def defer_attention(
        self, candidate_id: str, *, reason_code: str, provenance_refs: tuple[str, ...]
    ) -> AttentionFocus:
        event = current_agent_event()
        focus = self.attention_system.defer(
            candidate_id,
            reason_code=reason_code,
            provenance_refs=provenance_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_attention_state()
        return focus

    def ignore_attention(
        self, candidate_id: str, *, reason_code: str, provenance_refs: tuple[str, ...]
    ) -> AttentionFocus:
        event = current_agent_event()
        focus = self.attention_system.ignore(
            candidate_id,
            reason_code=reason_code,
            provenance_refs=provenance_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_attention_state()
        return focus

    def resume_attention(self, candidate_id: str) -> AttentionCandidate:
        candidate = self.attention_system.resume(candidate_id)
        self._persist_attention_state()
        return candidate

    def get_experience(self, experience_id: str) -> ExperienceRecord:
        return self.experience_store.get(experience_id)

    def list_experiences(self) -> list[ExperienceRecord]:
        return self.experience_store.list_records()

    def restore_belief_state(self) -> None:
        self.persistence_coordinator.restore(
            self.persistent_state.extensions,
            "beliefs",
            self.belief_store.restore,
            self._persist_belief_state,
        )

    def _persist_belief_state(self) -> None:
        self.persistence_coordinator.persist(
            self.persistent_state.extensions,
            "beliefs",
            self.belief_store.to_json,
        )

    def propose_belief_from_experience(
        self,
        experience_id: str,
        *,
        proposition: Proposition,
        source_trust: float,
        confidence: float,
        context_scope: tuple[str, ...] = (),
        valid_from: str | None = None,
        valid_until: str | None = None,
        belief_id: str | None = None,
    ) -> BeliefRecord:
        experience = self.experience_store.get(experience_id)
        record = self.belief_store.propose(
            proposition,
            identity_origin=experience.identity_origin,
            evidence=(
                BeliefEvidence(
                    reference=f"experience:{experience_id}",
                    evidence_type="external_claim",
                    source_trust=source_trust,
                    observed_at=experience.created_at,
                ),
            ),
            confidence=confidence,
            context_scope=context_scope,
            valid_from=valid_from,
            valid_until=valid_until,
            belief_id=belief_id,
        )
        self.link_experience_result(
            experience_id,
            kind="belief",
            reference=f"belief:{record.belief_id}",
            evidence_refs=(f"experience:{experience_id}",),
        )
        self._persist_belief_state()
        self._sync_belief_working_memory(None)
        return record

    def resolve_belief(
        self,
        belief_id: str,
        *,
        accept: bool,
        confidence: float,
        epistemic_status: EpistemicStatus,
        reason_code: str,
        evidence_refs: tuple[str, ...],
    ) -> BeliefRecord:
        event = current_agent_event()
        record = self.belief_store.resolve(
            belief_id,
            accept=accept,
            confidence=confidence,
            epistemic_status=epistemic_status,
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_belief_state()
        self._sync_belief_working_memory(None)
        return record

    def retract_belief(
        self, belief_id: str, *, reason_code: str, evidence_refs: tuple[str, ...]
    ) -> BeliefRecord:
        event = current_agent_event()
        record = self.belief_store.retract(
            belief_id,
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_belief_state()
        self._sync_belief_working_memory(None)
        return record

    def supersede_belief(
        self,
        old_belief_id: str,
        new_belief_id: str,
        *,
        reason_code: str,
        evidence_refs: tuple[str, ...],
    ) -> tuple[BeliefRecord, BeliefRecord]:
        event = current_agent_event()
        records = self.belief_store.supersede(
            old_belief_id,
            new_belief_id,
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_belief_state()
        self._sync_belief_working_memory(None)
        return records

    def expire_beliefs(self) -> list[BeliefRecord]:
        records = self.belief_store.expire()
        self._persist_belief_state()
        self._sync_belief_working_memory(None)
        return records

    def list_beliefs(self) -> list[BeliefRecord]:
        return self.belief_store.list_records()

    def _sync_belief_working_memory(self, context_id: str | None) -> None:
        for item in tuple(self.working_memory.items):
            if item.kind == WorkingMemoryKind.BELIEF:
                self.working_memory.forget(item.item_id)
        for belief in self.belief_store.active(context_id=context_id):
            self.working_memory.admit(
                working_memory_item(
                    item_id=f"belief:{belief.belief_id}",
                    kind=WorkingMemoryKind.BELIEF,
                    content=(
                        f"Adopted belief ({belief.epistemic_status.value}, "
                        f"confidence={belief.confidence:.3f}): "
                        f"{belief.proposition.normalized}"
                    ),
                    activation=0.8,
                    salience=belief.confidence,
                    retention_reason=RetentionReason.ESTABLISHED_BELIEF,
                    source="runtime.belief_store",
                )
            )

    def reassess_experience(
        self,
        experience_id: str,
        *,
        appraisal: ExperienceAppraisal,
        reason_code: str,
        evidence_refs: tuple[str, ...],
    ) -> ExperienceRecord:
        event = current_agent_event()
        record = self.experience_store.revise_appraisal(
            experience_id,
            appraisal=appraisal,
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        for reference in record.result_refs.get("memory", ()):
            if reference.startswith("episode:"):
                self.memory_system.link_experience(
                    reference.removeprefix("episode:"),
                    experience_id=record.experience_id,
                    subjective_salience=record.subjective_salience,
                    autobiographical_importance=record.autobiographical_importance,
                )
        self._persist_experience_state()
        return record

    def link_experience_result(
        self,
        experience_id: str,
        *,
        kind: str,
        reference: str,
        evidence_refs: tuple[str, ...],
    ) -> ExperienceRecord:
        event = current_agent_event()
        record = self.experience_store.link_result(
            experience_id,
            kind=kind,
            reference=reference,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self.narrative_self.observe_experience(record)
        self._persist_experience_state()
        self._persist_narrative_self_state()
        return record

    def restore_value_state(self) -> None:
        self.value_system.restore(self.persistent_state.values)
        self._persist_value_state()

    def _persist_value_state(self) -> None:
        self.persistent_state.values = self.value_system.to_json()

    def apply_value_impacts(
        self,
        appraisal: AppraisalResult,
        impacts: dict[str, float],
        *,
        kind: ValueUpdateKind,
        memory_ids: tuple[str, ...] = (),
        source: str = "runtime",
        proposal_id: str | None = None,
        origin_actor: OriginActor = OriginActor.SELF,
        origin_input_kind: OriginInputKind = OriginInputKind.INTERNAL_STATE,
        experience_ids: tuple[str, ...] = (),
        decision_id: str | None = None,
        context_id: str | None = None,
    ) -> list[ValueUpdateRecord]:
        event = current_agent_event()
        evidence = ValueEvidence(
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
            memory_ids=memory_ids,
            source=source,
            experience_ids=experience_ids,
            decision_id=decision_id,
            context_id=context_id,
            identity_origin=new_identity_origin(
                origin_actor,
                origin_input_kind,
                source_ref=source,
                event_id=None if event is None else event.event_id,
                event_sequence=None if event is None else event.processing_sequence,
            ),
        )
        proposals = self.value_system.proposals_from_appraisal(
            appraisal,
            impacts,
            kind=kind,
            evidence=evidence,
            proposal_id=proposal_id,
        )
        records = self.value_system.apply(proposals)
        self._persist_value_state()
        return records

    def apply_value_evidence_from_experience(
        self,
        experience_id: str,
        impacts: dict[str, float],
        *,
        proposal_id: str | None = None,
    ) -> list[ValueUpdateRecord]:
        experience = self.experience_store.get(experience_id)
        proposals = self.value_system.proposals_from_experience(
            experience, impacts, proposal_id=proposal_id
        )
        records = self.value_system.apply(proposals)
        for record in records:
            self.link_experience_result(
                experience_id,
                kind="value",
                reference=f"value:{record.value_id}@{record.after['revision']}",
                evidence_refs=record.evidence_ids,
            )
        self._sync_self_references()
        self._persist_self_model_state()
        self._persist_value_state()
        return records

    def evaluate_value_options(
        self, options: dict[str, dict[str, float]], *, context_id: str | None = None
    ) -> list[ActionScore]:
        return self.value_system.evaluate(options, context_id=context_id)

    def freeze_value(self, value_id: str, *, frozen: bool) -> ValueState:
        state = self.value_system.freeze(value_id, frozen=frozen)
        self._persist_value_state()
        return state

    def rollback_value(self, value_id: str, *, target_revision: int) -> ValueState:
        state = self.value_system.rollback(value_id, target_revision=target_revision)
        self._persist_value_state()
        return state

    def reset_values(
        self, value_ids: tuple[str, ...] | None = None
    ) -> list[ValueState]:
        states = self.value_system.reset(value_ids)
        self._persist_value_state()
        return states

    def restore_motivation_state(self) -> None:
        decisions = self.persistent_state.motivation_extensions.get(
            "goal_decisions", []
        )
        intrinsic_deliberations = self.persistent_state.motivation_extensions.get(
            "intrinsic_goal_deliberations", []
        )
        self.goal_manager.restore(
            self.persistent_state.active_goals,
            decisions if isinstance(decisions, list) else [],
            intrinsic_deliberations
            if isinstance(intrinsic_deliberations, list)
            else [],
        )
        self.commitment_store.restore(self.persistent_state.commitments)
        self.plan_store.restore(
            self.persistent_state.motivation_extensions.get(PLAN_STATE_KEY)
        )
        self.motivation_dynamics.restore(
            self.persistent_state.motivation_extensions.get("dynamics")
        )
        self._persist_motivation_state()

    def _persist_motivation_state(self) -> None:
        self.persistent_state.active_goals = self.goal_manager.goals_json()
        self.persistent_state.commitments = self.commitment_store.to_json()
        self.persistent_state.motivation_extensions["goal_decisions"] = (
            self.goal_manager.decisions_json()
        )
        self.persistent_state.motivation_extensions["intrinsic_goal_deliberations"] = (
            self.goal_manager.intrinsic_deliberations_json()
        )
        self.persistent_state.motivation_extensions["dynamics"] = (
            self.motivation_dynamics.to_json()
        )
        self.persistent_state.motivation_extensions[PLAN_STATE_KEY] = (
            self.plan_store.to_json()
        )

    def derive_structured_motivations(self) -> list[MotivationRecord]:
        """Translate existing structured state into evidence-bound motives."""
        observed: list[MotivationRecord] = []
        for episode in self.narrative_self.episodes.values():
            if episode.unresolved_tension < 0.4:
                continue
            observed.append(
                self.motivation_dynamics.observe_structured_signal(
                    MotivationKind.DRIVE,
                    MotivationSource.CLOSURE,
                    f"narrative:{episode.episode_id}",
                    signal=episode.unresolved_tension,
                    uncertainty=0.3,
                    source_refs=(
                        f"narrative:{episode.episode_id}",
                        *(f"experience:{item}" for item in episode.experience_ids),
                    ),
                )
            )
        for self_conflict in self.narrative_self.conflicts.values():
            if self_conflict.resolved_at is not None:
                continue
            observed.append(
                self.motivation_dynamics.observe_structured_signal(
                    MotivationKind.DRIVE,
                    MotivationSource.DELIBERATION,
                    f"self-conflict:{self_conflict.conflict_id}",
                    signal=0.7,
                    uncertainty=0.6,
                    source_refs=(
                        f"self-conflict:{self_conflict.conflict_id}",
                        *self_conflict.evidence_refs,
                    ),
                )
            )
        for projection in self.narrative_self.future_self.values():
            if projection.gap <= 0.0:
                continue
            motivation = self.motivation_dynamics.observe_structured_signal(
                MotivationKind.DESIRE,
                MotivationSource.LEARNING,
                f"future-self:{projection.projection_id}",
                signal=projection.gap,
                uncertainty=max(0.0, 1.0 - projection.current_level),
                source_refs=(
                    f"future-self:{projection.projection_id}",
                    *projection.evidence_refs,
                ),
            )
            self.narrative_self.link_future_motivation(
                projection.projection_id, motivation.motivation_id
            )
            observed.append(motivation)
        for relationship in self.relationship_store.list_relationships():
            evidence_refs = tuple(
                dict.fromkeys(
                    (
                        f"relationship:{relationship.relationship_id}@{relationship.revision}",
                        *relationship.unresolved_matter_refs,
                        *relationship.conflict_refs,
                        *relationship.commitment_refs,
                    )
                )
            )
            if len(evidence_refs) == 1:
                continue
            observed.append(
                self.motivation_dynamics.observe_structured_signal(
                    MotivationKind.DRIVE,
                    MotivationSource.SOCIAL,
                    f"relationship:{relationship.relationship_id}",
                    signal=max(0.5, relationship.axes.closeness),
                    uncertainty=relationship.uncertainty,
                    source_refs=evidence_refs,
                )
            )
        for tradeoff in self.value_system.tradeoffs:
            if not tradeoff.conflict_names:
                continue
            values = [self.value_system.get(item) for item in tradeoff.value_ids]
            evidence_refs = tuple(
                dict.fromkeys(
                    (
                        f"value-tradeoff:{tradeoff.tradeoff_id}",
                        *(f"value-conflict:{item}" for item in tradeoff.conflict_names),
                        *(
                            f"value:{value.value_id}@{value.revision}"
                            for value in values
                        ),
                    )
                )
            )
            motives = [
                self.motivation_dynamics.observe_structured_signal(
                    MotivationKind.DESIRE,
                    MotivationSource.DELIBERATION,
                    f"value:{value.value_id}",
                    signal=min(
                        1.0,
                        max(
                            value.weight,
                            abs(
                                tradeoff.contribution_by_value.get(value.value_id, 0.0)
                            ),
                        ),
                    ),
                    uncertainty=1.0 - value.confidence,
                    source_refs=evidence_refs,
                    value_ids=(value.value_id,),
                )
                for value in values
            ]
            for index, left_motive in enumerate(motives):
                for right_motive in motives[index + 1 :]:
                    self.motivation_dynamics.register_conflict(
                        left_motive.motivation_id, right_motive.motivation_id
                    )
            observed.extend(motives)
        for limitation in self.self_model.state.known_limitations.values():
            observed.append(
                self.motivation_dynamics.observe_structured_signal(
                    MotivationKind.DRIVE,
                    MotivationSource.LEARNING,
                    f"limitation:{limitation.limitation_id}",
                    signal=limitation.confidence,
                    uncertainty=1.0 - limitation.confidence,
                    source_refs=(
                        f"limitation:{limitation.limitation_id}",
                        *limitation.evidence_refs,
                    ),
                )
            )
        for uncertainty in self.self_model.state.epistemic_uncertainties.values():
            observed.append(
                self.motivation_dynamics.observe_structured_signal(
                    MotivationKind.INTEREST,
                    MotivationSource.LEARNING,
                    f"uncertainty:{uncertainty.uncertainty_id}",
                    signal=max(0.4, 1.0 - uncertainty.confidence),
                    uncertainty=uncertainty.confidence,
                    source_refs=(
                        f"uncertainty:{uncertainty.uncertainty_id}",
                        *uncertainty.evidence_refs,
                    ),
                )
            )
        self._persist_narrative_self_state()
        self._persist_motivation_state()
        return observed

    def reevaluate_motivation(
        self, *, max_goal_proposals: int | None = None
    ) -> tuple[MotivationEpisode, list[Goal]]:
        event = current_agent_event()
        self.derive_structured_motivations()
        candidates, held_ids = self.motivation_dynamics.goal_candidates(
            max_goal_proposals
        )
        goals: list[Goal] = []
        selected_ids: list[str] = []
        for candidate in candidates:
            goal = self.propose_goal(
                goal_type=GoalType.INTRINSIC,
                description=candidate.description,
                structured_target={
                    "motivation_id": candidate.motivation_id,
                    "target_ref": candidate.target_ref,
                    "source_refs": list(candidate.source_refs),
                    "motivation_revision": self.motivation_dynamics.get(
                        candidate.motivation_id
                    ).revision,
                    "motivation_strength": self.motivation_dynamics.get(
                        candidate.motivation_id
                    ).strength,
                    "motivation_persistence": self.motivation_dynamics.get(
                        candidate.motivation_id
                    ).persistence,
                    "motivation_uncertainty": self.motivation_dynamics.get(
                        candidate.motivation_id
                    ).uncertainty,
                },
                origin_actor=OriginActor.SELF,
                origin_input_kind=OriginInputKind.INTERNAL_STATE,
                origin_source_ref=f"motivation:{candidate.motivation_id}",
                priority=candidate.priority,
                urgency=candidate.urgency,
                expected_utility=candidate.priority,
                confidence=candidate.confidence,
                motivation_revision_ref=(
                    f"motivation:{candidate.motivation_id}@"
                    f"{self.motivation_dynamics.get(candidate.motivation_id).revision}"
                ),
                goal_id=f"intrinsic:{candidate.motivation_id}",
            )
            motivation = self.motivation_dynamics.link_goal(
                candidate.motivation_id, goal.goal_id
            )
            for experience_id in motivation.related_experience_ids:
                self.link_experience_result(
                    experience_id,
                    kind="goal",
                    reference=f"goal:{goal.goal_id}",
                    evidence_refs=(f"motivation:{motivation.motivation_id}",),
                )
            goals.append(goal)
            selected_ids.append(candidate.motivation_id)
        episode = self.motivation_dynamics.record_episode(
            selected_ids=tuple(selected_ids),
            held_ids=held_ids,
            generated_goal_ids=tuple(goal.goal_id for goal in goals),
            budget=max_goal_proposals,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        unresolved_intrinsic = any(
            goal.intrinsic_status
            in {
                IntrinsicGoalStatus.PROPOSAL,
                IntrinsicGoalStatus.DELIBERATING,
                IntrinsicGoalStatus.DEFERRED,
                IntrinsicGoalStatus.ENDORSED,
            }
            for goal in self.goal_manager.goals.values()
            if goal.goal_type == GoalType.INTRINSIC
        )
        if not candidates and not unresolved_intrinsic:
            self.goal_manager.record_no_intrinsic_goal(
                provenance_refs=("motivation-dynamics:no-eligible-proposal",),
                event_id=None if event is None else event.event_id,
                event_sequence=None if event is None else event.processing_sequence,
            )
        self._persist_motivation_state()
        return episode, goals

    def decay_motivation(self, elapsed_hours: float) -> list[MotivationRecord]:
        records = self.motivation_dynamics.decay(elapsed_hours)
        self._persist_motivation_state()
        return records

    def decay_motivation_record(
        self, motivation_id: str, elapsed_hours: float
    ) -> MotivationRecord | None:
        record = self.motivation_dynamics.decay_record(motivation_id, elapsed_hours)
        self._persist_motivation_state()
        return record

    def deliberate_intrinsic_goal(self, goal_id: str) -> IntrinsicGoalDeliberation:
        """Resolve an intrinsic proposal from structured authoritative state only."""
        event = current_agent_event()
        goal = self.goal_manager.get(goal_id)
        if goal.intrinsic_status not in {
            IntrinsicGoalStatus.PROPOSAL,
            IntrinsicGoalStatus.DEFERRED,
        }:
            raise ValueError("Intrinsic proposal is not available for deliberation")
        target = goal.structured_target or {}
        motivation_id = target.get("motivation_id")
        if not isinstance(motivation_id, str):
            raise ValueError("Intrinsic proposal requires an originating Motivation")
        motivation = self.motivation_dynamics.get(motivation_id)
        motivation_score = (
            0.5 * float(target.get("motivation_strength", motivation.strength))
            + 0.3 * float(target.get("motivation_persistence", motivation.persistence))
            + 0.2
            * (
                1.0
                - float(target.get("motivation_uncertainty", motivation.uncertainty))
            )
        )

        deliberation_value_effects = dict(goal.value_effects)
        for value_id in motivation.related_value_ids:
            deliberation_value_effects.setdefault(value_id, 1.0)
        value_scores = self.value_system.evaluate(
            {goal.goal_id: deliberation_value_effects}
        )
        value_score = max(-1.0, min(1.0, value_scores[0].total_score))
        value_conflicts = value_scores[0].conflicts
        value_ids = set(deliberation_value_effects)
        if goal.origin_value_id is not None:
            value_ids.add(goal.origin_value_id)
        values = [self.value_system.get(value_id) for value_id in sorted(value_ids)]
        value_revisions = {value.value_id: value.revision for value in values}

        active_goal_conflicts = tuple(
            conflict_id
            for conflict_id in goal.conflict_ids
            if self.goal_manager.get(conflict_id).status == GoalStatus.ACTIVE
        )
        active_commitments = [
            item
            for item in self.commitment_store.list_commitments()
            if item.status in ACCEPTED_COMMITMENT_STATUSES
        ]
        proposal_conflict_refs = {
            motivation_id,
            f"motivation:{motivation_id}",
            *goal.related_desire_ids,
            *value_ids,
            *(f"value:{item}" for item in value_ids),
            *goal.conflict_ids,
            *(f"goal:{item}" for item in goal.conflict_ids),
        }
        commitment_conflicts = tuple(
            item.commitment_id
            for item in active_commitments
            if any(
                conflict.subject_ref in proposal_conflict_refs
                for conflict in item.unresolved_conflicts
            )
            or (
                item.related_goal_id is not None
                and item.related_goal_id in goal.conflict_ids
            )
        )
        conflict_score = -1.0 if active_goal_conflicts or commitment_conflicts else 1.0

        attention_capacity = self.attention_system.capacity
        proposal_has_attention = any(
            candidate_id
            in {
                f"goal:{goal.goal_id}",
                f"motivation:{motivation_id}",
            }
            for candidate_id in self.attention_system.focus.candidate_ids
        )
        available_attention = max(
            0, attention_capacity - len(self.attention_system.focus.candidate_ids)
        )
        if proposal_has_attention:
            available_attention = max(1, available_attention)
        attention_score = available_attention / attention_capacity
        cost = _unit_target(target.get("estimated_cost", 0.0), "estimated_cost")
        risk = _unit_target(target.get("estimated_risk", 0.0), "estimated_risk")
        cost_risk_score = 1.0 - 0.5 * (cost + risk)

        cognitive_load = min(
            1.0, len(self.working_memory.items) / self.working_memory.item_capacity
        )
        attention_saturation = len(self.attention_system.focus.candidate_ids) / max(
            1, attention_capacity
        )
        quality = self.metacognition.current_quality(
            cognitive_load=cognitive_load,
            attention_saturation=attention_saturation,
            emotion_valence=self.emotion_engine.state.valence,
            emotion_arousal=self.emotion_engine.state.arousal,
            provenance_refs=(
                f"working-memory:{len(self.working_memory.items)}",
                f"attention-focus:{self.attention_system.focus.revision}",
                "emotion:current",
            ),
        )

        relationship = self._relationship_for_target(target)
        relationship_score = (
            0.5
            if relationship is None
            else max(
                -1.0,
                min(
                    1.0,
                    relationship.axes.trust
                    + relationship.reciprocity
                    - relationship.axes.caution
                    - relationship.uncertainty,
                ),
            )
        )
        relationship_revisions = (
            {}
            if relationship is None
            else {relationship.relationship_id: relationship.revision}
        )

        themes = set(_string_values(target.get("theme_codes"))) | set(
            _string_values(target.get("topic_tags"))
        )
        narrative_conflicts = tuple(
            item.conflict_id
            for item in self.narrative_self.conflicts.values()
            if item.resolved_at is None
            and any(
                themes.intersection(self.narrative_self.get_claim(claim_id).theme_codes)
                for claim_id in item.claim_ids
            )
        )
        narrative_score = (
            -1.0 if narrative_conflicts else (0.7 if goal.narrative_self_refs else 0.5)
        )

        capability_ids = _string_values(target.get("capability_ids"))
        capabilities = self.self_model.state.capabilities
        missing_capabilities = tuple(
            identifier
            for identifier in capability_ids
            if identifier not in capabilities
        )
        matching_limitations = tuple(
            item
            for item in self.self_model.state.known_limitations.values()
            if set(item.capability_ids).intersection(capability_ids)
            or themes.intersection(item.tags)
        )
        matching_uncertainties = tuple(
            item
            for item in self.self_model.state.epistemic_uncertainties.values()
            if themes.intersection(item.tags)
        )
        capability_score = (
            0.5
            if not capability_ids
            else sum(
                capabilities[item].confidence
                for item in capability_ids
                if item in capabilities
            )
            / max(1, len(capability_ids))
        )
        capability_score -= max(
            (item.confidence for item in matching_limitations), default=0.0
        )
        missing_information = tuple(
            dict.fromkeys(
                (
                    *_string_values(target.get("missing_information")),
                    *(f"capability:{item}" for item in missing_capabilities),
                    *(
                        f"uncertainty:{item.uncertainty_id}"
                        for item in matching_uncertainties
                    ),
                    *(("goal:needs_information",) if goal.needs_information else ()),
                )
            )
        )
        information_score = -1.0 if missing_information else 1.0

        factors = {
            "motivation": motivation_score,
            "values": value_score,
            "goal_commitment_conflicts": conflict_score,
            "attention_capacity": attention_score,
            "cost_risk": cost_risk_score,
            "metacognitive_quality": quality.estimated_quality,
            "relationship": relationship_score,
            "narrative_continuity": narrative_score,
            "capability": max(-1.0, min(1.0, capability_score)),
            "information": information_score,
        }
        score = max(
            -1.0,
            min(
                1.0,
                0.2 * motivation_score
                + 0.15 * value_score
                + 0.12 * conflict_score
                + 0.08 * attention_score
                + 0.1 * cost_risk_score
                + 0.1 * quality.estimated_quality
                + 0.06 * relationship_score
                + 0.07 * narrative_score
                + 0.07 * capability_score
                + 0.05 * information_score,
            ),
        )
        reasons: list[str] = ["structured_multi_factor_deliberation"]
        if motivation.status != MotivationStatus.ACTIVE:
            action = IntrinsicGoalAction.REJECT
            reasons.append("originating_motivation_inactive")
        elif (
            active_goal_conflicts
            or commitment_conflicts
            or value_conflicts
            or narrative_conflicts
        ):
            action = IntrinsicGoalAction.DEFER
            reasons.append("major_unresolved_conflict")
        elif missing_information:
            action = IntrinsicGoalAction.DEFER
            reasons.append("missing_information_or_capability")
        elif available_attention == 0 or quality.estimated_quality < 0.4:
            action = IntrinsicGoalAction.DEFER
            reasons.append("insufficient_deliberative_capacity")
        elif risk >= 0.85 or cost >= 0.9 or score < 0.2:
            action = IntrinsicGoalAction.REJECT
            reasons.append("cost_risk_or_utility_unacceptable")
        elif score >= 0.45:
            action = IntrinsicGoalAction.ENDORSE
            reasons.append("multi_factor_threshold_satisfied")
        else:
            action = IntrinsicGoalAction.DEFER
            reasons.append("endorsement_threshold_not_met")

        provenance = tuple(
            dict.fromkeys(
                (
                    goal.motivation_revision_ref
                    or f"motivation:{motivation_id}@{motivation.revision}",
                    *(f"value:{item.value_id}@{item.revision}" for item in values),
                    *(
                        f"value-evidence:{item}"
                        for value in values
                        for item in (
                            *value.supporting_evidence_ids,
                            *value.opposing_evidence_ids,
                        )
                    ),
                    *(f"goal:{item}" for item in active_goal_conflicts),
                    *(f"commitment:{item}" for item in commitment_conflicts),
                    f"attention-focus:{self.attention_system.focus.revision}",
                    *quality.provenance_refs,
                    *(
                        f"relationship:{key}@{revision}"
                        for key, revision in relationship_revisions.items()
                    ),
                    *goal.narrative_self_refs,
                    *(f"self-conflict:{item}" for item in narrative_conflicts),
                    f"self-model:{self.self_model.state.revision}",
                    *(
                        f"limitation:{item.limitation_id}"
                        for item in matching_limitations
                    ),
                    *missing_information,
                )
            )
        )
        self.goal_manager.begin_intrinsic_deliberation(goal_id)
        record = self.goal_manager.resolve_intrinsic_deliberation(
            goal_id,
            action=action,
            score=score,
            factor_scores=factors,
            reason_codes=tuple(reasons),
            provenance_refs=provenance,
            value_revision_refs=value_revisions,
            self_model_revision=self.self_model.state.revision,
            attention_revision=self.attention_system.focus.revision,
            relationship_revision_refs=relationship_revisions,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        return record

    def schedule_motivation_reviews(
        self, review_at: datetime
    ) -> list[MotivationRecord]:
        records = self.motivation_dynamics.schedule_next_reviews(review_at)
        self._persist_motivation_state()
        return records

    def record_no_intrinsic_goal(self) -> IntrinsicGoalDeliberation:
        event = current_agent_event()
        record = self.goal_manager.record_no_intrinsic_goal(
            provenance_refs=("motivation-dynamics:no-eligible-proposal",),
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        return record

    def generate_intrinsic_plan(self, goal_id: str) -> Plan:
        goal = self.goal_manager.get(goal_id)
        if goal.intrinsic_status != IntrinsicGoalStatus.ENDORSED:
            raise ValueError("Plan generation requires an endorsed intrinsic Goal")
        existing = self.plan_store.list_plans(goal_id=goal_id)
        if existing:
            return existing[0]
        target = goal.structured_target or {}
        motivation_id = str(target.get("motivation_id", goal_id))
        candidate = PlanCandidate(
            plan_id=f"intrinsic-plan:{motivation_id}",
            goal_id=goal_id,
            success_condition=PlanCondition(
                condition_code="intrinsic_goal_context_observed",
                required_evidence_types=("observation",),
            ),
            failure_condition=PlanCondition(
                condition_code="intrinsic_goal_blocked",
                required_evidence_types=("failure",),
            ),
            abandonment_condition=PlanCondition(
                condition_code="intrinsic_goal_abandoned",
                required_evidence_types=("abandonment",),
            ),
            steps=(
                StepDefinition(
                    step_id="observe_progress",
                    action_type=ActionType.INTERNAL,
                    action_code="observe_intrinsic_goal_progress",
                    parameters={
                        "action": {
                            "tool_name": "restricted_metadata_read",
                            "arguments": {"namespace": "project", "key": "name"},
                        }
                    },
                    expected_observation=ExpectedObservation(
                        observation_code="restricted_metadata_read",
                        evidence_types=("observation",),
                    ),
                    verification=VerificationPolicy(
                        verification_code="structured_observation",
                        required_evidence_types=("observation",),
                    ),
                    timeout_seconds=3600.0,
                ),
            ),
        )
        return self.create_plan(candidate, actor_id="subject_scheduler")

    def activate_endorsed_intrinsic_goal(self, goal_id: str) -> GoalDecision:
        goal = self.goal_manager.get(goal_id)
        plans = self.plan_store.list_plans(goal_id=goal_id)
        if goal.intrinsic_status != IntrinsicGoalStatus.ENDORSED or not plans:
            raise ValueError(
                "Intrinsic adoption requires endorsement and a generated Plan"
            )
        decision = self.adopt_goal(goal_id)
        if decision.action in {GoalDecisionAction.ACTIVATE, GoalDecisionAction.RESUME}:
            self.activate_plan(plans[0].plan_id)
        return decision

    def propose_goal(
        self,
        *,
        goal_type: GoalType,
        description: str,
        structured_target: dict[str, Any] | None = None,
        origin_value_id: str | None = None,
        origin_actor: OriginActor | None = None,
        origin_input_kind: OriginInputKind | None = None,
        origin_source_ref: str | None = None,
        priority: float = 0.5,
        urgency: float = 0.5,
        expected_utility: float = 0.5,
        confidence: float = 0.5,
        dependency_ids: tuple[str, ...] = (),
        conflict_ids: tuple[str, ...] = (),
        deadline: str | None = None,
        value_effects: dict[str, float] | None = None,
        related_desire_ids: tuple[str, ...] = (),
        motivation_revision_ref: str | None = None,
        needs_information: bool = False,
        goal_id: str | None = None,
    ) -> Goal:
        event = current_agent_event()
        if origin_value_id is not None:
            self.value_system.get(origin_value_id)
        if value_effects:
            self.value_system.evaluate({"goal_proposal": value_effects})
        target = structured_target or {}
        relationship = self._relationship_for_target(target)
        if relationship is not None:
            expected_utility = max(
                0.0,
                min(
                    1.0,
                    expected_utility
                    + 0.1
                    * (
                        relationship.reciprocity
                        + relationship.axes.trust
                        - relationship.axes.caution
                        - 0.5
                    ),
                ),
            )
        narrative_selection = self.narrative_self.select_relevant(
            theme_codes=_string_values(target.get("theme_codes"))
            + _string_values(target.get("topic_tags")),
            capability_ids=_string_values(target.get("capability_ids")),
        )
        narrative_refs = tuple(
            f"identity-claim:{claim_id}@{self.narrative_self.get_claim(claim_id).revision}"
            for claim_id in narrative_selection.claim_ids
        )
        projection_id = target.get("future_self_projection_id")
        if (
            isinstance(projection_id, str)
            and projection_id in self.narrative_self.future_self
        ):
            narrative_refs = (*narrative_refs, f"future-self:{projection_id}")
        identity_origin: IdentityOrigin | None = None
        if origin_actor is not None:
            identity_origin = new_identity_origin(
                origin_actor,
                origin_input_kind or OriginInputKind.SUGGESTION,
                source_ref=origin_source_ref,
                event_id=None if event is None else event.event_id,
                event_sequence=None if event is None else event.processing_sequence,
                endorsement=(
                    EndorsementStatus.PENDING
                    if goal_type == GoalType.INTRINSIC
                    and motivation_revision_ref is not None
                    else None
                ),
            )
        goal = self.goal_manager.propose(
            goal_type=goal_type,
            description=description,
            structured_target=structured_target,
            origin_event_id=None if event is None else event.event_id,
            origin_value_id=origin_value_id,
            identity_origin=identity_origin,
            priority=priority,
            urgency=urgency,
            expected_utility=expected_utility,
            confidence=confidence,
            dependency_ids=dependency_ids,
            conflict_ids=conflict_ids,
            deadline=deadline,
            value_effects=value_effects,
            value_revision_refs={
                value_id: self.value_system.get(value_id).revision
                for value_id in set(value_effects or {})
                | ({origin_value_id} if origin_value_id is not None else set())
            },
            narrative_self_refs=narrative_refs,
            related_desire_ids=related_desire_ids,
            motivation_revision_ref=motivation_revision_ref,
            needs_information=needs_information,
            goal_id=goal_id,
        )
        self._persist_motivation_state()
        return goal

    def adopt_goal(self, goal_id: str) -> GoalDecision:
        event = current_agent_event()
        value_scores = self._goal_value_scores()
        previous_status = self.goal_manager.get(goal_id).status
        decision = self.goal_manager.adopt(
            goal_id,
            value_score=value_scores.get(goal_id, 0.0),
            value_scores=value_scores,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        goal = self.goal_manager.get(goal_id)
        if previous_status != GoalStatus.FAILED and goal.status == GoalStatus.FAILED:
            self._apply_goal_outcome_appraisal(goal, GoalStatus.FAILED)
            self._sync_commitment_from_goal(
                goal, GoalStatus.FAILED, "deadline_expired", None
            )
        self._sync_motivation_working_memory()
        self._persist_motivation_state()
        return decision

    def transition_goal(
        self,
        goal_id: str,
        status: GoalStatus,
        *,
        reason: str,
        outcome: str | None = None,
    ) -> Goal:
        if status == GoalStatus.COMPLETED:
            plans = self.plan_store.list_plans(goal_id=goal_id)
            if plans and not any(plan.status == PlanStatus.COMPLETED for plan in plans):
                raise ValueError(
                    "Goal cannot complete before its Plan success condition"
                )
        event = current_agent_event()
        goal = self.goal_manager.transition(
            goal_id,
            status,
            reason=reason,
            outcome=outcome,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._apply_goal_outcome_appraisal(goal, status)
        self._sync_commitment_from_goal(goal, status, reason, outcome)
        if status in {
            GoalStatus.COMPLETED,
            GoalStatus.FAILED,
            GoalStatus.ABANDONED,
        }:
            self.motivation_dynamics.resolve_goal(
                goal.goal_id, success=status == GoalStatus.COMPLETED
            )
        self._sync_motivation_working_memory()
        self._persist_motivation_state()
        return goal

    def create_plan(self, candidate: PlanCandidate, *, actor_id: str) -> Plan:
        goal = self.goal_manager.get(candidate.goal_id)
        if goal.status in {
            GoalStatus.COMPLETED,
            GoalStatus.FAILED,
            GoalStatus.ABANDONED,
        }:
            raise ValueError("Terminal Goal cannot receive a Plan")
        plan = self.plan_decision_coordinator.create_plan(candidate, actor_id=actor_id)
        self._sync_plan_working_memory()
        return plan

    def activate_plan(self, plan_id: str) -> Plan:
        plan = self.plan_store.get(plan_id)
        if self.goal_manager.get(plan.goal_id).status != GoalStatus.ACTIVE:
            raise ValueError("Plan activation requires an active Goal")
        event = current_agent_event()
        plan = self.plan_store.activate(
            plan_id,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        self._sync_plan_working_memory()
        return plan

    def revise_plan(
        self,
        plan_id: str,
        candidate: PlanCandidate,
        *,
        expected_revision: int,
        reason_code: str,
        actor_id: str,
    ) -> Plan:
        event = current_agent_event()
        plan = self.plan_store.revise(
            plan_id,
            candidate,
            expected_revision=expected_revision,
            reason_code=reason_code,
            actor_id=actor_id,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        self._sync_plan_working_memory()
        return plan

    def start_plan_step(self, plan_id: str, step_id: str) -> Plan:
        self._require_active_plan_goal(plan_id)
        event = current_agent_event()
        plan = self.plan_store.start_step(
            plan_id,
            step_id,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        self._sync_plan_working_memory()
        return plan

    def complete_plan_step(
        self,
        plan_id: str,
        step_id: str,
        evidence: tuple[EvidenceReference, ...],
    ) -> Plan:
        self._require_active_plan_goal(plan_id)
        event = current_agent_event()
        plan = self.plan_store.complete_step(
            plan_id,
            step_id,
            evidence,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        if plan.status == PlanStatus.COMPLETED:
            goal = self.goal_manager.get(plan.goal_id)
            if goal.status != GoalStatus.ACTIVE:
                raise ValueError("Completed Plan is inconsistent with Goal lifecycle")
            self.transition_goal(
                plan.goal_id,
                GoalStatus.COMPLETED,
                reason="plan_success_condition_verified",
                outcome=f"plan:{plan.plan_id}@{plan.revision}",
            )
        self._persist_motivation_state()
        self._sync_plan_working_memory()
        return plan

    def fail_plan_step(
        self,
        plan_id: str,
        step_id: str,
        *,
        reason_code: str,
        evidence: tuple[EvidenceReference, ...] = (),
    ) -> Plan:
        self._require_active_plan_goal(plan_id)
        event = current_agent_event()
        plan = self.plan_store.fail_step(
            plan_id,
            step_id,
            reason_code=reason_code,
            evidence=evidence,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        self._sync_plan_working_memory()
        return plan

    def retry_plan_step(self, plan_id: str, step_id: str) -> Plan:
        return self.start_plan_step(plan_id, step_id)

    def timeout_plan_step(self, plan_id: str, step_id: str) -> Plan:
        event = current_agent_event()
        reference = (
            f"event:{event.event_id}"
            if event is not None
            else f"step:{step_id}:timeout"
        )
        return self.fail_plan_step(
            plan_id,
            step_id,
            reason_code="step_timeout",
            evidence=(
                EvidenceReference(
                    reference=reference,
                    evidence_type="failure",
                    observation_code="step_timeout",
                ),
            ),
        )

    def abandon_plan(
        self,
        plan_id: str,
        evidence: tuple[EvidenceReference, ...],
    ) -> Plan:
        event = current_agent_event()
        plan = self.plan_store.abandon(
            plan_id,
            evidence,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        self._sync_plan_working_memory()
        return plan

    def reevaluate_goals(self) -> list[GoalDecision]:
        event = current_agent_event()
        previous_statuses = {
            goal.goal_id: goal.status for goal in self.goal_manager.goals.values()
        }
        decisions = self.goal_manager.reevaluate(
            value_scores=self._goal_value_scores(),
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        for goal in self.goal_manager.goals.values():
            if (
                previous_statuses.get(goal.goal_id) != GoalStatus.FAILED
                and goal.status == GoalStatus.FAILED
            ):
                self._apply_goal_outcome_appraisal(goal, GoalStatus.FAILED)
                self._sync_commitment_from_goal(
                    goal, GoalStatus.FAILED, "deadline_expired", None
                )
                self.motivation_dynamics.resolve_goal(goal.goal_id, success=False)
        self._sync_motivation_working_memory()
        self._persist_motivation_state()
        return decisions

    def goal_decision_input(self) -> GoalDecisionInput:
        return self.goal_manager.decision_input(
            value_scores=self._goal_value_scores(),
            active_commitment_ids=(
                item.commitment_id
                for item in self.commitment_store.list_commitments()
                if item.status in ACCEPTED_COMMITMENT_STATUSES
            ),
        )

    def create_commitment(
        self,
        *,
        description: str,
        deadline: str | None = None,
        beneficiary: str = "self",
        scope: str | None = None,
        cost: float = 0.0,
        burden: float = 0.0,
        fulfillability: CommitmentFulfillability = CommitmentFulfillability.UNKNOWN,
        fulfillability_reason: str | None = None,
        related_desire_ids: tuple[str, ...] = (),
        conflicting_desire_ids: tuple[str, ...] = (),
        conflicting_value_ids: tuple[str, ...] = (),
        conflicting_commitment_ids: tuple[str, ...] = (),
        commitment_id: str | None = None,
        origin_actor: OriginActor | None = None,
        origin_source_ref: str | None = None,
        interlocutor_key: str | None = None,
    ) -> Commitment:
        event = current_agent_event()
        identifier = commitment_id or str(uuid4())
        if deadline is not None:
            parsed_deadline = datetime.fromisoformat(deadline)
            if parsed_deadline.tzinfo is None:
                raise ValueError("Commitment deadline must include a timezone")
            if parsed_deadline <= datetime.now(UTC):
                raise ValueError("Commitment deadline has expired")
        for desire_id in (*related_desire_ids, *conflicting_desire_ids):
            motivation = self.motivation_dynamics.get(desire_id)
            if motivation.kind.value != "desire":
                raise ValueError(f"Motivation is not a Desire: {desire_id}")
        for value_id in conflicting_value_ids:
            self.value_system.get(value_id)
        for other_id in conflicting_commitment_ids:
            self.commitment_store.get(other_id)
        identity_origin = new_identity_origin(
            origin_actor or OriginActor.SELF,
            OriginInputKind.REQUEST
            if origin_actor not in {None, OriginActor.SELF}
            else OriginInputKind.INTERNAL_STATE,
            source_ref=origin_source_ref or "runtime:commitment_proposal",
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        relationship_refs: tuple[str, ...] = ()
        if interlocutor_key is not None:
            relationship = self.relationship_store.ensure_interlocutor(interlocutor_key)
            relationship_refs = (f"relationship:{relationship.relationship_id}",)
        commitment = self.commitment_store.request(
            description=description,
            beneficiary=beneficiary,
            scope=scope or description,
            origin_event_id=None if event is None else event.event_id,
            identity_origin=identity_origin,
            deadline=deadline,
            cost=cost,
            burden=burden,
            fulfillability=fulfillability,
            fulfillability_reason=fulfillability_reason,
            related_desire_ids=related_desire_ids,
            relationship_refs=relationship_refs,
            conflicting_desire_ids=conflicting_desire_ids,
            conflicting_value_ids=conflicting_value_ids,
            conflicting_commitment_ids=conflicting_commitment_ids,
            commitment_id=identifier,
        )
        self._persist_motivation_state()
        return commitment

    def accept_commitment(
        self,
        commitment_id: str,
        *,
        self_endorsement: str,
        priority: float = 0.7,
        urgency: float = 0.7,
        expected_utility: float = 0.7,
        confidence: float = 0.8,
        value_effects: dict[str, float] | None = None,
        conflict_ids: tuple[str, ...] = (),
    ) -> Commitment:
        event = current_agent_event()
        commitment = self.commitment_store.get(commitment_id)
        if commitment.status != CommitmentStatus.PROPOSED:
            raise ValueError("Only proposed commitments can be accepted")
        if commitment.deadline is not None and datetime.fromisoformat(
            commitment.deadline
        ) <= datetime.now(UTC):
            raise ValueError("Commitment deadline has expired")
        commitment.identity_origin.endorse(
            self_endorsement,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        interlocutor_key = None
        if commitment.relationship_refs:
            relationship_id = commitment.relationship_refs[0].removeprefix(
                "relationship:"
            )
            relationship = self.relationship_store.get(relationship_id)
            interlocutor_key = relationship.interlocutor_keys[0]
        goal = self.propose_goal(
            goal_type=GoalType.COMMITMENT,
            description=commitment.description,
            structured_target={
                "commitment_id": commitment_id,
                **(
                    {}
                    if interlocutor_key is None
                    else {"interlocutor_key": interlocutor_key}
                ),
            },
            origin_actor=commitment.identity_origin.actor,
            origin_input_kind=commitment.identity_origin.input_kind,
            origin_source_ref=commitment.identity_origin.source_ref,
            priority=priority,
            urgency=urgency,
            expected_utility=expected_utility,
            confidence=confidence,
            conflict_ids=conflict_ids,
            deadline=commitment.deadline,
            value_effects=value_effects,
            related_desire_ids=commitment.related_desire_ids,
            goal_id=f"commitment:{commitment_id}",
        )
        self.adopt_goal(goal.goal_id)
        if self.goal_manager.get(goal.goal_id).status != GoalStatus.ACTIVE:
            raise ValueError("Commitment intention was not endorsed for activation")
        accepted = self.commitment_store.accept(
            commitment_id,
            acceptance_ref=self_endorsement,
            related_goal_id=goal.goal_id,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        if interlocutor_key is not None:
            self.relationship_store.link_commitment(
                interlocutor_key, f"commitment:{commitment_id}"
            )
        self._sync_self_references()
        self._persist_self_model_state()
        self._sync_motivation_working_memory()
        self._persist_motivation_state()
        return accepted

    def reassess_commitment(
        self,
        commitment_id: str,
        *,
        fulfillability: CommitmentFulfillability,
        reason: str,
    ) -> Commitment:
        event = current_agent_event()
        current = self.commitment_store.get(commitment_id)
        decision_ref = None
        unresolved_decision = any(
            (record := self.decision_store.records.get(ref.removeprefix("decision:")))
            is not None
            and record.status == DecisionStatus.AWAITING_OUTCOME
            for ref in current.decision_refs
        )
        if (
            fulfillability == CommitmentFulfillability.IMPOSSIBLE
            and not unresolved_decision
        ):
            candidates = [
                ActionCandidate(
                    candidate_id=f"commitment:{commitment_id}:renegotiate",
                    candidate_type=ActionType.RESPOND,
                    proposed_action="renegotiate_commitment",
                    parameters={"commitment_id": commitment_id},
                    prerequisites=(),
                    predicted_outcomes=(
                        PredictedOutcome(
                            outcome_id="revised_terms",
                            description="Beneficiary agrees to revised terms",
                            probability=0.6,
                            utility=0.6,
                        ),
                    ),
                    uncertainty=0.4,
                    estimated_cost=0.3,
                    estimated_risk=0.2,
                    value_effects={},
                    appraisal_contributions={"accountability": 0.7},
                ),
                ActionCandidate(
                    candidate_id=f"commitment:{commitment_id}:notify",
                    candidate_type=ActionType.RESPOND,
                    proposed_action="notify_beneficiary_of_impossibility",
                    parameters={"commitment_id": commitment_id},
                    prerequisites=(),
                    predicted_outcomes=(
                        PredictedOutcome(
                            outcome_id="beneficiary_informed",
                            description="Beneficiary receives timely notice",
                            probability=0.9,
                            utility=0.4,
                        ),
                    ),
                    uncertainty=0.1,
                    estimated_cost=0.1,
                    estimated_risk=0.1,
                    value_effects={},
                    appraisal_contributions={"accountability": 0.8},
                ),
                ActionCandidate(
                    candidate_id=f"commitment:{commitment_id}:defer",
                    candidate_type=ActionType.DEFER,
                    proposed_action="defer_for_new_evidence",
                    parameters={"commitment_id": commitment_id},
                    prerequisites=(),
                    predicted_outcomes=(),
                    uncertainty=0.8,
                    estimated_cost=0.4,
                    estimated_risk=0.8,
                    value_effects={},
                    appraisal_contributions={"accountability": -0.5},
                ),
            ]
            decision = self.create_decision(
                candidates,
                decision_id=f"commitment-impossible:{commitment_id}:{uuid4()}",
            )
            decision_ref = f"decision:{decision.decision_id}"
        updated = self.commitment_store.reassess(
            commitment_id,
            fulfillability=fulfillability,
            reason=reason,
            decision_ref=decision_ref,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        return updated

    def renegotiate_commitment(
        self,
        commitment_id: str,
        *,
        reason: str,
        proposed_scope: str | None = None,
        proposed_deadline: str | None = None,
    ) -> Commitment:
        event = current_agent_event()
        updated = self.commitment_store.renegotiate(
            commitment_id,
            reason=reason,
            proposed_scope=proposed_scope,
            proposed_deadline=proposed_deadline,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        return updated

    def repair_commitment(
        self,
        commitment_id: str,
        *,
        reason: str,
        evidence_refs: tuple[str, ...],
    ) -> Commitment:
        event = current_agent_event()
        current = self.commitment_store.get(commitment_id)
        if current.status != CommitmentStatus.BREACHED:
            raise ValueError("Only breached commitments can be repaired")
        updated = self.commitment_store.record_accountability(
            commitment_id,
            action=CommitmentLifecycleAction.REPAIR,
            reason=reason,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self.relationship_store.transition_commitment(
            f"commitment:{commitment_id}",
            status="repaired",
            evidence_ref=evidence_refs[0],
        )
        self.narrative_self.record_commitment_event(
            f"commitment:{commitment_id}",
            kind="repair",
            description=reason,
            evidence_refs=evidence_refs,
            relationship_refs=current.relationship_refs,
        )
        self._persist_narrative_self_state()
        self._persist_motivation_state()
        return updated

    def transition_commitment(
        self,
        commitment_id: str,
        status: CommitmentStatus,
        *,
        reason: str,
        outcome: str | None = None,
    ) -> Commitment:
        if status not in {
            CommitmentStatus.FULFILLED,
            CommitmentStatus.RELEASED,
            CommitmentStatus.BREACHED,
        }:
            raise ValueError("Use the dedicated commitment lifecycle operation")
        event = current_agent_event()
        commitment = self.commitment_store.get(commitment_id)
        goal = (
            None
            if commitment.related_goal_id is None
            else self.goal_manager.goals.get(commitment.related_goal_id)
        )
        if goal is not None and goal.status not in {
            GoalStatus.COMPLETED,
            GoalStatus.ABANDONED,
            GoalStatus.FAILED,
        }:
            mapped_status = {
                CommitmentStatus.FULFILLED: GoalStatus.COMPLETED,
                CommitmentStatus.RELEASED: GoalStatus.ABANDONED,
                CommitmentStatus.BREACHED: GoalStatus.FAILED,
            }[status]
            self.transition_goal(
                goal.goal_id,
                mapped_status,
                reason=f"commitment_{status.value}:{reason}",
                outcome=outcome,
            )
            return self.commitment_store.get(commitment_id)
        commitment = self.commitment_store.transition(
            commitment_id,
            status,
            reason=reason,
            outcome=outcome,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self.relationship_store.transition_commitment(
            f"commitment:{commitment_id}",
            status=status.value,
            evidence_ref=f"commitment:{commitment_id}:{status.value}",
        )
        if status == CommitmentStatus.BREACHED:
            self.narrative_self.record_commitment_event(
                f"commitment:{commitment_id}",
                kind="breach",
                description=reason,
                evidence_refs=(f"commitment:{commitment_id}:breached",),
                relationship_refs=commitment.relationship_refs,
            )
            self._persist_narrative_self_state()
        self._sync_motivation_working_memory()
        self._persist_motivation_state()
        return commitment

    def _goal_value_scores(self) -> dict[str, float]:
        options = {
            goal.goal_id: goal.value_effects
            for goal in self.goal_manager.goals.values()
            if goal.value_effects
        }
        if not options:
            return {}
        return {
            score.option_id: score.total_score
            for score in self.value_system.evaluate(options)
        }

    def _apply_goal_outcome_appraisal(self, goal: Goal, status: GoalStatus) -> None:
        progress = {
            GoalStatus.COMPLETED: 1.0,
            GoalStatus.FAILED: -1.0,
            GoalStatus.ABANDONED: -0.5,
        }.get(status)
        if progress is None:
            return
        self.emotion_engine.update_from_appraisal(
            AppraisalResult(
                novelty=None,
                goal_progress=progress,
                threat=0.0,
                controllability=0.5,
                certainty=goal.confidence,
                social_relevance=0.0,
                effort_cost=0.0,
                novelty_valid=False,
                reasons=(f"goal_{status.value}",),
            )
        )

    def _sync_commitment_from_goal(
        self,
        goal: Goal,
        status: GoalStatus,
        reason: str,
        outcome: str | None,
    ) -> None:
        commitment = next(
            (
                item
                for item in self.commitment_store.commitments.values()
                if item.related_goal_id == goal.goal_id
                and item.status
                in {CommitmentStatus.ACTIVE, CommitmentStatus.RENEGOTIATING}
            ),
            None,
        )
        mapped = {
            GoalStatus.COMPLETED: CommitmentStatus.FULFILLED,
            GoalStatus.ABANDONED: CommitmentStatus.RELEASED,
            GoalStatus.FAILED: CommitmentStatus.BREACHED,
        }.get(status)
        if commitment is None or mapped is None:
            return
        event = current_agent_event()
        self.commitment_store.transition(
            commitment.commitment_id,
            mapped,
            reason=f"goal_{status.value}:{reason}",
            outcome=outcome,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self.relationship_store.transition_commitment(
            f"commitment:{commitment.commitment_id}",
            status=mapped.value,
            evidence_ref=f"commitment:{commitment.commitment_id}:{mapped.value}",
        )
        if mapped == CommitmentStatus.BREACHED:
            self.narrative_self.record_commitment_event(
                f"commitment:{commitment.commitment_id}",
                kind="breach",
                description=reason,
                evidence_refs=(f"commitment:{commitment.commitment_id}:breached",),
                relationship_refs=commitment.relationship_refs,
            )
            self._persist_narrative_self_state()

    def _sync_motivation_working_memory(self) -> None:
        self._sync_plan_goal_statuses()
        for goal in self.goal_manager.goals.values():
            item_id = f"goal:{goal.goal_id}"
            if goal.status == GoalStatus.ACTIVE:
                self.working_memory.admit(
                    working_memory_item(
                        item_id=item_id,
                        kind=WorkingMemoryKind.GOAL,
                        content=goal.description,
                        activation=0.9,
                        salience=max(goal.priority, goal.urgency),
                        retention_reason=RetentionReason.ONGOING_GOAL,
                        source="runtime.goal_manager",
                    )
                )
            else:
                self.working_memory.forget(item_id)
        for commitment in self.commitment_store.commitments.values():
            item_id = f"commitment:{commitment.commitment_id}"
            if commitment.status in ACCEPTED_COMMITMENT_STATUSES:
                self.working_memory.admit(
                    working_memory_item(
                        item_id=item_id,
                        kind=WorkingMemoryKind.COMMITMENT,
                        content=commitment.description,
                        activation=0.9,
                        salience=0.9,
                        retention_reason=RetentionReason.ACTIVE_COMMITMENT,
                        source="runtime.commitment_store",
                    )
                )
            else:
                self.working_memory.forget(item_id)
        self._sync_plan_working_memory()

    def _require_active_plan_goal(self, plan_id: str) -> None:
        plan = self.plan_store.get(plan_id)
        if self.goal_manager.get(plan.goal_id).status != GoalStatus.ACTIVE:
            raise ValueError("Step lifecycle requires an active Goal")

    def _sync_plan_goal_statuses(self) -> None:
        event = current_agent_event()
        event_id = None if event is None else event.event_id
        event_sequence = None if event is None else event.processing_sequence
        for plan in self.plan_store.list_plans():
            goal = self.goal_manager.get(plan.goal_id)
            if plan.status == PlanStatus.ACTIVE and goal.status != GoalStatus.ACTIVE:
                self.plan_store.pause(
                    plan.plan_id,
                    event_id=event_id,
                    event_sequence=event_sequence,
                )
            elif plan.status == PlanStatus.PAUSED and goal.status == GoalStatus.ACTIVE:
                self.plan_store.resume(
                    plan.plan_id,
                    event_id=event_id,
                    event_sequence=event_sequence,
                )

    def _sync_plan_working_memory(self) -> None:
        for item in self.working_memory.items:
            if item.kind == WorkingMemoryKind.STEP:
                self.working_memory.forget(item.item_id)
        for plan, step, _state in self.plan_store.actionable_steps():
            self.working_memory.admit(
                working_memory_item(
                    item_id=f"plan-step:{plan.plan_id}:{step.step_id}",
                    kind=WorkingMemoryKind.STEP,
                    content=json.dumps(
                        {
                            "plan_id": plan.plan_id,
                            "plan_revision": plan.revision,
                            "step_id": step.step_id,
                            "action_type": step.action_type.value,
                            "action_code": step.action_code,
                            "expected_observation": (
                                step.expected_observation.observation_code
                            ),
                            "verification": step.verification.verification_code,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                    activation=1.0,
                    salience=1.0,
                    retention_reason=RetentionReason.ACTIONABLE_STEP,
                    source="runtime.plan_store",
                )
            )

    def restore_decision_state(self) -> None:
        payload = self.persistent_state.extensions.get("decision_records", [])
        self.decision_store.restore(payload if isinstance(payload, list) else [])
        self._persist_decision_state()

    def restore_feedback_state(self) -> None:
        payload = self.persistent_state.extensions.get("feedback")
        self.feedback_store.restore(payload if isinstance(payload, dict) else None)
        self._persist_feedback_state()

    def _persist_feedback_state(self) -> None:
        self.persistent_state.extensions["feedback"] = self.feedback_store.to_json()

    def restore_metacognition_state(self) -> None:
        self.metacognition.restore(
            self.persistent_state.extensions.get("metacognition")
        )
        self._persist_metacognition_state()

    def _persist_metacognition_state(self) -> None:
        self.persistent_state.extensions["metacognition"] = self.metacognition.to_json()

    def submit_feedback(
        self,
        *,
        target: FeedbackTarget,
        signals: tuple[FeedbackSignal, ...],
        idempotency_key: str,
        actor_type: str,
        actor_id: str | None,
        source: str,
        correction: str | None = None,
        expected_answer: str | None = None,
        feedback_id: str | None = None,
    ) -> FeedbackRecord:
        normalized = normalize_signals(signals)
        self._validate_feedback_content(normalized, correction, expected_answer)
        fingerprint = feedback_fingerprint(
            "create",
            {
                "feedback_id": feedback_id,
                "target": asdict(target),
                "signals": [item.value for item in normalized],
                "correction": correction,
                "expected_answer": expected_answer,
                "actor_type": actor_type,
            },
        )
        existing = self.feedback_store.idempotent_result(idempotency_key, fingerprint)
        if existing is not None:
            return existing
        identifier = feedback_id or f"feedback-{uuid4()}"
        if identifier in self.feedback_store.records:
            raise ValueError(f"Feedback already exists: {identifier}")
        self._validate_feedback_target(target)
        revision = 1
        propagation, correction_id, expected_id = self._propagate_feedback(
            identifier,
            revision,
            target,
            normalized,
            correction=correction,
            expected_answer=expected_answer,
        )
        record = self.feedback_store.create(
            signals=normalized,
            target=target,
            provenance=self._feedback_provenance(actor_type, actor_id, source),
            correction_memory_id=correction_id,
            expected_answer_memory_id=expected_id,
            propagation=propagation,
            idempotency_key=idempotency_key,
            fingerprint=fingerprint,
            feedback_id=identifier,
        )
        self._persist_feedback_state()
        self._persist_decision_state()
        self._persist_metacognition_state()
        return record

    def revise_feedback(
        self,
        feedback_id: str,
        *,
        expected_revision: int,
        signals: tuple[FeedbackSignal, ...],
        idempotency_key: str,
        actor_type: str,
        actor_id: str | None,
        source: str,
        correction: str | None = None,
        expected_answer: str | None = None,
    ) -> FeedbackRecord:
        normalized = normalize_signals(signals)
        self._validate_feedback_content(normalized, correction, expected_answer)
        fingerprint = feedback_fingerprint(
            "revise",
            {
                "feedback_id": feedback_id,
                "expected_revision": expected_revision,
                "signals": [item.value for item in normalized],
                "correction": correction,
                "expected_answer": expected_answer,
                "actor_type": actor_type,
            },
        )
        existing = self.feedback_store.idempotent_result(idempotency_key, fingerprint)
        if existing is not None:
            return existing
        current = self.feedback_store.get(feedback_id)
        if current.current_revision != expected_revision:
            raise ValueError(
                f"Feedback revision conflict: expected {expected_revision}, "
                f"current {current.current_revision}"
            )
        self._withdraw_feedback_effects(current.current)
        propagation, correction_id, expected_id = self._propagate_feedback(
            feedback_id,
            expected_revision + 1,
            current.current.target,
            normalized,
            correction=correction,
            expected_answer=expected_answer,
        )
        record = self.feedback_store.revise(
            feedback_id,
            expected_revision=expected_revision,
            signals=normalized,
            target=current.current.target,
            provenance=self._feedback_provenance(actor_type, actor_id, source),
            correction_memory_id=correction_id,
            expected_answer_memory_id=expected_id,
            propagation=propagation,
            idempotency_key=idempotency_key,
            fingerprint=fingerprint,
        )
        self._persist_feedback_state()
        self._persist_decision_state()
        self._persist_metacognition_state()
        return record

    def withdraw_feedback(
        self,
        feedback_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        actor_type: str,
        actor_id: str | None,
        source: str,
    ) -> FeedbackRecord:
        fingerprint = feedback_fingerprint(
            "withdraw",
            {
                "feedback_id": feedback_id,
                "expected_revision": expected_revision,
                "actor_type": actor_type,
            },
        )
        existing = self.feedback_store.idempotent_result(idempotency_key, fingerprint)
        if existing is not None:
            return existing
        current = self.feedback_store.get(feedback_id)
        if current.current_revision != expected_revision:
            raise ValueError(
                f"Feedback revision conflict: expected {expected_revision}, "
                f"current {current.current_revision}"
            )
        self._withdraw_feedback_effects(current.current)
        propagation = FeedbackPropagation(
            memory_id=current.current.propagation.memory_id,
            correction_memory_id=None,
            memory_before=current.current.propagation.memory_after,
            memory_after=current.current.propagation.memory_before,
            decision_id=current.current.propagation.decision_id,
            decision_outcome_applied=False,
            prediction_error=None,
            value_evidence=None,
            training_disposition=TrainingDisposition.INCLUDE,
            exclusion_refs=(),
            reason_codes=("feedback_withdrawn",),
        )
        record = self.feedback_store.revise(
            feedback_id,
            expected_revision=expected_revision,
            signals=(),
            target=current.current.target,
            provenance=self._feedback_provenance(actor_type, actor_id, source),
            correction_memory_id=None,
            expected_answer_memory_id=None,
            propagation=propagation,
            idempotency_key=idempotency_key,
            fingerprint=fingerprint,
            status=FeedbackStatus.WITHDRAWN,
        )
        self._persist_feedback_state()
        self._persist_decision_state()
        self._persist_metacognition_state()
        return record

    def _propagate_feedback(
        self,
        feedback_id: str,
        revision: int,
        target: FeedbackTarget,
        signals: tuple[FeedbackSignal, ...],
        *,
        correction: str | None,
        expected_answer: str | None,
    ) -> tuple[FeedbackPropagation, str | None, str | None]:
        event = current_agent_event()
        memory_id = self._feedback_memory_id(target)
        before: dict[str, str] = {}
        after: dict[str, str] = {}
        correction_id: str | None = None
        expected_id: str | None = None
        exclude = bool(
            set(signals)
            & {
                FeedbackSignal.BAD,
                FeedbackSignal.FACTUAL_ERROR,
                FeedbackSignal.STYLE_PROBLEM,
                FeedbackSignal.UNSAFE_BEHAVIOR,
                FeedbackSignal.DO_NOT_REMEMBER,
                FeedbackSignal.EXCLUDE_FROM_TRAINING,
            }
        )
        if memory_id is not None:
            memory = self.memory_system.get_episodic(memory_id)
            if memory is None:
                raise ValueError(f"Unknown episodic memory: {memory_id}")
            before = {
                "validation_status": memory.validation_status.value,
                "lifecycle_status": memory.lifecycle_status.value,
                "training_included": str(memory.training_included).lower(),
            }
            validation = memory.validation_status
            lifecycle = memory.lifecycle_status
            if FeedbackSignal.DO_NOT_REMEMBER in signals:
                validation = ValidationStatus.REJECTED
                lifecycle = MemoryLifecycleStatus.REJECTED
            elif (
                FeedbackSignal.CORRECTION in signals
                or FeedbackSignal.EXPECTED_ANSWER in signals
                or FeedbackSignal.FACTUAL_ERROR in signals
            ):
                validation = ValidationStatus.DISPUTED
                lifecycle = (
                    MemoryLifecycleStatus.CORRECTED
                    if correction
                    else MemoryLifecycleStatus.QUARANTINED
                )
            elif set(signals) & {
                FeedbackSignal.BAD,
                FeedbackSignal.STYLE_PROBLEM,
                FeedbackSignal.UNSAFE_BEHAVIOR,
            }:
                validation = ValidationStatus.DISPUTED
                lifecycle = MemoryLifecycleStatus.QUARANTINED
            elif FeedbackSignal.REMEMBER in signals:
                lifecycle = MemoryLifecycleStatus.ACTIVE
            if correction is not None:
                correction_id = self.memory_system.save_feedback_correction(
                    memory_id,
                    correction,
                    feedback_id=feedback_id,
                    kind="correction",
                )
            if expected_answer is not None:
                expected_id = self.memory_system.save_feedback_correction(
                    memory_id,
                    expected_answer,
                    feedback_id=feedback_id,
                    kind="expected_answer",
                )
            updated = self.memory_system.apply_feedback_policy(
                memory_id,
                validation_status=validation,
                lifecycle_status=lifecycle,
                training_included=not exclude,
                feedback_id=feedback_id,
            )
            if FeedbackSignal.DO_NOT_REMEMBER in signals:
                for linked_id in (correction_id, expected_id):
                    if linked_id is not None:
                        self.memory_system.apply_feedback_policy(
                            linked_id,
                            validation_status=ValidationStatus.REJECTED,
                            lifecycle_status=MemoryLifecycleStatus.REJECTED,
                            training_included=False,
                            feedback_id=feedback_id,
                        )
            after = {
                "validation_status": updated.validation_status.value,
                "lifecycle_status": updated.lifecycle_status.value,
                "training_included": str(updated.training_included).lower(),
            }
        decision_id = target.decision_id or (
            target.target_id
            if target.target_type == FeedbackTargetType.DECISION
            else None
        )
        prediction_error: float | None = None
        outcome_applied = False
        if decision_id is not None and set(signals) & {
            FeedbackSignal.GOOD,
            FeedbackSignal.BAD,
            FeedbackSignal.FACTUAL_ERROR,
            FeedbackSignal.STYLE_PROBLEM,
            FeedbackSignal.UNSAFE_BEHAVIOR,
        }:
            utility = self._feedback_utility(signals)
            decision = self.decision_store.record_feedback_outcome(
                decision_id,
                utility=utility,
                success=utility > 0.0,
                feedback_id=feedback_id,
                feedback_revision=revision,
                observed_event_id=None if event is None else event.event_id,
                observed_event_sequence=None
                if event is None
                else event.processing_sequence,
            )
            decision = self._apply_metacognitive_outcome(decision)
            prediction_error = decision.prediction_error
            outcome_applied = True
        if decision_id is not None:
            self.decision_store.set_training_policy(
                decision_id, included=not exclude, feedback_id=feedback_id
            )
        direction = "supporting" if self._feedback_utility(signals) > 0 else "opposing"
        value_impacts: dict[str, float] = {}
        if decision_id is not None:
            decision = self.decision_store.get(decision_id)
            selected = next(
                item.candidate
                for item in decision.considered_candidates
                if item.candidate.candidate_id == decision.selected_candidate_id
            )
            utility = self._feedback_utility(signals)
            value_impacts = {
                value_id: max(-1.0, min(1.0, effect * utility))
                for value_id, effect in selected.value_effects.items()
            }
        value_proposal = (
            None
            if not set(signals)
            & {
                FeedbackSignal.GOOD,
                FeedbackSignal.BAD,
                FeedbackSignal.FACTUAL_ERROR,
                FeedbackSignal.STYLE_PROBLEM,
                FeedbackSignal.UNSAFE_BEHAVIOR,
            }
            else ValueEvidenceProposal(
                proposal_id=f"feedback:{feedback_id}:{revision}",
                direction=direction,
                strength=abs(self._feedback_utility(signals)),
                reason_codes=tuple(item.value for item in signals),
                target_refs=(f"{target.target_type.value}:{target.target_id}",),
                value_impacts=value_impacts,
            )
        )
        propagation = FeedbackPropagation(
            memory_id=memory_id,
            correction_memory_id=correction_id or expected_id,
            memory_before=before,
            memory_after=after,
            decision_id=decision_id,
            decision_outcome_applied=outcome_applied,
            prediction_error=prediction_error,
            value_evidence=value_proposal,
            training_disposition=(
                TrainingDisposition.EXCLUDE if exclude else TrainingDisposition.INCLUDE
            ),
            exclusion_refs=((feedback_id,) if exclude else ()),
            reason_codes=tuple(item.value for item in signals),
        )
        self.memory_system.reevaluate_semantics_for_feedback(
            feedback_id, rejected=False
        )
        return propagation, correction_id, expected_id

    def _withdraw_feedback_effects(self, revision: FeedbackRevision) -> None:
        propagation = revision.propagation
        self.memory_system.reevaluate_semantics_for_feedback(
            self._feedback_id_for_revision(revision), rejected=True
        )
        if propagation.memory_id is not None and propagation.memory_before:
            before = propagation.memory_before
            self.memory_system.apply_feedback_policy(
                propagation.memory_id,
                validation_status=ValidationStatus(before["validation_status"]),
                lifecycle_status=MemoryLifecycleStatus(before["lifecycle_status"]),
                training_included=before.get("training_included", "true") == "true",
                feedback_id=self._feedback_id_for_revision(revision),
            )
            for correction_id in (
                revision.correction_memory_id,
                revision.expected_answer_memory_id,
            ):
                if correction_id is not None:
                    self.memory_system.withdraw_feedback_correction(
                        propagation.memory_id, correction_id
                    )
        if propagation.decision_id is not None:
            feedback_id = self._feedback_id_for_revision(revision)
            self.decision_store.withdraw_feedback_outcome(
                propagation.decision_id, feedback_id=feedback_id
            )
            self.decision_store.clear_post_assessment(propagation.decision_id)
            self.metacognition.withdraw_outcome(propagation.decision_id)
            self._sync_metacognitive_narrative()
            self.decision_store.set_training_policy(
                propagation.decision_id, included=True, feedback_id=feedback_id
            )

    def _feedback_id_for_revision(self, revision: FeedbackRevision) -> str:
        for record in self.feedback_store.records.values():
            if revision in record.revisions:
                return record.feedback_id
        raise ValueError("Feedback revision is not attached to a record")

    def _validate_feedback_target(self, target: FeedbackTarget) -> None:
        memory_id = self._feedback_memory_id(target)
        if memory_id is not None:
            memory = self.memory_system.get_episodic(memory_id)
            if memory is None:
                raise ValueError(f"Unknown episodic memory: {memory_id}")
            if target.context_id is not None and memory.context_id != target.context_id:
                raise ValueError("Feedback context does not own the target episode")
            if (
                target.experience_id is not None
                and memory.experience_id != target.experience_id
            ):
                raise ValueError("Feedback experience does not own the target episode")
        if target.target_type == FeedbackTargetType.DECISION:
            self.decision_store.get(target.target_id)
        if target.decision_id is not None:
            self.decision_store.get(target.decision_id)
        if target.target_type == FeedbackTargetType.CONTEXT:
            if self.context_registry.get(target.target_id) is None:
                raise ValueError(f"Unknown context: {target.target_id}")

    @staticmethod
    def _feedback_memory_id(target: FeedbackTarget) -> str | None:
        if target.episode_id is not None:
            return target.episode_id
        if target.target_type in {
            FeedbackTargetType.RESPONSE,
            FeedbackTargetType.EPISODE,
            FeedbackTargetType.MEMORY,
        }:
            return target.target_id
        return None

    @staticmethod
    def _validate_feedback_content(
        signals: tuple[FeedbackSignal, ...],
        correction: str | None,
        expected_answer: str | None,
    ) -> None:
        if FeedbackSignal.CORRECTION in signals and not correction:
            raise ValueError("correction signal requires correction content")
        if FeedbackSignal.EXPECTED_ANSWER in signals and not expected_answer:
            raise ValueError("expected_answer signal requires expected answer content")
        if correction is not None and FeedbackSignal.CORRECTION not in signals:
            raise ValueError("Correction content requires the correction signal")
        if (
            expected_answer is not None
            and FeedbackSignal.EXPECTED_ANSWER not in signals
        ):
            raise ValueError(
                "Expected answer content requires the expected_answer signal"
            )

    @staticmethod
    def _feedback_utility(signals: tuple[FeedbackSignal, ...]) -> float:
        if FeedbackSignal.UNSAFE_BEHAVIOR in signals:
            return -1.0
        if FeedbackSignal.BAD in signals or FeedbackSignal.FACTUAL_ERROR in signals:
            return -0.75
        if FeedbackSignal.STYLE_PROBLEM in signals:
            return -0.4
        if FeedbackSignal.GOOD in signals:
            return 1.0
        return 0.0

    @staticmethod
    def _feedback_provenance(
        actor_type: str, actor_id: str | None, source: str
    ) -> FeedbackProvenance:
        event = current_agent_event()
        return FeedbackProvenance(
            actor_type=actor_type,
            actor_id=actor_id,
            source=source,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
            submitted_at=datetime.now(UTC).isoformat(),
        )

    def _persist_decision_state(self) -> None:
        self.persistent_state.extensions["decision_records"] = (
            self.decision_store.to_json()
        )

    def create_decision(
        self,
        candidates: list[ActionCandidate],
        *,
        context_id: str | None = None,
        satisfied_prerequisites: set[str] | None = None,
        decision_id: str | None = None,
    ) -> DecisionRecord:
        event = current_agent_event()
        for candidate in candidates:
            self.plan_store.validate_candidate(candidate)
        if context_id is not None and self.context_registry.get(context_id) is None:
            raise ValueError(f"Unknown context: {context_id}")
        completed_goals = {
            goal.goal_id
            for goal in self.goal_manager.goals.values()
            if goal.status == GoalStatus.COMPLETED
        }
        completed_steps = {
            f"step:{plan.plan_id}:{plan.revision}:{state.step_id}:completed"
            for plan in self.plan_store.list_plans()
            for state in plan.step_states
            if state.status == StepStatus.COMPLETED
        }
        satisfied_prerequisites = set(satisfied_prerequisites or ()) | completed_steps
        emotion = self.emotion_engine.state
        source_experience = (
            None
            if context_id is None
            else self.experience_store.latest_for_context(context_id)
        )
        active_beliefs = self.belief_store.active(context_id=context_id)
        narrative_claim_ids = tuple(
            dict.fromkeys(
                claim_id
                for candidate in candidates
                for claim_id in self.narrative_self.select_relevant(
                    theme_codes=_string_values(candidate.parameters.get("topic_tags"))
                    + _string_values(candidate.parameters.get("theme_codes")),
                    capability_ids=_string_values(
                        candidate.parameters.get("capability_ids")
                    ),
                ).claim_ids
            )
        )
        decision_context = (
            None if context_id is None else self.context_registry.get(context_id)
        )
        relationship_influence = self.relationship_store.influence(
            () if decision_context is None else decision_context.participant_ids
        )
        relationship_candidates = [
            replace(
                candidate,
                appraisal_contributions={
                    **candidate.appraisal_contributions,
                    "relationship_caution": -relationship_influence.threat
                    * candidate.estimated_risk,
                    "relationship_reciprocity": (
                        relationship_influence.expected_reciprocity - 0.5
                    )
                    * (1.0 - candidate.estimated_risk)
                    if candidate.candidate_type.value
                    in {"respond", "request_information"}
                    else 0.0,
                },
            )
            for candidate in candidates
        ]
        identifier = decision_id or str(uuid4())
        narrative_refs = tuple(
            f"identity-claim:{claim_id}@{self.narrative_self.get_claim(claim_id).revision}"
            for claim_id in narrative_claim_ids
        )
        pre_assessment = self.metacognition.assess_pre(
            identifier,
            relationship_candidates,
            self_model=self.self_model.state,
            narrative_self_refs=narrative_refs,
            cognitive_load=min(
                1.0, len(self.working_memory.items) / self.working_memory.item_capacity
            ),
            attention_saturation=min(
                1.0,
                len(self.attention_system.focus.candidate_ids)
                / self.attention_system.capacity,
            ),
            emotion_valence=emotion.valence,
            emotion_arousal=emotion.arousal,
            quality_provenance_refs=(
                f"working-memory:occupancy:{len(self.working_memory.items)}",
                f"attention-focus@{self.attention_system.focus.revision}",
                "emotion:current",
            ),
        )
        record = self.decision_store.create(
            relationship_candidates,
            triggering_event_id=None if event is None else event.event_id,
            triggering_event_sequence=None
            if event is None
            else event.processing_sequence,
            context_id=context_id,
            active_goal_ids=tuple(
                goal.goal_id for goal in self.goal_manager.list_goals(GoalStatus.ACTIVE)
            ),
            value_revision_refs={
                value.value_id: value.revision
                for value in self.value_system.list_values()
            },
            emotion_snapshot={
                "valence": emotion.valence,
                "arousal": emotion.arousal,
                "optimal_loss": emotion.optimal_loss,
            },
            adapter_id=self.adapter_id,
            adapter_hash=self.adapter_hash,
            activation_sequence=self.activation_sequence,
            identity_origin_refs={
                **{
                    f"goal:{goal.goal_id}": goal.identity_origin.origin_id
                    for goal in self.goal_manager.list_goals(GoalStatus.ACTIVE)
                },
                **{
                    f"value:{value.value_id}": value.origin_provenance.origin_id
                    for value in self.value_system.list_values()
                    if value.origin_provenance is not None
                },
                **{
                    f"belief:{belief.belief_id}": belief.identity_origin.origin_id
                    for belief in active_beliefs
                },
            },
            experience_refs=(
                () if source_experience is None else (source_experience.experience_id,)
            ),
            belief_revision_refs={
                belief.belief_id: belief.revision for belief in active_beliefs
            },
            narrative_self_refs=narrative_refs,
            metacognition_pre_assessment_id=pre_assessment.assessment_id,
            satisfied_prerequisites=completed_goals
            | {
                f"capability:{capability.capability_id}"
                for capability in self.self_model.state.capabilities.values()
                if capability.confidence >= 0.5
            }
            | (satisfied_prerequisites or set()),
            value_evaluator=lambda options: self._decision_value_evaluator(
                options, context_id=context_id
            ),
            self_model_evaluator=self.self_model.evaluate_candidates,
            metacognition_evaluator=lambda values: self._metacognitive_candidate_scores(
                values, pre_assessment.recommended_action
            ),
            decision_id=identifier,
        )
        decision_scores = self.value_system.evaluate(
            {
                candidate.candidate_id: candidate.value_effects
                for candidate in candidates
                if candidate.value_effects
            },
            context_id=context_id,
        )
        tradeoffs = self.value_system.record_tradeoffs(
            decision_scores,
            context_id=context_id,
            decision_id=record.decision_id,
        )
        if tradeoffs:
            record = self.decision_store.link_value_tradeoffs(
                record.decision_id, tuple(item.tradeoff_id for item in tradeoffs)
            )
        if source_experience is not None:
            self.link_experience_result(
                source_experience.experience_id,
                kind="decision",
                reference=f"decision:{record.decision_id}",
                evidence_refs=(f"decision:{record.decision_id}",),
            )
        self._sync_self_model_working_memory(candidates)
        self._persist_decision_state()
        self._persist_metacognition_state()
        return record

    def correct_relationship(
        self,
        relationship_id: str,
        *,
        reason: str,
        evidence_refs: tuple[str, ...],
        **updates: Any,
    ) -> RelationshipState:
        event = current_agent_event()
        return self.relationship_store.correct(
            relationship_id,
            reason=reason,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
            **updates,
        )

    def attach_relationship_alias(
        self,
        relationship_id: str,
        interlocutor_key: str,
        *,
        evidence_refs: tuple[str, ...],
    ) -> RelationshipState:
        event = current_agent_event()
        return self.relationship_store.attach_alias(
            relationship_id,
            interlocutor_key,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )

    def split_relationship_alias(
        self,
        relationship_id: str,
        interlocutor_key: str,
        *,
        reason: str,
        evidence_refs: tuple[str, ...],
    ) -> RelationshipState:
        event = current_agent_event()
        return self.relationship_store.split_alias(
            relationship_id,
            interlocutor_key,
            reason=reason,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )

    def _relationship_for_target(
        self, target: dict[str, Any]
    ) -> RelationshipState | None:
        key = target.get("interlocutor_key")
        return (
            self.relationship_store.for_interlocutor(key)
            if isinstance(key, str)
            else None
        )

    def _public_subject_summary(
        self, context_id: str, interlocutor_keys: tuple[str, ...]
    ) -> PublicSubjectSummary:
        relationships: list[str] = []
        for key in interlocutor_keys:
            state = self.relationship_store.for_interlocutor(key)
            if state is None:
                continue
            relationships.append(
                f"trust={state.axes.trust:.3f}; familiarity={state.axes.familiarity:.3f}; "
                f"closeness={state.axes.closeness:.3f}; caution={state.axes.caution:.3f}; "
                f"reciprocity={state.reciprocity:.3f}; uncertainty={state.uncertainty:.3f}"
            )
        quality = self._current_cognitive_quality()
        return PublicSubjectSummary(
            values=tuple(
                f"{value.name}; importance={value.weight:.3f}; "
                f"confidence={value.confidence:.3f}; protectedness={value.protectedness:.3f}"
                for value in self.value_system.list_values()[:5]
            ),
            goals=tuple(
                f"{goal.description}; status={goal.status.value}; "
                f"priority={goal.priority:.3f}; confidence={goal.confidence:.3f}"
                for goal in self.goal_manager.list_goals(GoalStatus.ACTIVE)[:5]
            ),
            commitments=tuple(
                f"{commitment.description}; status={commitment.status.value}; "
                f"fulfillability={commitment.fulfillability.value}"
                for commitment in self.commitment_store.list_commitments()
                if commitment.status in ACCEPTED_COMMITMENT_STATUSES
            )[:5],
            relationships=tuple(relationships[:5]),
            beliefs=tuple(
                f"{belief.proposition.normalized}; status={belief.epistemic_status.value}; "
                f"confidence={belief.confidence:.3f}"
                for belief in self.belief_store.active(context_id=context_id)[:5]
            ),
            metacognition=(
                f"estimated_quality={quality.estimated_quality:.3f}; "
                f"cognitive_load={quality.cognitive_load:.3f}; "
                f"attention_saturation={quality.attention_saturation:.3f}",
            ),
        )

    def generate_decision_candidates(self, situation: str) -> list[ActionCandidate]:
        raw = self.provider.generate(schema_candidate_prompt(situation))
        return parse_candidate_output(raw)

    def current_plan_candidates(self) -> list[ActionCandidate]:
        return [
            self.plan_store.action_candidate(plan.plan_id, step.step_id)
            for plan, step, _state in self.plan_store.actionable_steps()
        ]

    def create_plan_action_decision(self, plan_id: str, step_id: str) -> DecisionRecord:
        plan = self.plan_store.get(plan_id)
        decision_id = f"plan-action:{uuid5(NAMESPACE_URL, f'{plan_id}@{plan.revision}/{step_id}')}"
        existing = self.decision_store.records.get(decision_id)
        if existing is not None:
            return existing
        candidate = self.plan_store.action_candidate(plan_id, step_id)
        fallback = ActionCandidate(
            candidate_id=f"{decision_id}:no-op",
            candidate_type=ActionType.NO_OP,
            proposed_action="do_not_execute_plan_step",
            parameters={},
            prerequisites=(),
            predicted_outcomes=(
                PredictedOutcome(
                    outcome_id="plan_step_not_executed",
                    description="Plan step remains pending",
                    probability=1.0,
                    utility=-1.0,
                ),
            ),
            uncertainty=0.0,
            estimated_cost=0.0,
            estimated_risk=0.0,
            value_effects={},
            appraisal_contributions={},
        )
        return self.create_decision(
            [candidate, fallback],
            decision_id=decision_id,
        )

    def start_action_plan_step(self, plan_id: str, step_id: str) -> None:
        state = self.plan_store.get(plan_id).step_state(step_id)
        if state.status == StepStatus.READY:
            self.start_plan_step(plan_id, step_id)
        elif state.status != StepStatus.IN_PROGRESS:
            raise ValueError("Action Plan step is not executable")

    def record_action_plan_observation(
        self,
        plan_id: str,
        step_id: str,
        observation_id: str,
        observation_code: str,
    ) -> Plan:
        plan = self.plan_store.get(plan_id)
        definition = plan.step_definition(step_id)
        evidence_types = tuple(
            dict.fromkeys(
                (
                    *definition.expected_observation.evidence_types,
                    *definition.verification.required_evidence_types,
                )
            )
        )
        return self.complete_plan_step(
            plan_id,
            step_id,
            tuple(
                EvidenceReference(
                    reference=f"action-observation:{observation_id}",
                    evidence_type=evidence_type,
                    observation_code=observation_code,
                )
                for evidence_type in evidence_types
            ),
        )

    def record_decision_outcome(
        self,
        decision_id: str,
        *,
        description: str,
        utility: float,
        success: bool,
    ) -> DecisionRecord:
        event = current_agent_event()
        record = self.decision_store.record_outcome(
            decision_id,
            description=description,
            utility=utility,
            success=success,
            observed_event_id=None if event is None else event.event_id,
            observed_event_sequence=None
            if event is None
            else event.processing_sequence,
        )
        record = self._apply_metacognitive_outcome(record)
        proposals = self.value_system.proposals_from_decision_outcome(record)
        updates = self.value_system.apply(proposals)
        self.value_system.record_reassessment(record, updates)
        self._persist_value_state()
        self._sync_self_references()
        self._persist_self_model_state()
        self._persist_decision_state()
        self._persist_metacognition_state()
        return record

    def record_decision_compensation(
        self, decision_id: str, *, receipt_id: str
    ) -> DecisionRecord:
        event = current_agent_event()
        record = self.decision_store.record_compensation(
            decision_id,
            receipt_id=receipt_id,
            observed_event_id=None if event is None else event.event_id,
            observed_event_sequence=None
            if event is None
            else event.processing_sequence,
        )
        self._persist_decision_state()
        return record

    def _metacognitive_candidate_scores(
        self, candidates: tuple[ActionCandidate, ...], recommended_action: ActionType
    ) -> dict[str, dict[str, float]]:
        scores: dict[str, dict[str, float]] = {
            candidate.candidate_id: {} for candidate in candidates
        }
        for candidate in candidates:
            contribution = 0.0
            if candidate.candidate_type == recommended_action:
                contribution = 0.75
            elif recommended_action.value in {
                "defer",
                "request_information",
                "delegate",
                "observe",
            } and candidate.candidate_type.value in {"respond", "internal"}:
                contribution = -0.75
            scores.setdefault(candidate.candidate_id, {})[
                "metacognition:recommended_action"
            ] = contribution
        return scores

    def _current_cognitive_quality(self) -> CognitiveQuality:
        emotion = self.emotion_engine.state
        return self.metacognition.current_quality(
            cognitive_load=min(
                1.0, len(self.working_memory.items) / self.working_memory.item_capacity
            ),
            attention_saturation=min(
                1.0,
                len(self.attention_system.focus.candidate_ids)
                / self.attention_system.capacity,
            ),
            emotion_valence=emotion.valence,
            emotion_arousal=emotion.arousal,
            provenance_refs=(
                f"working-memory:occupancy:{len(self.working_memory.items)}",
                f"attention-focus@{self.attention_system.focus.revision}",
                "emotion:current",
            ),
        )

    def _apply_metacognitive_outcome(self, record: DecisionRecord) -> DecisionRecord:
        if record.metacognition_pre_assessment_id is None:
            return record
        selected = next(
            item.candidate
            for item in record.considered_candidates
            if item.candidate.candidate_id == record.selected_candidate_id
        )
        capability_ids = _string_values(selected.parameters.get("capability_ids"))
        if record.actual_outcome is not None and not record.actual_outcome.success:
            for capability_id in capability_ids:
                current = self.self_model.state.capabilities.get(capability_id)
                self.self_model.update_capability_from_decision(
                    capability_id,
                    capability_id if current is None else current.description,
                    record,
                    tags=_string_values(selected.parameters.get("topic_tags")),
                )
        assessment = self.metacognition.assess_post(
            record,
            self_model_revision=self.self_model.state.revision,
            cognitive_quality=self._current_cognitive_quality(),
        )
        record = self.decision_store.link_post_assessment(
            record.decision_id, assessment.assessment_id
        )
        self._sync_metacognitive_narrative()
        self._persist_self_model_state()
        self._persist_narrative_self_state()
        return record

    def _sync_metacognitive_narrative(self) -> None:
        claim_prefix = "narrative:metacognitive-hypothesis:"
        active_claim_ids = {
            f"narrative:{item.hypothesis_id}"
            for item in self.metacognition.hypotheses.values()
        }
        for claim in tuple(self.narrative_self.claims.values()):
            if (
                claim.claim_id.startswith(claim_prefix)
                and claim.claim_id not in active_claim_ids
                and claim.status != IdentityClaimStatus.REVISED
            ):
                self.narrative_self.revise_claim(
                    claim.claim_id,
                    confidence=min(claim.confidence, 0.49),
                    reason_code="metacognitive_hypothesis_retired",
                    counterevidence_refs=(
                        "metacognition:outcome-withdrawn-or-revised",
                    ),
                    status=IdentityClaimStatus.REVISED,
                )
        for hypothesis in self.metacognition.hypotheses.values():
            claim_id = f"narrative:{hypothesis.hypothesis_id}"
            current = self.narrative_self.claims.get(claim_id)
            if current is None:
                self.narrative_self.propose_claim(
                    kind=IdentityClaimKind.LIMITATION,
                    statement=f"Recurring {hypothesis.hypothesis_code} in {hypothesis.scope_id}",
                    polarity=-1,
                    theme_codes=(hypothesis.scope_id, hypothesis.hypothesis_code),
                    confidence=hypothesis.confidence,
                    stability=0.4,
                    evidence_refs=hypothesis.evidence_refs,
                    related_decision_refs=tuple(
                        ref.split(":", 2)[1] for ref in hypothesis.evidence_refs
                    ),
                    claim_id=claim_id,
                )
            elif set(hypothesis.evidence_refs) - set(current.evidence_refs):
                self.narrative_self.revise_claim(
                    claim_id,
                    confidence=hypothesis.confidence,
                    reason_code="metacognitive_outcome_update",
                    evidence_refs=tuple(
                        ref
                        for ref in hypothesis.evidence_refs
                        if ref not in current.evidence_refs
                    ),
                    status=IdentityClaimStatus.HYPOTHESIS,
                )

    def decision_dataset(self) -> list[DecisionDatasetRecord]:
        return DecisionDatasetGenerator().generate(
            self.decision_store.list_records(DecisionStatus.RESOLVED)
        )

    def _decision_value_evaluator(
        self,
        options: dict[str, dict[str, float]],
        *,
        context_id: str | None = None,
    ) -> dict[str, dict[str, float]]:
        if not options:
            return {}
        return {
            score.option_id: {
                contribution.value_id: contribution.contribution
                for contribution in score.contributions
            }
            for score in self.value_system.evaluate(options, context_id=context_id)
        }

    def restore_self_model_state(self) -> None:
        self.self_model.restore(self.persistent_state.self_model)
        self._sync_self_references()
        self._persist_self_model_state()

    def _persist_self_model_state(self) -> None:
        self.persistent_state.self_model = self.self_model.to_json()

    def update_capability_from_decision(
        self,
        capability_id: str,
        description: str,
        decision_id: str,
        *,
        tags: tuple[str, ...] = (),
    ) -> SelfModelState:
        event = current_agent_event()
        self.self_model.update_capability_from_decision(
            capability_id,
            description,
            self.decision_store.get(decision_id),
            tags=tags,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_self_model_state()
        return self.self_model.state

    def manual_correct_capability(
        self,
        capability_id: str,
        description: str,
        confidence: float,
        *,
        reason: str,
        tags: tuple[str, ...] = (),
    ) -> SelfModelState:
        event = current_agent_event()
        self.self_model.manual_correct_capability(
            capability_id,
            description,
            confidence,
            reason=reason,
            tags=tags,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_self_model_state()
        return self.self_model.state

    def add_self_limitation(
        self, limitation: KnownLimitation, *, reason: str
    ) -> SelfModelState:
        event = current_agent_event()
        self.self_model.add_limitation(
            limitation,
            reason=reason,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_self_model_state()
        return self.self_model.state

    def add_self_uncertainty(
        self, uncertainty: EpistemicUncertainty, *, reason: str
    ) -> SelfModelState:
        event = current_agent_event()
        self.self_model.add_uncertainty(
            uncertainty,
            reason=reason,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_self_model_state()
        return self.self_model.state

    def propose_identity_revision(
        self,
        *,
        proposed_summary: str | None,
        proposed_traits: dict[str, float],
        evidence_refs: tuple[str, ...],
        source: str,
        origin_actor: OriginActor | None = None,
        origin_input_kind: OriginInputKind | None = None,
        proposal_id: str | None = None,
    ) -> IdentityRevisionProposal:
        event = current_agent_event()
        proposal = self.self_model.propose_identity_revision(
            proposed_summary=proposed_summary,
            proposed_traits=proposed_traits,
            evidence_refs=evidence_refs,
            source=source,
            identity_origin=(
                None
                if origin_actor is None
                else new_identity_origin(
                    origin_actor,
                    origin_input_kind or OriginInputKind.SUGGESTION,
                    source_ref="self_model_proposal",
                    event_id=None if event is None else event.event_id,
                    event_sequence=None if event is None else event.processing_sequence,
                )
            ),
            proposal_id=proposal_id,
        )
        self._persist_self_model_state()
        return proposal

    def resolve_identity_revision(
        self, proposal_id: str, *, apply: bool, reason: str
    ) -> IdentityRevisionProposal:
        event = current_agent_event()
        proposal = self.self_model.resolve_identity_revision(
            proposal_id,
            apply=apply,
            reason=reason,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_self_model_state()
        return proposal

    def rollback_self_model(
        self, target_revision: int, *, reason: str
    ) -> SelfModelState:
        event = current_agent_event()
        state = self.self_model.rollback(
            target_revision,
            reason=reason,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_self_model_state()
        return state

    def form_autobiographical_episode(
        self, experience_id: str
    ) -> AutobiographicalEpisode:
        experience = self.experience_store.get(experience_id)
        episode = self.narrative_self.observe_experience(experience)
        if episode is None:
            raise ValueError(
                "Experience does not meet autobiographical importance threshold"
            )
        self.link_experience_result(
            experience_id,
            kind="narrative",
            reference=f"narrative:{episode.episode_id}",
            evidence_refs=(f"experience:{experience_id}",),
        )
        self._sync_self_references()
        self._persist_self_model_state()
        self._persist_narrative_self_state()
        return episode

    def create_narrative_chapter(
        self,
        *,
        title: str,
        theme_codes: tuple[str, ...],
        episode_ids: tuple[str, ...],
        chapter_id: str | None = None,
    ) -> NarrativeChapter:
        chapter = self.narrative_self.create_chapter(
            title=title,
            theme_codes=theme_codes,
            episode_ids=episode_ids,
            chapter_id=chapter_id,
        )
        self._persist_narrative_self_state()
        return chapter

    def propose_narrative_claim(
        self,
        *,
        kind: IdentityClaimKind,
        statement: str,
        polarity: int,
        theme_codes: tuple[str, ...],
        confidence: float,
        stability: float,
        evidence_refs: tuple[str, ...],
        related_experience_ids: tuple[str, ...] = (),
        related_value_refs: tuple[str, ...] = (),
        related_goal_refs: tuple[str, ...] = (),
        related_decision_refs: tuple[str, ...] = (),
        claim_id: str | None = None,
    ) -> IdentityClaim:
        claim = self.narrative_self.propose_claim(
            kind=kind,
            statement=statement,
            polarity=polarity,
            theme_codes=theme_codes,
            confidence=confidence,
            stability=stability,
            evidence_refs=evidence_refs,
            related_experience_ids=related_experience_ids,
            related_value_refs=related_value_refs,
            related_goal_refs=related_goal_refs,
            related_decision_refs=related_decision_refs,
            claim_id=claim_id,
        )
        self._persist_narrative_self_state()
        return claim

    def revise_narrative_claim(
        self,
        claim_id: str,
        *,
        confidence: float,
        reason_code: str,
        evidence_refs: tuple[str, ...] = (),
        counterevidence_refs: tuple[str, ...] = (),
        status: IdentityClaimStatus | None = None,
    ) -> IdentityClaim:
        claim = self.narrative_self.revise_claim(
            claim_id,
            confidence=confidence,
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            counterevidence_refs=counterevidence_refs,
            status=status,
        )
        self._persist_narrative_self_state()
        return claim

    def set_future_self_projection(
        self,
        *,
        description: str,
        theme_codes: tuple[str, ...],
        desired_level: float,
        current_level: float,
        evidence_refs: tuple[str, ...],
        projection_id: str | None = None,
    ) -> FutureSelfProjection:
        projection = self.narrative_self.set_future_self(
            description=description,
            theme_codes=theme_codes,
            desired_level=desired_level,
            current_level=current_level,
            evidence_refs=evidence_refs,
            projection_id=projection_id,
        )
        motivation = self.motivation_dynamics.observe_future_self_gap(
            projection.projection_id,
            gap=projection.gap,
            uncertainty=1.0 - min(1.0, len(evidence_refs) * 0.25),
        )
        if motivation is not None:
            projection = self.narrative_self.link_future_motivation(
                projection.projection_id, motivation.motivation_id
            )
        self._persist_narrative_self_state()
        self._persist_motivation_state()
        return projection

    def create_narrative_continuity_link(
        self,
        earlier_ref: str,
        later_ref: str,
        *,
        relation_code: str,
        evidence_refs: tuple[str, ...],
        confidence: float,
        link_id: str | None = None,
    ) -> ContinuityLink:
        link = self.narrative_self.link_continuity(
            earlier_ref,
            later_ref,
            relation_code=relation_code,
            evidence_refs=evidence_refs,
            confidence=confidence,
            link_id=link_id,
        )
        self._persist_narrative_self_state()
        return link

    def _sync_self_references(self) -> None:
        self.self_model.sync_references(
            commitment_refs=(
                commitment.commitment_id
                for commitment in self.commitment_store.commitments.values()
                if commitment.status != CommitmentStatus.PROPOSED
            ),
            value_revision_refs={
                value.value_id: value.revision
                for value in self.value_system.list_values()
            },
            autobiographical_summary_refs=(
                f"narrative:{episode.episode_id}"
                for episode in self.narrative_self.episodes.values()
            ),
        )

    def _sync_self_model_working_memory(
        self, candidates: list[ActionCandidate]
    ) -> None:
        for item in tuple(self.working_memory.items):
            if item.kind == WorkingMemoryKind.SELF_MODEL:
                self.working_memory.forget(item.item_id)
        for candidate in candidates:
            selection = self.self_model.select_relevant(candidate)
            narrative = self.narrative_self.select_relevant(
                theme_codes=_string_values(candidate.parameters.get("topic_tags"))
                + _string_values(candidate.parameters.get("theme_codes")),
                capability_ids=_string_values(
                    candidate.parameters.get("capability_ids")
                ),
            )
            for index, rendered in enumerate(
                (*selection.rendered_items, *narrative.rendered_items)
            ):
                self.working_memory.admit(
                    working_memory_item(
                        item_id=f"self:{candidate.candidate_id}:{index}",
                        kind=WorkingMemoryKind.SELF_MODEL,
                        content=rendered,
                        activation=0.85,
                        salience=0.8,
                        retention_reason=RetentionReason.RELEVANT_SELF_MODEL,
                        source="runtime.self_model",
                    )
                )

    def _resolve_context(
        self,
        context_id: str | None,
        *,
        source_channel: str,
        source_session_id: str | None,
        interlocutor_key: str | None,
        create_context: bool,
    ) -> ContextFrame:
        identifier = context_id or self.default_context_id
        participants = () if interlocutor_key is None else (interlocutor_key,)
        if identifier is None or (
            create_context and self.context_registry.get(identifier) is None
        ):
            frame = self.context_registry.create(
                context_id=identifier,
                source_channel=source_channel,
                source_session_id=source_session_id,
                participant_ids=participants,
            )
            self.default_context_id = frame.context_id
        else:
            existing_frame = self.context_registry.get(identifier)
            if existing_frame is None:
                raise KeyError(identifier)
            frame = self.context_registry.resume(identifier)
        if (
            interlocutor_key is not None
            and self.context_registry.get_interlocutor(interlocutor_key) is None
        ):
            self.context_registry.register_interlocutor(
                InterlocutorModel(identity_key=interlocutor_key)
            )
        if interlocutor_key is not None:
            self.relationship_store.ensure_interlocutor(interlocutor_key)
        return frame

    def _admit_runtime_context(
        self,
        memory_context: MemoryContext,
        emotion_state: EmotionState,
        event: Any,
        current_context: ContextFrame,
    ) -> None:
        event_id = None if event is None else event.event_id
        event_sequence = None if event is None else event.processing_sequence
        context_id = current_context.context_id
        for record in memory_context.db1_results:
            self.working_memory.admit(
                working_memory_item(
                    item_id=f"episode:{record.id}",
                    kind=WorkingMemoryKind.EPISODIC,
                    reference=f"episode:{record.id}",
                    source_event_id=record.source_event_id,
                    source_event_sequence=record.processing_sequence,
                    context_id=record.context_id,
                    source=record.source,
                    source_channel=record.source_channel,
                    source_session_id=record.source_session_id,
                    activation=0.7,
                    salience=0.6,
                )
            )
        for semantic_record in memory_context.db2_results:
            self.working_memory.admit(
                working_memory_item(
                    item_id=f"semantic:{semantic_record.id}",
                    kind=WorkingMemoryKind.SEMANTIC,
                    reference=f"semantic:{semantic_record.id}",
                    source_event_id=None,
                    source_event_sequence=None,
                    context_id=semantic_record.context_id,
                    source=semantic_record.source,
                    source_channel=semantic_record.source_channel,
                    source_session_id=semantic_record.source_session_id,
                    activation=0.65,
                    salience=0.65,
                )
            )
        self.working_memory.admit(
            working_memory_item(
                item_id="emotion:current",
                kind=WorkingMemoryKind.EMOTION,
                content=(
                    f"Current valence={emotion_state.valence:.3f}, "
                    f"arousal={emotion_state.arousal:.3f}"
                ),
                source_event_id=event_id,
                source_event_sequence=event_sequence,
                context_id=context_id,
                source="runtime.emotion",
                source_channel=current_context.source_channel,
                source_session_id=current_context.source_session_id,
                activation=1.0,
                salience=max(0.5, emotion_state.arousal),
                retention_reason=RetentionReason.CURRENT_EMOTION,
            )
        )
        self._admit_structured_state(
            self.goal_manager.goals_json(),
            kind=WorkingMemoryKind.GOAL,
            reason=RetentionReason.ONGOING_GOAL,
            required_status=GoalStatus.ACTIVE.value,
        )
        self._admit_structured_state(
            self.commitment_store.to_json(),
            kind=WorkingMemoryKind.COMMITMENT,
            reason=RetentionReason.ACTIVE_COMMITMENT,
            required_status={status.value for status in ACCEPTED_COMMITMENT_STATUSES},
        )
        unresolved = self.persistent_state.working_memory_metadata.get(
            "unresolved_items", []
        )
        if isinstance(unresolved, list):
            self._admit_structured_state(
                [entry for entry in unresolved if isinstance(entry, dict)],
                kind=WorkingMemoryKind.UNRESOLVED,
                reason=RetentionReason.UNRESOLVED,
            )

    def _admit_structured_state(
        self,
        entries: list[dict[str, Any]],
        *,
        kind: WorkingMemoryKind,
        reason: RetentionReason,
        required_status: str | set[str] | None = None,
    ) -> None:
        for entry in entries:
            allowed_statuses = (
                {required_status}
                if isinstance(required_status, str)
                else required_status
            )
            if (
                allowed_statuses is not None
                and entry.get("status") not in allowed_statuses
            ):
                continue
            entry_id = entry.get("id", entry.get("goal_id", entry.get("commitment_id")))
            content = entry.get("content", entry.get("description", entry.get("text")))
            if not isinstance(entry_id, str) or not isinstance(content, str):
                continue
            self.working_memory.admit(
                working_memory_item(
                    item_id=f"{kind.value}:{entry_id}",
                    kind=kind,
                    content=content,
                    activation=0.9,
                    salience=0.9,
                    retention_reason=reason,
                )
            )

    def _resolve_working_memory(self, item: WorkingMemoryItem) -> str | None:
        if item.reference is None:
            return item.content
        if item.reference.startswith("episode:"):
            episode = self.memory_system.get_episodic(
                item.reference.removeprefix("episode:")
            )
            if (
                episode is None
                or episode.archived
                or episode.lifecycle_status != MemoryLifecycleStatus.ACTIVE
            ):
                return None
            return (
                "Past recorded interaction (not a current fact): "
                f"User: {episode.user_input} | Assistant: {episode.response}"
            )
        if item.reference.startswith("semantic:"):
            semantic = self.memory_system.get_semantic(
                item.reference.removeprefix("semantic:")
            )
            if semantic is None or not self.memory_system.semantic_is_retrievable(
                semantic
            ):
                return None
            return f"Stored semantic record (not an adopted belief): {semantic.text}"
        return None


def _generate_fallback(provider: ModelProvider, prompt: str) -> str:
    generate_fallback = getattr(provider, "generate_fallback", None)
    if not callable(generate_fallback):
        raise RuntimeError("Model generation failed and no fallback model is available")
    return str(generate_fallback(prompt))


def _string_values(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _unit_target(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


def _fallback_used(provider: ModelProvider) -> bool:
    return bool(getattr(provider, "last_fallback_used", False))
