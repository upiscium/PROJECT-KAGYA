"""Integrated runtime main loop."""

from dataclasses import dataclass
from datetime import UTC, datetime
import time
from typing import Any, Protocol
from uuid import uuid4

from kagya.body import EmotionEngineAllostasis, EmotionState, EmotionUpdate
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
    DecisionDatasetGenerator,
    DecisionDatasetRecord,
    DecisionRecord,
    DecisionStatus,
    DecisionStore,
    parse_candidate_output,
    schema_candidate_prompt,
)
from kagya.identity import (
    EndorsementStatus,
    EpistemicUncertainty,
    IdentityOrigin,
    IdentityRevisionProposal,
    KnownLimitation,
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
from kagya.memory import DualMemorySystem, MemoryContext, MemoryLifecycleStatus
from kagya.memory import ValidationStatus
from kagya.memory.quality import assess_generation_health
from kagya.models import ModelProvider
from kagya.motivation import (
    Commitment,
    CommitmentStatus,
    CommitmentStore,
    Goal,
    GoalDecision,
    GoalDecisionInput,
    GoalManager,
    GoalStatus,
    GoalType,
    MotivationDynamics,
    MotivationEpisode,
    MotivationRecord,
)
from kagya.persona import ConsciousAgent, PromptBuilder, ResponsePostprocessor
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


class KagyaMainLoop:
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
        self.commitment_store = CommitmentStore()
        self.motivation_dynamics = MotivationDynamics()
        self.decision_store = DecisionStore()
        self.self_model = SelfModel()
        self.experience_store = ExperienceStore()
        self.belief_store = BeliefStore()
        self.default_context_id: str | None = None
        self.restore_appraisal_state()
        self.restore_value_state()
        self.restore_motivation_state()
        self.restore_decision_state()
        self.restore_self_model_state()
        self.restore_experience_state()
        self.restore_belief_state()

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
        previous_context_frames = self.context_registry.frames
        previous_interlocutors = self.context_registry.interlocutors
        previous_default_context_id = self.default_context_id
        previous_calibration = self.surprisal_calculator.export_history()
        previous_experience_state = self.experience_store.to_json()
        previous_motivation_state = self.motivation_dynamics.to_json()
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
            appraisal = self.appraiser.assess(
                loss_measurement,
                AppraisalSignals(
                    controllability=self.settings.appraisal.default_controllability
                    if loss_measurement.valid
                    else 0.5,
                    certainty=self.settings.appraisal.default_certainty
                    if loss_measurement.valid
                    else 0.2,
                    social_relevance=0.5 if current_context.participant_ids else 0.0,
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
            )
            prompt = self.prompt_builder.build(
                user_input,
                emotion_state,
                working_memory_view,
                current_context=current_context,
                attachments=attachments or [],
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
            fallback_used = bool(
                getattr(self.provider, "last_fallback_used", False)
            )
            generation_duration = max(
                time.perf_counter() - generation_started, 1e-9
            )
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
                hidden_thought=processed_response.hidden_thought,
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
            )
            experience = self.experience_store.integrate(
                build_chat_experience(
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
                        event_sequence=None
                        if event is None
                        else event.processing_sequence,
                    ),
                    context_id=current_context.context_id,
                    interlocutor_ids=current_context.participant_ids,
                    appraisal=appraisal,
                    valence=emotion_state.valence,
                    arousal=emotion_state.arousal,
                    prediction_error=(
                        loss_measurement.mean_token_loss
                        if loss_measurement.valid
                        else None
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
            )
            self._persist_experience_state()
            self.motivation_dynamics.observe_experience(experience)
            self._persist_motivation_state()
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
            self.motivation_dynamics.restore(previous_motivation_state)
            self._persist_motivation_state()
            raise
        return ChatResult(
            episode_id=episode_id,
            experience_id=experience.experience_id,
            response=processed_response.visible_response,
            hidden_thought=processed_response.hidden_thought
            if debug
            else processed_response.hidden_thought,
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

    def _metric(
        self, method: str, name: str, value: float, **labels: str
    ) -> None:
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
        self.persistent_state.extensions["appraisal_calibration"] = (
            self.surprisal_calculator.export_history()
        )

    def restore_experience_state(self) -> None:
        self.experience_store.restore(
            self.persistent_state.extensions.get("experiences")
        )
        self._persist_experience_state()

    def _persist_experience_state(self) -> None:
        self.persistent_state.extensions["experiences"] = (
            self.experience_store.to_json()
        )

    def get_experience(self, experience_id: str) -> ExperienceRecord:
        return self.experience_store.get(experience_id)

    def list_experiences(self) -> list[ExperienceRecord]:
        return self.experience_store.list_records()

    def restore_belief_state(self) -> None:
        self.belief_store.restore(self.persistent_state.extensions.get("beliefs"))
        self._persist_belief_state()

    def _persist_belief_state(self) -> None:
        self.persistent_state.extensions["beliefs"] = self.belief_store.to_json()

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
        self._persist_experience_state()
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
        self.goal_manager.restore(
            self.persistent_state.active_goals,
            decisions if isinstance(decisions, list) else [],
        )
        self.commitment_store.restore(self.persistent_state.commitments)
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
        self.persistent_state.motivation_extensions["dynamics"] = (
            self.motivation_dynamics.to_json()
        )

    def reevaluate_motivation(self) -> tuple[MotivationEpisode, list[Goal]]:
        event = current_agent_event()
        candidates, held_ids = self.motivation_dynamics.goal_candidates()
        goals: list[Goal] = []
        selected_ids: list[str] = []
        for candidate in candidates:
            goal = self.propose_goal(
                goal_type=GoalType.INTRINSIC,
                description=candidate.description,
                structured_target={
                    "motivation_id": candidate.motivation_id,
                    "target_ref": candidate.target_ref,
                },
                origin_actor=OriginActor.SELF,
                origin_input_kind=OriginInputKind.INTERNAL_STATE,
                origin_source_ref=f"motivation:{candidate.motivation_id}",
                priority=candidate.priority,
                urgency=candidate.urgency,
                expected_utility=candidate.priority,
                confidence=candidate.confidence,
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
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        return episode, goals

    def decay_motivation(self, elapsed_hours: float) -> list[MotivationRecord]:
        records = self.motivation_dynamics.decay(elapsed_hours)
        self._persist_motivation_state()
        return records

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
        needs_information: bool = False,
        goal_id: str | None = None,
    ) -> Goal:
        event = current_agent_event()
        if origin_value_id is not None:
            self.value_system.get(origin_value_id)
        if value_effects:
            self.value_system.evaluate({"goal_proposal": value_effects})
        identity_origin: IdentityOrigin | None = None
        if origin_actor is not None:
            identity_origin = new_identity_origin(
                origin_actor,
                origin_input_kind or OriginInputKind.SUGGESTION,
                source_ref=origin_source_ref,
                event_id=None if event is None else event.event_id,
                event_sequence=None if event is None else event.processing_sequence,
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
                for item in self.commitment_store.list_commitments(
                    CommitmentStatus.ACTIVE
                )
            ),
        )

    def create_commitment(
        self,
        *,
        description: str,
        priority: float = 0.7,
        urgency: float = 0.7,
        expected_utility: float = 0.7,
        confidence: float = 0.8,
        deadline: str | None = None,
        value_effects: dict[str, float] | None = None,
        conflict_ids: tuple[str, ...] = (),
        commitment_id: str | None = None,
        origin_actor: OriginActor | None = None,
        origin_source_ref: str | None = None,
    ) -> Commitment:
        event = current_agent_event()
        identifier = commitment_id or str(uuid4())
        if deadline is not None:
            parsed_deadline = datetime.fromisoformat(deadline)
            if parsed_deadline.tzinfo is None:
                raise ValueError("Commitment deadline must include a timezone")
            if parsed_deadline <= datetime.now(UTC):
                raise ValueError("Commitment deadline has expired")
        goal = self.propose_goal(
            goal_type=GoalType.COMMITMENT,
            description=description,
            origin_actor=origin_actor,
            origin_input_kind=OriginInputKind.REQUEST,
            origin_source_ref=origin_source_ref,
            priority=priority,
            urgency=urgency,
            expected_utility=expected_utility,
            confidence=confidence,
            conflict_ids=conflict_ids,
            deadline=deadline,
            value_effects=value_effects,
            goal_id=f"commitment:{identifier}",
        )
        self.adopt_goal(goal.goal_id)
        adopted_goal = self.goal_manager.get(goal.goal_id)
        if adopted_goal.status != GoalStatus.ACTIVE:
            raise ValueError("Commitment goal was not endorsed for activation")
        commitment = self.commitment_store.create(
            description=description,
            related_goal_id=goal.goal_id,
            origin_event_id=None if event is None else event.event_id,
            identity_origin=adopted_goal.identity_origin,
            deadline=deadline,
            commitment_id=identifier,
        )
        self._sync_self_references()
        self._persist_self_model_state()
        self._sync_motivation_working_memory()
        self._persist_motivation_state()
        return self.commitment_store.get(commitment.commitment_id)

    def transition_commitment(
        self,
        commitment_id: str,
        status: CommitmentStatus,
        *,
        reason: str,
        outcome: str | None = None,
    ) -> Commitment:
        if status == CommitmentStatus.ACTIVE:
            raise ValueError("Commitment is already active")
        event = current_agent_event()
        commitment = self.commitment_store.get(commitment_id)
        goal = self.goal_manager.goals.get(commitment.related_goal_id)
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
                and item.status == CommitmentStatus.ACTIVE
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

    def _sync_motivation_working_memory(self) -> None:
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
            if commitment.status == CommitmentStatus.ACTIVE:
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

    def restore_decision_state(self) -> None:
        payload = self.persistent_state.extensions.get("decision_records", [])
        self.decision_store.restore(payload if isinstance(payload, list) else [])
        self._persist_decision_state()

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
        if context_id is not None and self.context_registry.get(context_id) is None:
            raise ValueError(f"Unknown context: {context_id}")
        completed_goals = {
            goal.goal_id
            for goal in self.goal_manager.goals.values()
            if goal.status == GoalStatus.COMPLETED
        }
        emotion = self.emotion_engine.state
        source_experience = (
            None
            if context_id is None
            else self.experience_store.latest_for_context(context_id)
        )
        active_beliefs = self.belief_store.active(context_id=context_id)
        record = self.decision_store.create(
            candidates,
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
            decision_id=decision_id,
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
        return record

    def generate_decision_candidates(self, situation: str) -> list[ActionCandidate]:
        raw = self.provider.generate(schema_candidate_prompt(situation))
        return parse_candidate_output(raw)

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
        proposals = self.value_system.proposals_from_decision_outcome(record)
        updates = self.value_system.apply(proposals)
        self.value_system.record_reassessment(record, updates)
        self._persist_value_state()
        self._sync_self_references()
        self._persist_self_model_state()
        self._persist_decision_state()
        return record

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

    def _sync_self_references(self) -> None:
        self.self_model.sync_references(
            commitment_refs=(
                commitment.commitment_id
                for commitment in self.commitment_store.commitments.values()
            ),
            value_revision_refs={
                value.value_id: value.revision
                for value in self.value_system.list_values()
            },
        )

    def _sync_self_model_working_memory(
        self, candidates: list[ActionCandidate]
    ) -> None:
        for item in tuple(self.working_memory.items):
            if item.kind == WorkingMemoryKind.SELF_MODEL:
                self.working_memory.forget(item.item_id)
        for candidate in candidates:
            selection = self.self_model.select_relevant(candidate)
            for index, rendered in enumerate(selection.rendered_items):
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
            required_status=CommitmentStatus.ACTIVE.value,
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
        required_status: str | None = None,
    ) -> None:
        for entry in entries:
            if required_status is not None and entry.get("status") != required_status:
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
            episode = self.memory_system.get_episodic(item.reference.removeprefix("episode:"))
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
            semantic = self.memory_system.get_semantic(item.reference.removeprefix("semantic:"))
            if (
                semantic is None
                or semantic.archived
                or semantic.metadata.get("publication_status", "published")
                != "published"
            ):
                return None
            return f"Stored semantic record (not an adopted belief): {semantic.text}"
        return None


def _generate_fallback(provider: ModelProvider, prompt: str) -> str:
    generate_fallback = getattr(provider, "generate_fallback", None)
    if not callable(generate_fallback):
        raise RuntimeError("Model generation failed and no fallback model is available")
    return str(generate_fallback(prompt))


def _fallback_used(provider: ModelProvider) -> bool:
    return bool(getattr(provider, "last_fallback_used", False))
