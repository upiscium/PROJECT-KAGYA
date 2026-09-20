"""Integrated runtime main loop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import inspect
from typing import TYPE_CHECKING

from kagya.body import EmotionEngineAllostasis, EmotionState, EmotionUpdate
from kagya.cognition import (
    AppraisalResult,
    AppraisalSignals,
    CognitiveAppraiser,
    LossCalibration,
    LossMeasurement,
    SurprisalCalculator,
    model_key,
)
from kagya.config import Settings
from kagya.memory import DualMemorySystem, MemoryContext, MemoryRecordType
from kagya.models import ModelProvider
from kagya.persona import (
    ConsciousAgent,
    ContextPromptView,
    PromptBuilder,
    ResponsePostprocessor,
)
from kagya.runtime.chat_context import ChatContextSelectors, resolve_chat_context
from kagya.runtime.session_participant import (
    SessionTurnOperation,
    SessionTurnParticipant,
)
from kagya.runtime.session_state import SessionState
from kagya.runtime.context import ContextRegistry
from kagya.runtime.transaction_coordinator import (
    CoordinatedResult,
    TransactionBoundValue,
    TransactionParticipant,
)
from kagya.runtime.working_memory import (
    WorkingMemory,
    WorkingMemorySourceKind,
    WorkingMemoryView,
)

if TYPE_CHECKING:
    from kagya.memory.episodic_participant import MemoryEpisodicParticipant


@dataclass(frozen=True)
class ChatResult:
    """Visible/structured result safe to pass to ordinary callers."""

    episode_id: str
    response: str
    loss: float | None
    valence: float
    arousal: float
    optimal_loss: float
    model_id: str
    adapter_id: str | None


@dataclass(frozen=True)
class DebugChatTrace:
    """Request-scoped diagnostic data that must never become durable authority."""

    hidden_thought: str
    prompt: str
    memory_context: MemoryContext
    working_memory_view: WorkingMemoryView
    diagnostics: ChatDiagnostics


@dataclass(frozen=True)
class ChatDiagnostics:
    """Request-scoped structured cognition diagnostics."""

    measurement: LossMeasurement
    appraisal: AppraisalResult
    temporal_update: EmotionUpdate
    emotion_update: EmotionUpdate


@dataclass(frozen=True)
class _ComputedChat:
    response: str
    loss: float | None
    valence: float
    arousal: float
    optimal_loss: float
    trace: DebugChatTrace | None
    diagnostics: ChatDiagnostics
    memory_participant: MemoryEpisodicParticipant
    session_participant: SessionTurnParticipant


class KagyaMainLoop:
    """Connect prediction error, emotion, memory, generation, and storage."""

    def __init__(
        self,
        settings: Settings,
        provider: ModelProvider,
        memory_system: DualMemorySystem,
        *,
        session_state: SessionState | None = None,
        working_memory: WorkingMemory | None = None,
        emotion_engine: EmotionEngineAllostasis | None = None,
        loss_calibration: LossCalibration | None = None,
        prompt_builder: PromptBuilder | None = None,
        agent: ConsciousAgent | None = None,
        postprocessor: ResponsePostprocessor | None = None,
        adapter_id: str | None = None,
        context_registry: ContextRegistry | None = None,
    ) -> None:
        from kagya.memory.working_memory_resolver import MemoryWorkingMemoryResolver

        self.settings = settings
        self.provider = provider
        self.memory_system = memory_system
        self.working_memory_resolver = MemoryWorkingMemoryResolver(memory_system)
        self.session_state = session_state or SessionState()
        self.working_memory = (
            working_memory
            if working_memory is not None
            else WorkingMemory(
                item_capacity=settings.working_memory.item_capacity,
                projection_max_bytes=settings.working_memory.projection_max_bytes,
            )
        )
        self.surprisal_calculator = SurprisalCalculator(provider)
        self.appraiser = CognitiveAppraiser()
        self.primary_model_key = model_key(
            settings.model.provider, settings.model.primary_id
        )
        approved_keys = tuple(
            sorted(
                {
                    self.primary_model_key,
                    model_key(settings.model.provider, settings.model.fallback_id),
                }
            )
        )
        if loss_calibration is None:
            self.loss_calibration = LossCalibration(
                approved_keys,
                initial_baseline=settings.emotion.baseline_surprisal,
                initial_scale=settings.appraisal.initial_loss_scale,
                minimum_scale=settings.appraisal.minimum_loss_scale,
            )
        else:
            if not isinstance(loss_calibration, LossCalibration):
                raise TypeError("loss_calibration must be LossCalibration")
            if loss_calibration.approved_keys != approved_keys:
                raise ValueError(
                    "loss_calibration approved keys do not match settings"
                )
            self.loss_calibration = loss_calibration
        self.emotion_engine = emotion_engine or EmotionEngineAllostasis(
            EmotionState(optimal_loss=settings.emotion.baseline_surprisal),
            adaptation_rate=settings.emotion.decay_rate,
            appraisal_response_rate=settings.emotion.appraisal_response_rate,
            resting_valence=settings.emotion.resting_valence,
            resting_arousal=settings.emotion.resting_arousal,
            valence_recovery_rate=settings.emotion.valence_recovery_rate,
            arousal_recovery_rate=settings.emotion.arousal_recovery_rate,
        )
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.agent = agent or ConsciousAgent(provider)
        self.postprocessor = postprocessor or ResponsePostprocessor()
        self.adapter_id = adapter_id
        self.context_registry = (
            context_registry if context_registry is not None else ContextRegistry()
        )

    def chat(
        self,
        user_input: str,
        selectors: ChatContextSelectors | None = None,
    ) -> CoordinatedResult[ChatResult]:
        """Compute one live turn inside the serialized AgentRuntime handler."""

        return self._chat_plan(
            user_input,
            capture_debug=False,
            selectors=selectors,
        )

    def _chat_plan(
        self,
        user_input: str,
        *,
        capture_debug: bool,
        selectors: ChatContextSelectors | None,
    ) -> CoordinatedResult[ChatResult]:
        computed = self._run_chat(
            user_input,
            capture_debug=capture_debug,
            selectors=selectors,
            context_registry=self.context_registry,
            working_memory=self.working_memory,
            emotion_engine=self.emotion_engine,
        )
        return CoordinatedResult(
            TransactionBoundValue(
                lambda transaction_id: self._chat_result(computed, transaction_id)
            ),
            self._participants(computed),
        )

    def chat_debug(
        self,
        user_input: str,
        selectors: ChatContextSelectors | None = None,
    ) -> CoordinatedResult[tuple[ChatResult, DebugChatTrace]]:
        """Compute one live debug turn inside the serialized AgentRuntime handler."""

        return self._debug_chat_plan(
            user_input,
            selectors=selectors,
        )

    def emotion_tick(self) -> None:
        """Advance idle emotion state inside the serialized runtime handler."""

        self.emotion_engine.advance_to()
        return None

    def _debug_chat_plan(
        self,
        user_input: str,
        *,
        selectors: ChatContextSelectors | None,
    ) -> CoordinatedResult[tuple[ChatResult, DebugChatTrace]]:
        computed = self._run_chat(
            user_input,
            capture_debug=True,
            selectors=selectors,
            context_registry=self.context_registry,
            working_memory=self.working_memory,
            emotion_engine=self.emotion_engine,
        )
        if computed.trace is None:  # pragma: no cover - internal invariant
            raise RuntimeError("Debug trace was not captured")
        trace = computed.trace
        return CoordinatedResult(
            TransactionBoundValue(
                lambda transaction_id: (
                    self._chat_result(computed, transaction_id),
                    trace,
                )
            ),
            self._participants(computed),
        )

    def _run_chat(
        self,
        user_input: str,
        *,
        capture_debug: bool,
        selectors: ChatContextSelectors | None,
        context_registry: ContextRegistry,
        working_memory: WorkingMemory,
        emotion_engine: EmotionEngineAllostasis,
    ) -> _ComputedChat:
        from kagya.memory.episodic_participant import (
            EpisodicWrite,
            MemoryEpisodicParticipant,
        )

        current_context = resolve_chat_context(context_registry, selectors)
        provenance = (
            current_context.context_id,
            current_context.source_channel,
            current_context.source_session_id,
        )
        context_view = ContextPromptView.from_frame(current_context)
        context_text = self.session_state.context_text()
        temporal_update = emotion_engine.advance_to()
        measurement = self.surprisal_calculator.measure(
            context_text,
            user_input,
            model_key=self.primary_model_key,
            calibration=self.loss_calibration,
        )
        appraisal = self.appraiser.appraise(measurement, AppraisalSignals())
        emotion_update = emotion_engine.update_from_appraisal(
            appraisal,
            primary_loss=measurement.raw_loss if measurement.valid else None,
        )
        emotion_state = emotion_update.state
        memory_context = self.memory_system.retrieve_context(user_input)
        working_memory.advance()
        candidates = [
            (WorkingMemorySourceKind.EPISODIC, record.id, rank)
            for rank, record in enumerate(memory_context.db1_results)
        ] + [
            (WorkingMemorySourceKind.SEMANTIC, record.id, rank)
            for rank, record in enumerate(memory_context.db2_results)
        ]
        # Admit larger rank numbers first (lower retrieval priority). Equal-rank
        # references use ascending source kind and source ID as the total tie-break.
        for source_kind, source_id, rank in sorted(
            candidates,
            key=lambda candidate: (-candidate[2], candidate[0].value, candidate[1]),
        ):
            working_memory.admit(
                source_kind,
                source_id,
                activation=1.0,
                salience=1.0 / (rank + 1),
            )
        working_memory_view = working_memory.select_contextual(
            self.working_memory_resolver,
            context_registry,
            current_context.context_id,
        )
        prompt = self._build_prompt(
            user_input, emotion_state, working_memory_view, context_view
        )
        raw_response = self.agent.generate(prompt)
        processed_response = self.postprocessor.process(raw_response)
        memory_participant = MemoryEpisodicParticipant(
            self.memory_system,
            EpisodicWrite(
                user_input=user_input,
                response=processed_response.visible_response,
                loss=measurement.raw_loss,
                emotion_valence=emotion_state.valence,
                emotion_arousal=emotion_state.arousal,
                record_type=MemoryRecordType.EPISODIC_LOG,
                created_at=datetime.now(UTC).isoformat(),
                context_id=provenance[0],
                source_channel=provenance[1],
                source_session_id=provenance[2],
                schema_version=3,
            ),
        )
        session_participant = SessionTurnParticipant(
            self.session_state,
            SessionTurnOperation(
                user_input=user_input,
                response=processed_response.visible_response,
            ),
        )
        trace = None
        diagnostics = ChatDiagnostics(
            measurement=measurement,
            appraisal=appraisal,
            temporal_update=temporal_update,
            emotion_update=emotion_update,
        )
        if capture_debug:
            trace = DebugChatTrace(
                hidden_thought=processed_response.hidden_thought,
                prompt=prompt,
                memory_context=memory_context,
                working_memory_view=working_memory_view,
                diagnostics=diagnostics,
            )
        return _ComputedChat(
            response=processed_response.visible_response,
            loss=measurement.raw_loss,
            valence=emotion_state.valence,
            arousal=emotion_state.arousal,
            optimal_loss=emotion_state.optimal_loss,
            trace=trace,
            diagnostics=diagnostics,
            memory_participant=memory_participant,
            session_participant=session_participant,
        )

    def _build_prompt(
        self,
        user_input: str,
        emotion_state: EmotionState,
        working_memory_view: WorkingMemoryView,
        context_view: ContextPromptView,
    ) -> str:
        """Pass Context projection while retaining older injected builders."""

        build = self.prompt_builder.build
        parameters = inspect.signature(build).parameters.values()
        accepts_context = any(
            parameter.name == "context_view"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if accepts_context:
            return build(
                user_input,
                emotion_state,
                working_memory_view,
                context_view=context_view,
            )
        return build(user_input, emotion_state, working_memory_view)

    def _chat_result(
        self, computed: _ComputedChat, transaction_id: str
    ) -> ChatResult:
        return ChatResult(
            episode_id=computed.memory_participant.episode_id(transaction_id),
            response=computed.response,
            loss=computed.loss,
            valence=computed.valence,
            arousal=computed.arousal,
            optimal_loss=computed.optimal_loss,
            model_id=self.settings.model.primary_id,
            adapter_id=self.adapter_id,
        )

    @staticmethod
    def _participants(computed: _ComputedChat) -> tuple[TransactionParticipant, ...]:
        return (computed.memory_participant, computed.session_participant)
