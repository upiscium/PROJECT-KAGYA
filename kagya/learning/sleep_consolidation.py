"""Sleep-time consolidation and learning cycle."""

from dataclasses import dataclass, replace
from threading import RLock

from kagya.config import Settings
from kagya.learning.adapter_registry import AdapterEntry, AdapterRegistry
from kagya.learning.dream_dataset_generator import DreamDatasetGenerator
from kagya.learning.qlora_trainer import QloraTrainer, QloraTrainingResult
from kagya.memory import (
    DualMemorySystem,
    EpisodicMemoryRecord,
    MemorySemanticParticipant,
    SemanticBatchEntry,
    SemanticBatchOperation,
    SemanticCreateIntent,
    SemanticStore,
    semantic_id_for_batch_entry,
)
from kagya.models import ModelProvider
from kagya.persona import ResponsePostprocessor
from kagya.runtime import (
    AgentEvent,
    AgentEventType,
    AgentEventSource,
    AgentRuntime,
    CoordinatedResult,
    TransactionBoundValue,
    TransactionCoordinator,
    TransactionKind,
)
from kagya.memory.semantic_lifecycle import (
    SemanticLifecycle,
    SemanticRevision,
    SemanticRevisionOperation,
    SemanticRevisionReason,
    SemanticSourceEdge,
    SemanticSourceKind,
    SemanticSourceStatus,
    semantic_content_digest,
)


@dataclass(frozen=True)
class SleepCycleResult:
    selected_episode_ids: list[str]
    semantic_memory_ids: list[str]
    dream_dataset_path: str | None
    training_result: QloraTrainingResult | None
    adapter_entry: AdapterEntry | None


class SleepCycleManager:
    """Run sleep-time consolidation and adapter candidate registration."""

    def __init__(
        self,
        settings: Settings,
        memory_system: DualMemorySystem,
        model_provider: ModelProvider,
        adapter_registry: AdapterRegistry,
        *,
        dream_dataset_generator: DreamDatasetGenerator | None = None,
        qlora_trainer: QloraTrainer | None = None,
        postprocessor: ResponsePostprocessor | None = None,
        semantic_store: SemanticStore | None = None,
    ) -> None:
        self.settings = settings
        self.memory_system = memory_system
        self.model_provider = model_provider
        self.adapter_registry = adapter_registry
        self.dream_dataset_generator = dream_dataset_generator or DreamDatasetGenerator()
        self.qlora_trainer = qlora_trainer or QloraTrainer(settings)
        self.postprocessor = postprocessor or ResponsePostprocessor()
        self.semantic_store = semantic_store or SemanticStore.from_memory_root(
            memory_system.settings.memory.persist_directory
        )
        self._runtime: AgentRuntime | None = None
        self._learning_lock = RLock()

    def bind_runtime(self, runtime: AgentRuntime) -> None:
        if self._runtime is not None and self._runtime is not runtime:
            raise RuntimeError("SleepCycleManager is already bound to another runtime")
        self._runtime = runtime

    def run(self) -> SleepCycleResult | CoordinatedResult[SleepCycleResult]:
        if not self.settings.sleep.enabled:
            return SleepCycleResult([], [], None, None, None)
        episodes = self.select_high_emotion_episodes()
        if not episodes:
            return SleepCycleResult([], [], None, None, None)
        event = self._validated_sleep_event()
        if len(episodes) > 128:
            raise ValueError("Semantic batch exceeds its bounded entry limit")
        semantic_plan = self._generate_semantic_memories(episodes, event)
        result = SleepCycleResult(
            selected_episode_ids=[episode.id for episode in episodes],
            semantic_memory_ids=[
                semantic_id_for_batch_entry(
                    semantic_plan.operation.transaction_id, entry.batch_index
                )
                for entry in semantic_plan.operation.entries
            ],
            dream_dataset_path=None,
            training_result=None,
            adapter_entry=None,
        )
        return CoordinatedResult(
            TransactionBoundValue(lambda _transaction_id: result),
            (semantic_plan,),
        )

    def complete_post_commit(self, result: SleepCycleResult) -> SleepCycleResult:
        """Run non-transactional learning only after Semantic commit succeeds.

        Semantic publication is the durable transaction authority.  Dataset
        generation, training, and candidate registration are deliberately a
        serialized follow-up workflow because their files are not rollbackable
        transaction participants.
        """

        if not result.selected_episode_ids:
            return result
        if result.training_result is not None or result.adapter_entry is not None:
            return result
        with self._learning_lock:
            episodes: list[EpisodicMemoryRecord] = []
            for episode_id in result.selected_episode_ids:
                committed = self.memory_system.get_committed_episodic(episode_id)
                if committed is None:
                    raise RuntimeError("Selected Sleep episode is no longer authoritative")
                episodes.append(committed.record)
            self.dream_dataset_generator.generate(
                episodes, self.settings.sleep.dream_dataset_path
            )
            training_result = self.qlora_trainer.train(
                self.settings.sleep.dream_dataset_path
            )
            adapter_entry = self.adapter_registry.register_candidate(
                adapter_id=training_result.adapter_id,
                adapter_path=training_result.adapter_path,
                dataset_path=training_result.dataset_path,
                dataset_hash=training_result.dataset_hash,
                base_model=self.settings.model.primary_id,
                notes=(
                    "registered by sleep cycle dry-run"
                    if training_result.dry_run
                    else "registered by sleep cycle"
                ),
            )
            return replace(
                result,
                dream_dataset_path=str(self.settings.sleep.dream_dataset_path),
                training_result=training_result,
                adapter_entry=adapter_entry,
            )

    def select_high_emotion_episodes(self) -> list[EpisodicMemoryRecord]:
        episodes = self.memory_system._get_unarchived_episodic_records()
        selected = [episode for episode in episodes if _is_high_emotion(episode)]
        return selected[: self.settings.sleep.max_episodes_per_cycle]

    def _generate_semantic_memories(
        self, episodes: list[EpisodicMemoryRecord], event: AgentEvent
    ) -> MemorySemanticParticipant:
        if event.processing_sequence is None:
            raise RuntimeError("Sleep event has no processing sequence")
        transaction_id = TransactionCoordinator.derive_transaction_id(
            event, TransactionKind.EVENT_MUTATION
        )
        entries: list[SemanticBatchEntry] = []
        for batch_index, episode in enumerate(episodes):
            committed = self.memory_system.get_committed_episodic(episode.id)
            if committed is None or committed.record != episode:
                raise RuntimeError("Selected Sleep episode is no longer authoritative")
            semantic_text = self.model_provider.generate(
                "Extract one concise semantic memory from this high-emotion episode.\n"
                f"User: {episode.user_input}\nAssistant: {episode.response}"
            )
            processed = self.postprocessor.process(semantic_text)
            revision = SemanticRevision(
                semantic_id=semantic_id_for_batch_entry(transaction_id, batch_index),
                revision=0,
                semantic_content=processed.visible_response,
                content_digest=semantic_content_digest(processed.visible_response),
                created_at=event.requested_at,
                lifecycle=SemanticLifecycle.ACTIVE,
                source_edges=(
                    SemanticSourceEdge(
                        source_kind=SemanticSourceKind.EPISODIC,
                        source_id=episode.id,
                        captured_context_id=episode.context_id,
                        source_status=SemanticSourceStatus.AVAILABLE,
                    ),
                ),
                operation=SemanticRevisionOperation.CREATE,
                reason=SemanticRevisionReason.CREATION,
                event_id=event.event_id,
                event_sequence=event.processing_sequence,
            )
            entries.append(
                SemanticBatchEntry(batch_index, SemanticCreateIntent(revision))
            )
        operation = SemanticBatchOperation(transaction_id, tuple(entries))
        return MemorySemanticParticipant(self.memory_system, self.semantic_store, operation)

    def _validated_sleep_event(self) -> AgentEvent:
        if self._runtime is None:
            raise RuntimeError("Sleep requires the bound AgentRuntime")
        event = self._runtime.current_event()
        if event is None:
            raise RuntimeError("Sleep requires the active runtime event")
        if (
            event.event_type is not AgentEventType.SLEEP
            or event.source is not AgentEventSource.API_SLEEP_RUN
            or event.processing_sequence is None
            or event.processing_sequence <= 0
        ):
            raise RuntimeError("Sleep method does not match the active runtime event")
        return event


def _is_high_emotion(episode: EpisodicMemoryRecord) -> bool:
    return episode.emotion_arousal > 0.7 or abs(episode.emotion_valence) > 0.6
