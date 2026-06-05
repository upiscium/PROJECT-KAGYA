"""Sleep-time consolidation and learning cycle."""

from dataclasses import dataclass

from kagya.config import Settings
from kagya.learning.adapter_registry import AdapterEntry, AdapterRegistry
from kagya.learning.dream_dataset_generator import DreamDatasetGenerator
from kagya.learning.qlora_trainer import QloraTrainer, QloraTrainingResult
from kagya.memory import DualMemorySystem, EpisodicMemoryRecord
from kagya.models import ModelProvider


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
    ) -> None:
        self.settings = settings
        self.memory_system = memory_system
        self.model_provider = model_provider
        self.adapter_registry = adapter_registry
        self.dream_dataset_generator = dream_dataset_generator or DreamDatasetGenerator()
        self.qlora_trainer = qlora_trainer or QloraTrainer(settings)

    def run(self) -> SleepCycleResult:
        if not self.settings.sleep.enabled:
            return SleepCycleResult([], [], None, None, None)
        episodes = self.select_high_emotion_episodes()
        if not episodes:
            return SleepCycleResult([], [], None, None, None)
        semantic_ids = self._generate_semantic_memories(episodes)
        self.dream_dataset_generator.generate(episodes, self.settings.sleep.dream_dataset_path)
        training_result = self.qlora_trainer.train(self.settings.sleep.dream_dataset_path)
        adapter_entry = self.adapter_registry.register_candidate(
            adapter_id=training_result.adapter_id,
            adapter_path=training_result.adapter_path,
            dataset_path=training_result.dataset_path,
            dataset_hash=training_result.dataset_hash,
            base_model=self.settings.model.primary_id,
            notes="registered by sleep cycle dry-run" if training_result.dry_run else "registered by sleep cycle",
        )
        return SleepCycleResult(
            selected_episode_ids=[episode.id for episode in episodes],
            semantic_memory_ids=semantic_ids,
            dream_dataset_path=str(self.settings.sleep.dream_dataset_path),
            training_result=training_result,
            adapter_entry=adapter_entry,
        )

    def select_high_emotion_episodes(self) -> list[EpisodicMemoryRecord]:
        episodes = self.memory_system._get_unarchived_episodic_records()
        threshold = self.settings.sleep.min_emotion_score
        selected = [episode for episode in episodes if _is_high_emotion(episode, threshold)]
        return selected[: self.settings.sleep.max_episodes_per_cycle]

    def _generate_semantic_memories(self, episodes: list[EpisodicMemoryRecord]) -> list[str]:
        semantic_ids: list[str] = []
        for episode in episodes:
            semantic_text = self.model_provider.generate(
                "Extract one concise semantic memory from this high-emotion episode.\n"
                f"User: {episode.user_input}\nAssistant: {episode.response}"
            )
            semantic_ids.append(
                self.memory_system.save_semantic(
                    semantic_text,
                    source_episode_ids=[episode.id],
                    metadata={"source": "sleep_cycle"},
                )
            )
        return semantic_ids


def _is_high_emotion(episode: EpisodicMemoryRecord, threshold: float) -> bool:
    return episode.emotion_arousal > threshold or abs(episode.emotion_valence) > threshold
