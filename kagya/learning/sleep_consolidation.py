"""Sleep-time consolidation and learning cycle."""

from dataclasses import dataclass
from uuid import uuid4

from kagya.config import Settings
from kagya.learning.adapter_registry import AdapterEntry, AdapterRegistry
from kagya.learning.dream_dataset_generator import DreamDatasetGenerator
from kagya.learning.qlora_trainer import QloraTrainer, QloraTrainingResult
from kagya.memory import DualMemorySystem, EpisodicMemoryRecord
from kagya.memory import ConsolidationStatus, ValidationStatus
from kagya.models import ModelProvider
from kagya.training.dataset_governance import DatasetGovernanceStore, DatasetSplit


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
        self.dream_dataset_generator = (
            dream_dataset_generator or DreamDatasetGenerator()
        )
        self.qlora_trainer = qlora_trainer or QloraTrainer(settings)

    def run(self) -> SleepCycleResult:
        if not self.settings.sleep.enabled:
            return SleepCycleResult([], [], None, None, None)
        episodes = self.select_high_emotion_episodes()
        if not episodes:
            return SleepCycleResult([], [], None, None, None)
        attempt_id = str(uuid4())
        pipeline_version = "sleep-v2"
        for episode in episodes:
            self.memory_system.set_consolidation_state(
                episode.id,
                status=ConsolidationStatus.IN_PROGRESS,
                pipeline_version=pipeline_version,
                attempt_id=attempt_id,
            )
        dataset_path = (
            self.settings.sleep.dream_dataset_path.parent
            / "runs"
            / attempt_id
            / self.settings.sleep.dream_dataset_path.name
        )
        try:
            semantic_texts = self._generate_semantic_texts(episodes)
            governed = DatasetGovernanceStore(
                self.settings.sleep.training_artifact_directory / "datasets"
            ).create_from_episodes(episodes, source_job_id=attempt_id)
            dataset = governed.split_bytes(DatasetSplit.TRAIN)
            if not dataset:
                raise ValueError("Governed dataset has no eligible training records")
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with dataset_path.open("xb") as output:
                output.write(dataset)
            training_result = self.qlora_trainer.train(
                dataset_path,
                dataset_revision=governed.revision,
                dataset_manifest_hash=governed.manifest_hash,
            )
            semantic_ids = [
                self.memory_system.save_semantic(
                    text,
                    source_episode_ids=[episode.id],
                    metadata={
                        "source": "sleep_cycle",
                        "publication_status": "staged",
                        "attempt_id": attempt_id,
                        "pipeline_version": pipeline_version,
                        "context_id": episode.context_id,
                        "source_channel": episode.source_channel,
                        "source_session_id": episode.source_session_id,
                    },
                )
                for episode, text in zip(episodes, semantic_texts, strict=True)
            ]
            adapter_entry = self.adapter_registry.register_candidate(
                adapter_id=training_result.adapter_id,
                adapter_path=training_result.adapter_path,
                dataset_path=training_result.dataset_path,
                dataset_hash=training_result.dataset_hash,
                base_model=self.settings.model.primary_id,
                base_model_revision=self.settings.model.revision,
                adapter_hash=training_result.adapter_hash,
                notes="registered by sleep cycle dry-run"
                if training_result.dry_run
                else "registered by sleep cycle",
            )
            for semantic_id in semantic_ids:
                self.memory_system.publish_semantic(semantic_id)
            for episode in episodes:
                self.memory_system.set_consolidation_state(
                    episode.id,
                    status=ConsolidationStatus.COMPLETED,
                    pipeline_version=pipeline_version,
                    attempt_id=attempt_id,
                )
        except Exception:
            for episode in episodes:
                self.memory_system.set_consolidation_state(
                    episode.id,
                    status=ConsolidationStatus.FAILED,
                    pipeline_version=pipeline_version,
                    attempt_id=attempt_id,
                )
            raise
        return SleepCycleResult(
            selected_episode_ids=[episode.id for episode in episodes],
            semantic_memory_ids=semantic_ids,
            dream_dataset_path=str(dataset_path),
            training_result=training_result,
            adapter_entry=adapter_entry,
        )

    def select_high_emotion_episodes(self) -> list[EpisodicMemoryRecord]:
        episodes = self.memory_system._get_unarchived_episodic_records()
        arousal_threshold = self.settings.memory.consolidation_min_arousal
        salience_threshold = self.settings.memory.consolidation_min_subjective_salience
        selected = [
            episode
            for episode in episodes
            if episode.emotion_arousal >= arousal_threshold
            or episode.subjective_salience >= salience_threshold
        ]
        selected = [
            episode
            for episode in selected
            if episode.validation_status == ValidationStatus.VERIFIED
            and episode.generation_health.healthy
            and episode.training_included
            and not (
                episode.consolidation_status == ConsolidationStatus.COMPLETED
                and episode.consolidation_version == "sleep-v2"
            )
        ]
        return selected[: self.settings.sleep.max_episodes_per_cycle]

    def _generate_semantic_texts(
        self, episodes: list[EpisodicMemoryRecord]
    ) -> list[str]:
        semantic_texts: list[str] = []
        for episode in episodes:
            semantic_text = self.model_provider.generate(
                "Extract one concise semantic memory from this high-emotion episode.\n"
                f"User: {episode.user_input}\nAssistant: {episode.response}"
            )
            semantic_texts.append(semantic_text)
        return semantic_texts


def _is_high_emotion(episode: EpisodicMemoryRecord, threshold: float) -> bool:
    return (
        episode.emotion_arousal > threshold or abs(episode.emotion_valence) > threshold
    )
