from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import tomllib

from project_kagya.conscious_agent import ConsciousAgent, GeneratedResponse
from project_kagya.chat_backend import GemmaChatBackend
from project_kagya.data_quality_evaluation import DataQualityEvaluator, EvaluatedExample
from project_kagya.drift_control import DriftController
from project_kagya.dual_memory_system import DualMemorySystem
from project_kagya.embodied_emotion import EmbodiedEmotion
from project_kagya.emotion_engine import EmotionState
from project_kagya.multimodal_fastapi_interface import create_app
from project_kagya.qlora_training import QLoRATrainer, QLoRATrainingSummary
from project_kagya.sleep_consolidation import (
    DreamExample,
    EpisodicEntry,
    SemanticCandidate,
    SleepCycleManager,
    SleepConsolidationResult,
)
from project_kagya.sleep_consolidation_training import (
    SleepConsolidationTrainingPipeline,
    TrainingExample,
    TrainingSummary,
)
from project_kagya.thought_quality_assurance import (
    ThoughtExample,
    ThoughtQualityAssurer,
    ThoughtValidationReport,
)


@dataclass(slots=True)
class RuntimeConfig:
    backend: str = "dummy"
    input_text: str = "hello"
    initial_valence: float = 0.0
    initial_arousal: float = 0.0


@dataclass(slots=True)
class ModelConfig:
    model_name: str = "google/gemma-4-E4B"
    adapter_path: str = ""
    load_in_4bit: bool = True


@dataclass(slots=True)
class MemoryConfig:
    top_k: int = 3


@dataclass(slots=True)
class EmotionConfig:
    optimal_loss: float = 2.5
    adaptation_rate: float = 0.15


@dataclass(slots=True)
class SleepConfig:
    high_arousal_threshold: float = 0.7
    high_valence_threshold: float = 0.6
    dream_dataset_path: Path = Path("dream_dataset.jsonl")


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    file_path: str = "project-kagya.log"


@dataclass(slots=True)
class PathsConfig:
    chroma_dir: Path = Path(".kagya/chroma")
    adapter_dir: Path = Path(".kagya/adapters")
    log_dir: Path = Path(".kagya/logs")
    sleep_dir: Path = Path(".kagya/sleep")


@dataclass(slots=True)
class ProjectSettings:
    runtime: RuntimeConfig
    model: ModelConfig
    memory: MemoryConfig
    emotion: EmotionConfig
    sleep: SleepConfig
    logging: LoggingConfig
    paths: PathsConfig


@dataclass(slots=True)
class PipelineResult:
    prompt: str
    response: str
    memory_context: str
    drift_accepted: bool
    thought_validation: ThoughtValidationReport
    data_quality: dict[str, int]
    sleep_summary: SleepConsolidationResult
    training_summary: TrainingSummary
    qlora_summary: QLoRATrainingSummary


def load_settings(settings_path: Path) -> ProjectSettings:
    if not settings_path.exists():
        raise FileNotFoundError(f"settings file not found: {settings_path}")

    raw = tomllib.loads(settings_path.read_text(encoding="utf-8"))
    return ProjectSettings(
        runtime=_runtime_config(raw.get("runtime", {})),
        model=_model_config(raw.get("model", {})),
        memory=_memory_config(raw.get("memory", {})),
        emotion=_emotion_config(raw.get("emotion", {})),
        sleep=_sleep_config(raw.get("sleep", {})),
        logging=_logging_config(raw.get("logging", {})),
        paths=_paths_config(raw.get("paths", {})),
    )


class KagyaRuntime:
    def __init__(self, settings_path: Path) -> None:
        self.settings_path = settings_path
        self.settings = load_settings(settings_path)

    @staticmethod
    def _log(message: str) -> None:
        print(f"[kagya] {message}")

    def run_demo(self) -> int:
        from dataclasses import asdict

        self._log("demo: start")
        embodied = EmbodiedEmotion()
        emotion = EmotionState(
            valence=self.settings.runtime.initial_valence,
            arousal=self.settings.runtime.initial_arousal,
            optimal_loss=self.settings.emotion.optimal_loss,
        )

        embodied.update_body_state({"type": "conversation", "intensity": 2.0})
        embodied.update_body_state({"type": "sleep", "duration": 1.5})
        adjusted = embodied.modulate_emotion(1.5, emotion)

        print("body_state:", asdict(embodied.get_body_state()))
        print("emotion_state:", asdict(adjusted))
        self._log("demo: done")
        return 0

    def run_server(self) -> None:
        self._log("serve: start")
        app = create_app(self._build_chat_backend())
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("uvicorn is required for --serve") from exc

        uvicorn.run(app, host="127.0.0.1", port=8000)

    def run_consolidation(self) -> SleepConsolidationResult:
        self._log("consolidate: start")
        manager = self._build_sleep_manager()
        episodes = self._build_sample_episodes()
        result = manager.run(episodes)
        self._log(
            f"consolidate: done selected={result.selected} confirmed={result.confirmed} "
            f"generated={result.generated} written={result.written_lines}"
        )
        return result

    def run_training(self) -> TrainingSummary:
        self._log("train: start")
        pipeline = SleepConsolidationTrainingPipeline(QLoRATrainer())
        model = _TrainingModel()
        summary = pipeline.train(
            self.settings.sleep.dream_dataset_path,
            model,
            self.settings.paths.adapter_dir,
        )
        self._log(
            f"train: done total={summary.total_lines} valid={summary.valid_examples} "
            f"trained={summary.trained_examples}"
        )
        return summary

    def run_full_pipeline(self) -> PipelineResult:
        self._log("pipeline: start")
        model = _PipelineModel()
        embodied = EmbodiedEmotion()
        emotion = EmotionState(
            valence=self.settings.runtime.initial_valence,
            arousal=self.settings.runtime.initial_arousal,
            optimal_loss=self.settings.emotion.optimal_loss,
        )
        memory = DualMemorySystem(top_k=self.settings.memory.top_k)
        agent = ConsciousAgent(model)
        drift = DriftController()
        thought_assurer = ThoughtQualityAssurer()
        data_evaluator = DataQualityEvaluator()
        sleep_manager = self._build_sleep_manager()

        embodied.update_body_state({"type": "conversation", "intensity": 2.0})
        embodied.update_body_state({"type": "sleep", "duration": 1.5})
        adjusted = embodied.modulate_emotion(1.5, emotion)
        self._log(
            f"pipeline: emotion valence={adjusted.valence:.3f} arousal={adjusted.arousal:.3f}"
        )

        memory_context = memory.retrieve_context(self.settings.runtime.input_text)
        self._log("pipeline: memory retrieved")
        prompt = agent.build_prompt(adjusted.valence, adjusted.arousal, memory_context)
        before = drift.snapshot(model)
        response = agent.generate_response(prompt)
        after = drift.snapshot(model)
        drift_report = drift.measure_drift(before, after)
        drift_accepted = drift.should_accept_update(drift_report)
        self._log(f"pipeline: response generated drift_accepted={drift_accepted}")

        episode_id = memory.save_episodic(
            self.settings.runtime.input_text,
            response.text,
            adjusted.valence,
            adjusted.arousal,
        )
        thought_text = (
            memory_context if memory_context != "[No Relevant Memory]" else prompt
        )
        confidence = 0.9 if drift_accepted else 0.4
        status = "confirmed" if drift_accepted else "tentative"

        thought_example = ThoughtExample(
            input=self.settings.runtime.input_text,
            thought=thought_text,
            output=response.text,
            source_ids=[episode_id],
            confidence=confidence,
            status=status,
        )
        thought_validation = thought_assurer.validate_example(thought_example)
        evaluated_example = EvaluatedExample(**asdict(thought_example))
        data_report = data_evaluator.evaluate([evaluated_example])
        data_quality = data_evaluator.summarize_issues([evaluated_example])
        self._log(
            "pipeline: validation "
            f"thought_valid={thought_validation.valid} valid_examples={data_report.valid_examples}"
        )

        episodes = [
            EpisodicEntry(
                id=episode_id,
                input=self.settings.runtime.input_text,
                response=response.text,
                valence=max(0.75, adjusted.valence),
                arousal=max(0.8, adjusted.arousal),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        ]
        sleep_summary = sleep_manager.run(episodes)
        self._log(
            f"pipeline: sleep written={sleep_summary.written_lines} generated={sleep_summary.generated}"
        )

        training_pipeline = SleepConsolidationTrainingPipeline(QLoRATrainer())
        training_summary = training_pipeline.train(
            self.settings.sleep.dream_dataset_path,
            model,
            self.settings.paths.adapter_dir,
        )
        self._log(
            f"pipeline: train valid={training_summary.valid_examples} trained={training_summary.trained_examples}"
        )
        qlora_summary = QLoRATrainingSummary(
            total_examples=training_summary.valid_examples,
            trained_examples=training_summary.trained_examples,
            adapter_path=training_summary.adapter_path,
        )
        if training_summary.adapter_path is not None:
            model = training_pipeline.load_adapter(model, training_summary.adapter_path)
            if hasattr(model, "load_adapter"):
                model.load_adapter(training_summary.adapter_path)

        self._log("pipeline: done")
        return PipelineResult(
            prompt=prompt,
            response=response.text,
            memory_context=memory_context,
            drift_accepted=drift_accepted,
            thought_validation=thought_validation,
            data_quality=data_quality,
            sleep_summary=sleep_summary,
            training_summary=training_summary,
            qlora_summary=qlora_summary,
        )

    def _build_sleep_manager(self) -> SleepCycleManager:
        return SleepCycleManager(
            extractor=_SleepExtractor(),
            dream_generator=_SleepDreamGenerator(),
            output_path=self.settings.sleep.dream_dataset_path,
            confidence_threshold=0.5,
        )

    def _build_chat_backend(self) -> GemmaChatBackend:
        return GemmaChatBackend(
            model_name=self.settings.model.model_name,
            top_k=self.settings.memory.top_k,
            initial_valence=self.settings.runtime.initial_valence,
            initial_arousal=self.settings.runtime.initial_arousal,
            optimal_loss=self.settings.emotion.optimal_loss,
            load_in_4bit=self.settings.model.load_in_4bit,
            adapter_path=self.settings.model.adapter_path or None,
        )

    def _build_sample_episodes(self) -> list[EpisodicEntry]:
        return [
            EpisodicEntry(
                id="episode-1",
                input=self.settings.runtime.input_text,
                response=f"response for {self.settings.runtime.input_text}",
                valence=0.75,
                arousal=0.8,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        ]


class _PipelineModel:
    def __init__(self) -> None:
        self.state: dict[str, float] = {
            "tone": 0.0,
            "length": 0.0,
            "vocabulary": 0.0,
            "known_sample": 0.0,
            "thought": 0.0,
            "iterations": 0.0,
        }
        self.loaded_adapter: str | None = None

    def generate(self, prompt: str) -> str:
        self.state["tone"] += 0.05
        self.state["length"] += min(len(prompt) / 1000.0, 0.05)
        self.state["vocabulary"] += 0.03
        self.state["known_sample"] += 0.02
        self.state["thought"] += 0.01
        return f"response: {prompt.splitlines()[0]}"

    def snapshot_state(self) -> dict[str, float]:
        return dict(self.state)

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        self.state = {
            key: float(value) if isinstance(value, (int, float)) else 0.0
            for key, value in snapshot.items()
        }

    def train_on_texts(self, prompts: list[str]) -> None:
        self.state["iterations"] += float(len(prompts))

    def save_pretrained(self, path: str) -> None:
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir.joinpath("adapter.json").write_text(
            '{\n  "status": "saved"\n}\n', encoding="utf-8"
        )

    def load_adapter(self, adapter_dir: str) -> "_PipelineModel":
        self.loaded_adapter = adapter_dir
        return self


class _TrainingModel(_PipelineModel):
    pass


class _SleepExtractor:
    def extract(self, episodes: Sequence[EpisodicEntry]) -> Sequence[SemanticCandidate]:
        return [
            SemanticCandidate(
                id=episode.id,
                fact=episode.response,
                source_ids=[episode.id],
                confidence=0.9,
                status="confirmed",
                timestamp=episode.timestamp,
            )
            for episode in episodes
        ]


class _SleepDreamGenerator:
    def generate(self, candidate: SemanticCandidate) -> DreamExample:
        return DreamExample(
            input=candidate.fact,
            thought=f"reason about {candidate.fact}",
            output=f"reply about {candidate.fact}",
            source_ids=list(candidate.source_ids),
            confidence=candidate.confidence,
            status=candidate.status,
        )


def _runtime_config(section: dict[str, Any]) -> RuntimeConfig:
    return RuntimeConfig(
        backend=str(section.get("backend", "dummy")),
        input_text=str(section.get("input_text", "hello")),
        initial_valence=float(section.get("initial_valence", 0.0)),
        initial_arousal=float(section.get("initial_arousal", 0.0)),
    )


def _model_config(section: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        model_name=str(section.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")),
        adapter_path=str(section.get("adapter_path", "")),
        load_in_4bit=bool(section.get("load_in_4bit", True)),
    )


def _memory_config(section: dict[str, Any]) -> MemoryConfig:
    return MemoryConfig(top_k=int(section.get("top_k", 3)))


def _emotion_config(section: dict[str, Any]) -> EmotionConfig:
    return EmotionConfig(
        optimal_loss=float(section.get("optimal_loss", 2.5)),
        adaptation_rate=float(section.get("adaptation_rate", 0.15)),
    )


def _sleep_config(section: dict[str, Any]) -> SleepConfig:
    return SleepConfig(
        high_arousal_threshold=float(section.get("high_arousal_threshold", 0.7)),
        high_valence_threshold=float(section.get("high_valence_threshold", 0.6)),
        dream_dataset_path=Path(
            section.get("dream_dataset_path", "dream_dataset.jsonl")
        ),
    )


def _logging_config(section: dict[str, Any]) -> LoggingConfig:
    return LoggingConfig(
        level=str(section.get("level", "INFO")),
        file_path=str(section.get("file_path", "project-kagya.log")),
    )


def _paths_config(section: dict[str, Any]) -> PathsConfig:
    return PathsConfig(
        chroma_dir=Path(section.get("chroma_dir", ".kagya/chroma")),
        adapter_dir=Path(section.get("adapter_dir", ".kagya/adapters")),
        log_dir=Path(section.get("log_dir", ".kagya/logs")),
        sleep_dir=Path(section.get("sleep_dir", ".kagya/sleep")),
    )
