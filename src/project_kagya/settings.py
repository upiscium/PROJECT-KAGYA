from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


@dataclass(frozen=True)
class RuntimeSettings:
    backend: str = "dummy"
    input_text: str = "hello"
    initial_valence: float = 0.0
    initial_arousal: float = 0.0


@dataclass(frozen=True)
class ModelSettings:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path: str | None = None
    load_in_4bit: bool = True


@dataclass(frozen=True)
class MemorySettings:
    top_k: int = 3


@dataclass(frozen=True)
class EmotionSettings:
    optimal_loss: float = 2.5
    adaptation_rate: float = 0.15


@dataclass(frozen=True)
class SleepSettings:
    high_arousal_threshold: float = 0.7
    high_valence_threshold: float = 0.6
    dream_dataset_path: str = "dream_dataset.jsonl"


@dataclass(frozen=True)
class LoggingSettings:
    level: str = "INFO"
    file_path: str = "project-kagya.log"


@dataclass(frozen=True)
class PathSettings:
    chroma_dir: str = ".kagya/chroma"
    adapter_dir: str = ".kagya/adapters"
    log_dir: str = ".kagya/logs"
    sleep_dir: str = ".kagya/sleep"


@dataclass(frozen=True)
class AppSettings:
    runtime: RuntimeSettings = RuntimeSettings()
    model: ModelSettings = ModelSettings()
    memory: MemorySettings = MemorySettings()
    emotion: EmotionSettings = EmotionSettings()
    sleep: SleepSettings = SleepSettings()
    logging: LoggingSettings = LoggingSettings()
    paths: PathSettings = PathSettings()
    source_path: Path | None = None


class SettingsError(ValueError):
    pass


def load_settings(path: str | Path = "settings.toml") -> AppSettings:
    settings_path = Path(path)
    if not settings_path.exists():
        raise SettingsError(f"settings file not found: {settings_path}")

    with settings_path.open("rb") as handle:
        raw = tomllib.load(handle)

    return AppSettings(
        runtime=_build_runtime_settings(raw.get("runtime", {})),
        model=_build_model_settings(raw.get("model", {})),
        memory=_build_memory_settings(raw.get("memory", {})),
        emotion=_build_emotion_settings(raw.get("emotion", {})),
        sleep=_build_sleep_settings(raw.get("sleep", {})),
        logging=_build_logging_settings(raw.get("logging", {})),
        paths=_build_path_settings(raw.get("paths", {})),
        source_path=settings_path.resolve(),
    )


def _build_runtime_settings(raw: Any) -> RuntimeSettings:
    return RuntimeSettings(
        backend=str(raw.get("backend", "dummy")),
        input_text=str(raw.get("input_text", "hello")),
        initial_valence=float(raw.get("initial_valence", 0.0)),
        initial_arousal=float(raw.get("initial_arousal", 0.0)),
    )


def _build_model_settings(raw: Any) -> ModelSettings:
    adapter_path = raw.get("adapter_path")
    return ModelSettings(
        model_name=str(raw.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")),
        adapter_path=str(adapter_path) if adapter_path else None,
        load_in_4bit=bool(raw.get("load_in_4bit", True)),
    )


def _build_memory_settings(raw: Any) -> MemorySettings:
    return MemorySettings(top_k=int(raw.get("top_k", 3)))


def _build_emotion_settings(raw: Any) -> EmotionSettings:
    return EmotionSettings(
        optimal_loss=float(raw.get("optimal_loss", 2.5)),
        adaptation_rate=float(raw.get("adaptation_rate", 0.15)),
    )


def _build_sleep_settings(raw: Any) -> SleepSettings:
    return SleepSettings(
        high_arousal_threshold=float(raw.get("high_arousal_threshold", 0.7)),
        high_valence_threshold=float(raw.get("high_valence_threshold", 0.6)),
        dream_dataset_path=str(raw.get("dream_dataset_path", "dream_dataset.jsonl")),
    )


def _build_logging_settings(raw: Any) -> LoggingSettings:
    return LoggingSettings(
        level=str(raw.get("level", "INFO")),
        file_path=str(raw.get("file_path", "project-kagya.log")),
    )


def _build_path_settings(raw: Any) -> PathSettings:
    return PathSettings(
        chroma_dir=str(raw.get("chroma_dir", ".kagya/chroma")),
        adapter_dir=str(raw.get("adapter_dir", ".kagya/adapters")),
        log_dir=str(raw.get("log_dir", ".kagya/logs")),
        sleep_dir=str(raw.get("sleep_dir", ".kagya/sleep")),
    )


def resolve_path(settings: AppSettings, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute() or settings.source_path is None:
        return path
    return settings.source_path.parent / path


def resolve_model_adapter_path(settings: AppSettings) -> Path | None:
    adapter_path = settings.model.adapter_path
    if adapter_path:
        return resolve_path(settings, adapter_path)
    return resolve_path(settings, settings.paths.adapter_dir)


def resolve_sleep_output_path(settings: AppSettings) -> Path:
    sleep_path = Path(settings.sleep.dream_dataset_path)
    if sleep_path.is_absolute():
        return sleep_path
    return resolve_path(
        settings,
        Path(settings.paths.sleep_dir) / sleep_path,
    ) or Path(settings.sleep.dream_dataset_path)


def resolve_log_path(settings: AppSettings) -> Path:
    log_path = Path(settings.logging.file_path)
    if log_path.is_absolute():
        return log_path
    return resolve_path(
        settings,
        Path(settings.paths.log_dir) / log_path,
    ) or Path(settings.logging.file_path)
