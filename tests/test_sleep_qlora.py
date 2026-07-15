import json
from pathlib import Path
import pytest

from kagya.config import Settings, load_settings
from kagya.learning import (
    AdapterRegistry,
    AdapterStatus,
    DreamDatasetGenerator,
    DreamDatasetRecord,
    QloraTrainer,
    SleepCycleManager,
    format_training_text,
)
from kagya.learning.qlora_requirements import check_qlora_production_readiness
from kagya.memory import ConsolidationStatus, DeterministicEmbeddingFunction, DualMemorySystem
from kagya.models import DummyProvider


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_high_emotion_episode_selection_follows_threshold_rules(tmp_path: Path) -> None:
    settings = _settings_for_sleep(tmp_path)
    memory = _memory(settings)
    threshold = settings.sleep.min_emotion_score
    high_arousal = memory.save_episodic("high arousal", "out", emotion_arousal=threshold + 0.01)
    high_valence = memory.save_episodic("high valence", "out", emotion_valence=-(threshold + 0.01))
    memory.save_episodic("low emotion", "out", emotion_arousal=threshold, emotion_valence=threshold)
    manager = SleepCycleManager(settings, memory, DummyProvider(), AdapterRegistry(settings))

    selected = manager.select_high_emotion_episodes()

    assert {episode.id for episode in selected} == {high_arousal, high_valence}


def test_high_emotion_episode_selection_uses_configured_sleep_threshold(tmp_path: Path) -> None:
    base_settings = _settings_for_sleep(tmp_path)
    settings = base_settings.model_copy(
        update={"sleep": base_settings.sleep.model_copy(update={"min_emotion_score": 0.4})}
    )
    memory = _memory(settings)
    selected_by_config = memory.save_episodic("configured threshold", "out", emotion_arousal=0.41)
    memory.save_episodic("below threshold", "out", emotion_valence=-0.4)
    manager = SleepCycleManager(settings, memory, DummyProvider(), AdapterRegistry(settings))

    selected = manager.select_high_emotion_episodes()

    assert [episode.id for episode in selected] == [selected_by_config]


def test_dream_dataset_jsonl_is_generated_with_expected_fields(tmp_path: Path) -> None:
    settings = _settings_for_sleep(tmp_path)
    memory = _memory(settings)
    episode_id = memory.save_episodic(
        "dream input",
        "dream output",
        hidden_thought="dream thought",
        emotion_arousal=0.9,
    )
    episode = memory._get_unarchived_episodic_records()[0]
    assert episode.id == episode_id

    records = DreamDatasetGenerator().generate([episode], settings.sleep.dream_dataset_path)

    lines = settings.sleep.dream_dataset_path.read_text(encoding="utf-8").splitlines()
    assert records == [
        DreamDatasetRecord(
            "dream input", "", "dream output", source_id=episode_id
        )
    ]
    assert json.loads(lines[0]) == {
        "schema_version": 2,
        "source_kind": "verified_episode",
        "source_id": episode_id,
        "validation_status": "verified",
        "input": "dream input",
        "thought": "",
        "output": "dream output",
    }


def test_dataset_records_include_think_only_in_training_format() -> None:
    record = DreamDatasetRecord("input", "internal", "output")

    raw_record = record.to_json()
    training_text = format_training_text(record)

    assert "<think>" not in json.dumps(raw_record)
    assert "<think>" in training_text
    assert "</think>" in training_text
    assert training_text.endswith("output<eos>")


def test_committed_dream_dataset_is_not_overwritten(tmp_path: Path) -> None:
    settings = _settings_for_sleep(tmp_path)
    memory = _memory(settings)
    memory.save_episodic("input", "output")
    episode = memory._get_unarchived_episodic_records()[0]
    generator = DreamDatasetGenerator()

    generator.generate([episode], settings.sleep.dream_dataset_path)

    with pytest.raises(FileExistsError):
        generator.generate([episode], settings.sleep.dream_dataset_path)


def test_qlora_dry_run_returns_adapter_candidate_result(tmp_path: Path) -> None:
    settings = _settings_for_sleep(tmp_path)
    dataset_path = settings.sleep.dream_dataset_path
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps({"input": "i", "thought": "t", "output": "o"}) + "\n",
        encoding="utf-8",
    )

    result = QloraTrainer(settings).train(dataset_path)

    assert result.dry_run is True
    assert result.adapter_id.startswith("adapter-")
    assert result.adapter_path.exists()
    manifest = json.loads((result.adapter_path / "dry_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["qlora"]["alpha"] == settings.qlora.alpha
    assert manifest["qlora"]["dropout"] == settings.qlora.dropout
    assert manifest["qlora"]["max_steps"] == settings.qlora.max_steps
    assert result.training_records == 1


def test_qlora_non_dry_run_trains_and_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    settings = _settings_for_sleep(tmp_path)
    settings = settings.model_copy(update={"qlora": settings.qlora.model_copy(update={"dry_run": False})})
    dataset_path = settings.sleep.dream_dataset_path
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps({"input": "i", "thought": "t", "output": "o"}) + "\n",
        encoding="utf-8",
    )
    calls: dict[str, object] = {}

    class FakeProcessor:
        chat_template = "gemma-template"
        tokenizer = object()

        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> object:
            calls["processor_model_id"] = model_id
            calls["processor_kwargs"] = kwargs
            return FakeProcessor()

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> object:
            calls["model_id"] = model_id
            calls["model_kwargs"] = kwargs
            class Model:
                def named_modules(self):
                    return [(f"model.{name}", object()) for name in settings.qlora.target_modules]

            return Model()

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeDataset:
        @staticmethod
        def from_list(items: list[dict[str, str]]) -> list[dict[str, str]]:
            calls["dataset"] = items
            return items

    class FakeLoraConfig:
        def __init__(self, **kwargs: object) -> None:
            calls["peft_config"] = kwargs

    class FakeSFTConfig:
        def __init__(self, **kwargs: object) -> None:
            calls["training_args"] = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            calls["trainer_kwargs"] = kwargs

        def train(self, **kwargs: object) -> None:
            calls["trained"] = True
            calls["train_kwargs"] = kwargs

        def save_model(self, path: str) -> None:
            calls["saved_path"] = path

    monkeypatch.setattr(
        QloraTrainer,
        "_load_training_dependencies",
        lambda self: {
            "AutoModelForImageTextToText": FakeModelLoader,
            "AutoProcessor": FakeProcessor,
            "BitsAndBytesConfig": FakeBitsAndBytesConfig,
            "Dataset": FakeDataset,
            "LoraConfig": FakeLoraConfig,
            "SFTConfig": FakeSFTConfig,
            "SFTTrainer": FakeTrainer,
            "prepare_model_for_kbit_training": lambda model, **kwargs: model,
        },
    )

    result = QloraTrainer(settings).train(dataset_path)

    manifest = json.loads((result.adapter_path / "training_manifest.json").read_text(encoding="utf-8"))
    assert result.dry_run is False
    assert result.training_records == 1
    assert calls["processor_model_id"] == settings.model.primary_id
    assert calls["processor_kwargs"] == {"revision": settings.model.processor_revision}
    assert calls["model_id"] == settings.model.primary_id
    assert calls["trained"] is True
    assert Path(calls["saved_path"]).name == f".{result.adapter_id}.tmp"
    assert calls["dataset"] == [{"text": format_training_text(DreamDatasetRecord("i", "t", "o"))}]
    assert calls["peft_config"] == {
        "r": settings.qlora.r,
        "lora_alpha": settings.qlora.lora_alpha,
        "lora_dropout": settings.qlora.lora_dropout,
        "target_modules": settings.qlora.target_modules,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    assert calls["training_args"] == {
        "output_dir": str(Path(calls["saved_path"])),
        "learning_rate": settings.qlora.learning_rate,
        "num_train_epochs": settings.qlora.num_train_epochs,
        "max_steps": settings.qlora.max_steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": settings.qlora.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "max_length": settings.qlora.max_sequence_length,
        "optim": "paged_adamw_8bit",
        "bf16": True,
        "seed": settings.qlora.seed,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
    }
    assert calls["train_kwargs"] == {"resume_from_checkpoint": False}
    assert "quantization_config" in calls["model_kwargs"]
    assert calls["model_kwargs"]["revision"] == settings.model.revision
    assert manifest["dry_run"] is False
    assert manifest["dataset_hash"] == result.dataset_hash
    assert manifest["qlora"]["max_steps"] == settings.qlora.max_steps


def test_qlora_prod_check_rejects_default_dry_run_config(tmp_path: Path) -> None:
    settings = _settings_for_sleep(tmp_path)

    report = check_qlora_production_readiness(
        settings,
        dependency_available=lambda name: True,
        cuda_available=lambda: True,
    )

    assert report.ready is False
    assert "qlora.dry_run must be false for production training" in report.failures
    assert "model.provider must be transformers" in report.failures


def test_qlora_prod_check_accepts_ready_boundaries(tmp_path: Path) -> None:
    settings = _settings_for_sleep(tmp_path)
    eval_set = tmp_path / "eval_set.json"
    eval_set.write_text('{"cases": [{"prompt": "p", "expected": "e"}]}', encoding="utf-8")
    settings = settings.model_copy(
        update={
            "model": settings.model.model_copy(
                update={
                    "provider": "transformers",
                    "revision": "model-commit",
                    "processor_revision": "processor-commit",
                }
            ),
            "qlora": settings.qlora.model_copy(update={"dry_run": False}),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"eval_sets": [eval_set]}
            ),
        }
    )

    report = check_qlora_production_readiness(
        settings,
        dependency_available=lambda name: True,
        cuda_available=lambda: True,
    )

    assert report.ready is True
    assert report.failures == []


def test_qlora_prod_check_reports_missing_dependency_and_cuda(tmp_path: Path) -> None:
    settings = _settings_for_sleep(tmp_path)
    eval_set = tmp_path / "eval_set.json"
    eval_set.write_text('{"cases": [{"prompt": "p", "expected": "e"}]}', encoding="utf-8")
    settings = settings.model_copy(
        update={
            "model": settings.model.model_copy(
                update={
                    "provider": "transformers",
                    "revision": "model-commit",
                    "processor_revision": "processor-commit",
                }
            ),
            "qlora": settings.qlora.model_copy(update={"dry_run": False}),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"eval_sets": [eval_set]}
            ),
        }
    )

    report = check_qlora_production_readiness(
        settings,
        dependency_available=lambda name: name != "trl",
        cuda_available=lambda: False,
    )

    assert report.ready is False
    assert "missing training dependencies: trl" in report.failures
    assert "CUDA must be available for the production QLoRA path" in report.failures


def test_sleep_cycle_registers_candidate_and_never_active(tmp_path: Path) -> None:
    settings = _settings_for_sleep(tmp_path)
    memory = _memory(settings)
    registry = AdapterRegistry(settings)
    memory.save_episodic(
        "sleep input",
        "sleep output",
        hidden_thought="sleep thought",
        emotion_arousal=0.9,
    )
    manager = SleepCycleManager(settings, memory, DummyProvider(), registry)

    result = manager.run()

    assert len(result.selected_episode_ids) == 1
    assert len(result.semantic_memory_ids) == 1
    assert result.training_result is not None
    assert result.adapter_entry is not None
    assert result.adapter_entry.status == AdapterStatus.CANDIDATE
    assert all(entry.status != AdapterStatus.ACTIVE for entry in registry.list())
    assert result.dream_dataset_path is not None
    assert Path(result.dream_dataset_path).exists()
    assert "/runs/" in result.dream_dataset_path
    assert memory.retrieve_context("DummyProvider deterministic response.").db2_results

    repeated = manager.run()
    assert repeated.selected_episode_ids == []
    assert repeated.semantic_memory_ids == []


def test_sleep_failure_keeps_staged_semantics_hidden_and_episode_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings_for_sleep(tmp_path)
    memory = _memory(settings)
    registry = AdapterRegistry(settings)
    episode_id = memory.save_episodic(
        "sleep failure", "output", emotion_arousal=0.9
    )
    monkeypatch.setattr(
        registry,
        "register_candidate",
        lambda **kwargs: (_ for _ in ()).throw(OSError("registry unavailable")),
    )
    manager = SleepCycleManager(settings, memory, DummyProvider(), registry)

    with pytest.raises(OSError, match="registry unavailable"):
        manager.run()

    record = memory.get_episodic(episode_id)
    assert record is not None
    assert record.consolidation_status == ConsolidationStatus.FAILED
    assert memory.retrieve_context("sleep failure").db2_results == []


def _settings_for_sleep(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "hippocampus_sleep_test",
                    "db2_collection": "cortex_sleep_test",
                }
            ),
            "sleep": settings.sleep.model_copy(
                update={
                    "dream_dataset_path": tmp_path / "dreams" / "dream_dataset.jsonl",
                    "job_registry_path": tmp_path / "training_jobs.json",
                    "training_artifact_directory": tmp_path / "training_artifacts",
                }
            ),
            "qlora": settings.qlora.model_copy(
                update={"output_dir": tmp_path / "adapters", "dry_run": True}
            ),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "path": tmp_path / "adapter_registry.json",
                    "eval_result_dir": tmp_path / "eval_results",
                    "eval_sets": [],
                }
            ),
        }
    )


def _memory(settings: Settings) -> DualMemorySystem:
    return DualMemorySystem(settings, embedding_function=DeterministicEmbeddingFunction())
