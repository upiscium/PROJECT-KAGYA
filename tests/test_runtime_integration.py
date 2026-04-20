from __future__ import annotations

from pathlib import Path

from project_kagya.runtime import KagyaRuntime, load_settings


def _write_settings(path: Path, dataset_name: str = "dream_dataset.jsonl") -> None:
    path.write_text(
        "\n".join(
            [
                "[runtime]",
                'backend = "dummy"',
                'input_text = "hello"',
                "initial_valence = 0.0",
                "initial_arousal = 0.0",
                "",
                "[model]",
                'model_name = "Qwen/Qwen2.5-1.5B-Instruct"',
                'adapter_path = ""',
                "load_in_4bit = true",
                "",
                "[memory]",
                "top_k = 3",
                "",
                "[emotion]",
                "optimal_loss = 2.5",
                "adaptation_rate = 0.15",
                "",
                "[sleep]",
                "high_arousal_threshold = 0.7",
                "high_valence_threshold = 0.6",
                f'dream_dataset_path = "{dataset_name}"',
                "",
                "[logging]",
                'level = "INFO"',
                'file_path = "project-kagya.log"',
                "",
                "[paths]",
                'chroma_dir = ".kagya/chroma"',
                'adapter_dir = ".kagya/adapters"',
                'log_dir = ".kagya/logs"',
                'sleep_dir = ".kagya/sleep"',
            ]
        ),
        encoding="utf-8",
    )


def test_load_settings_reads_runtime_config(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.toml"
    _write_settings(settings_path)

    settings = load_settings(settings_path)

    assert settings.runtime.input_text == "hello"
    assert settings.sleep.dream_dataset_path.name == "dream_dataset.jsonl"


def test_full_pipeline_wires_modules_and_persists_artifacts(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.toml"
    dataset_path = tmp_path / "dream_dataset.jsonl"
    adapter_dir = tmp_path / "adapters"
    _write_settings(settings_path, dataset_path.name)

    runtime = KagyaRuntime(settings_path)
    runtime.settings.sleep.dream_dataset_path = dataset_path
    runtime.settings.paths.adapter_dir = adapter_dir

    result = runtime.run_full_pipeline()

    assert result.response
    assert result.prompt
    assert result.drift_accepted is True
    assert result.thought_validation.valid is True
    assert result.sleep_summary.written_lines == 1
    assert result.training_summary.trained_examples == 1
    assert result.training_summary.adapter_path == str(adapter_dir)
    assert dataset_path.exists()
    assert adapter_dir.exists()
    assert (adapter_dir / "adapter.json").exists()
