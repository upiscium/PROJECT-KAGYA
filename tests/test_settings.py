from pathlib import Path

from project_kagya.settings import (
    load_settings,
    resolve_log_path,
    resolve_sleep_output_path,
)


def test_load_settings_from_toml(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.toml"
    settings_path.write_text(
        """
[runtime]
backend = "dummy"
input_text = "hello settings"
initial_valence = 0.25
initial_arousal = 0.75

[model]
model_name = "local-model"
adapter_path = ""
load_in_4bit = false

[memory]
top_k = 5

[emotion]
optimal_loss = 3.0
adaptation_rate = 0.1

[sleep]
high_arousal_threshold = 0.8
high_valence_threshold = 0.5
dream_dataset_path = "dreams.jsonl"
""",
        encoding="utf-8",
    )

    settings = load_settings(settings_path)

    assert settings.runtime.input_text == "hello settings"
    assert settings.runtime.initial_arousal == 0.75
    assert settings.model.model_name == "local-model"
    assert settings.memory.top_k == 5
    assert settings.sleep.dream_dataset_path == "dreams.jsonl"
    assert settings.logging.level == "INFO"
    assert settings.paths.adapter_dir == ".kagya/adapters"


def test_resolve_relative_paths_from_settings_file(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.toml"
    settings_path.write_text("[runtime]\n", encoding="utf-8")

    settings = load_settings(settings_path)

    assert resolve_log_path(settings).name == "project-kagya.log"
    assert resolve_sleep_output_path(settings).name == "dream_dataset.jsonl"
