from pathlib import Path
import math

import pytest
import yaml
from pydantic import ValidationError

from kagya.api.server import app
from kagya.config import Settings, load_settings
from kagya.config.schema import AppraisalSettings, EmotionSettings


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def read_raw_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_config_yaml_loads_into_typed_settings() -> None:
    settings = load_settings(CONFIG_PATH)

    assert isinstance(settings, Settings)
    assert settings.project.name == read_raw_config()["project"]["name"]


def test_existing_config_uses_appraisal_defaults() -> None:
    settings = load_settings(CONFIG_PATH)

    assert settings.appraisal.initial_loss_scale == 1.0
    assert settings.appraisal.minimum_loss_scale == 0.01


def test_existing_config_uses_emotion_recovery_defaults() -> None:
    settings = load_settings(CONFIG_PATH)

    assert settings.emotion.appraisal_response_rate == 0.4
    assert settings.emotion.resting_valence == 0.0
    assert settings.emotion.resting_arousal == 0.0
    assert settings.emotion.valence_recovery_rate == 0.01
    assert settings.emotion.arousal_recovery_rate == 0.02


def _emotion_settings_payload(**overrides: float) -> dict[str, float]:
    payload = {
        "baseline_surprisal": 1.0,
        "high_emotion_threshold": 0.8,
        "decay_rate": 0.05,
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("appraisal_response_rate", -0.1),
        ("appraisal_response_rate", 1.1),
        ("resting_valence", -1.1),
        ("resting_valence", 1.1),
        ("resting_arousal", -0.1),
        ("resting_arousal", 1.1),
        ("valence_recovery_rate", -0.1),
        ("arousal_recovery_rate", -0.1),
    ],
)
def test_emotion_recovery_fields_enforce_bounds(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        EmotionSettings.model_validate(_emotion_settings_payload(**{field: value}))


@pytest.mark.parametrize(
    "field",
    [
        "baseline_surprisal",
        "high_emotion_threshold",
        "decay_rate",
        "appraisal_response_rate",
        "resting_valence",
        "resting_arousal",
        "valence_recovery_rate",
        "arousal_recovery_rate",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_all_emotion_fields_must_be_finite(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        EmotionSettings.model_validate(_emotion_settings_payload(**{field: value}))


def test_emotion_rejects_timer_configuration_before_u5() -> None:
    with pytest.raises(ValidationError):
        EmotionSettings.model_validate(
            _emotion_settings_payload(timer_enabled=True)  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field", ["initial_loss_scale", "minimum_loss_scale"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_appraisal_scales_must_be_positive(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        AppraisalSettings.model_validate({field: value})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["initial_loss_scale", "minimum_loss_scale"])
def test_appraisal_scales_must_be_finite(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        AppraisalSettings.model_validate({field: value})


def test_appraisal_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        AppraisalSettings.model_validate(
            {"initial_loss_scale": 1.0, "minimum_loss_scale": 0.01, "timer": 1.0}
        )


def test_model_ids_come_from_config() -> None:
    raw_config = read_raw_config()
    settings = load_settings(CONFIG_PATH)

    assert settings.model.primary_id == raw_config["model"]["primary_id"]
    assert settings.model.fallback_id == raw_config["model"]["fallback_id"]


def test_api_settings_come_from_config() -> None:
    raw_config = read_raw_config()
    settings = load_settings(CONFIG_PATH)

    assert settings.api.host == raw_config["api"]["host"]
    assert settings.api.port == raw_config["api"]["port"]
    assert settings.api.admin_token_env == raw_config["api"]["admin_token_env"]
    assert settings.api.cors_origins == raw_config["api"]["cors_origins"]


def test_fastapi_app_is_importable() -> None:
    assert app.title == load_settings(CONFIG_PATH).project.name
