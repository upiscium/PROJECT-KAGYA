from pathlib import Path
import math

import pytest
import yaml
from pydantic import ValidationError

from kagya.api.server import app
from kagya.config import Settings, load_settings
from kagya.config.schema import (
    AppraisalSettings,
    EmotionSettings,
    ValueConflictSettings,
    ValueSeedSettings,
    ValueSystemSettings,
)
from kagya.identity.value_system import (
    ValueScope,
    recompute_seed_contract_digest,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def read_raw_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_config_yaml_loads_into_typed_settings() -> None:
    settings = load_settings(CONFIG_PATH)

    assert isinstance(settings, Settings)
    assert settings.project.name == read_raw_config()["project"]["name"]


def test_baseline_values_are_loaded() -> None:
    values = load_settings(CONFIG_PATH).values

    assert [(seed.value_id, seed.name) for seed in values.seeds] == [
        ("care", "care"),
        ("honesty", "honesty"),
    ]
    assert values.seeds[0].scope == "subject"
    assert values.seeds[0].polarity == 1
    assert (values.seeds[0].strength, values.seeds[0].confidence) == (0.8, 0.8)
    assert (values.seeds[0].stability, values.seeds[0].allowed_update_rate) == (
        0.8,
        0.05,
    )
    assert (values.seeds[1].stability, values.seeds[1].allowed_update_rate) == (0.85, 0.04)
    assert values.seeds[0].protectedness == 0.0
    assert values.seeds[0].negotiability == 1.0
    assert values.conflicts[0].conflict_id == "compassionate-honesty"


def test_baseline_seeds_convert_to_exact_declarations_without_admission() -> None:
    seeds = load_settings(CONFIG_PATH).values.seeds

    declarations = [seed.to_declaration() for seed in seeds]

    assert [declaration.value_id for declaration in declarations] == ["care", "honesty"]
    assert all(declaration.concept is None for declaration in declarations)
    assert all(declaration.scope is ValueScope.SUBJECT for declaration in declarations)
    assert all(declaration.context_ids == () for declaration in declarations)
    assert all(len(recompute_seed_contract_digest(declaration)) == 64 for declaration in declarations)


def test_values_are_optional_for_pre_r11_config() -> None:
    raw = read_raw_config()
    raw.pop("values")

    settings = Settings.model_validate(raw)

    assert settings.values.seeds == []
    assert settings.values.conflicts == []


def _seed_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "value_id": "care",
        "name": "care",
        "scope": "subject",
        "polarity": 1,
        "strength": 0.8,
        "confidence": 0.8,
        "stability": 0.8,
        "allowed_update_rate": 0.05,
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("polarity", 0),
        ("strength", -0.1),
        ("strength", 1.1),
        ("allowed_update_rate", 0.0),
        ("allowed_update_rate", 1.1),
        ("strength", math.nan),
        ("confidence", math.inf),
        ("stability", -math.inf),
    ],
)
def test_value_seed_bounds_and_finiteness(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ValueSeedSettings.model_validate(_seed_payload(**{field: value}))


@pytest.mark.parametrize("value", [1.0, True, "1"])
def test_value_polarity_requires_exact_integer(value: object) -> None:
    with pytest.raises(ValidationError):
        ValueSeedSettings.model_validate(_seed_payload(polarity=value))


def test_value_models_reject_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        ValueSeedSettings.model_validate(_seed_payload(weight=0.5))
    with pytest.raises(ValidationError):
        ValueConflictSettings.model_validate(
            {
                "conflict_id": "a-b",
                "name": "a-b",
                "left_value_id": "a",
                "right_value_id": "b",
                "reason": "not configuration authority",
            }
        )


def test_value_seed_ids_must_be_unique() -> None:
    seed = _seed_payload()
    with pytest.raises(ValidationError):
        ValueSystemSettings.model_validate({"seeds": [seed, seed]})


def _conflict(left: str = "care", right: str = "honesty") -> dict[str, str]:
    return {
        "conflict_id": "care-honesty",
        "name": "care-honesty",
        "left_value_id": left,
        "right_value_id": right,
    }


def test_value_conflicts_reject_duplicate_reversed_self_and_unknown_pairs() -> None:
    seeds = [_seed_payload(), _seed_payload(value_id="honesty", name="honesty")]
    with pytest.raises(ValidationError):
        ValueSystemSettings.model_validate(
            {"seeds": seeds, "conflicts": [_conflict(), _conflict()]}
        )
    with pytest.raises(ValidationError):
        ValueConflictSettings.model_validate(_conflict("honesty", "care"))
    with pytest.raises(ValidationError):
        ValueConflictSettings.model_validate(_conflict("care", "care"))
    with pytest.raises(ValidationError):
        ValueSystemSettings.model_validate(
            {"seeds": seeds, "conflicts": [_conflict("care", "other")]}
        )


def test_value_seed_and_conflict_lists_have_maximum_sizes() -> None:
    seed = _seed_payload()
    with pytest.raises(ValidationError):
        ValueSystemSettings.model_validate({"seeds": [seed] * 129})
    conflicts = [_conflict(f"care{i}", f"honesty{i}") for i in range(257)]
    with pytest.raises(ValidationError):
        ValueSystemSettings.model_validate({"conflicts": conflicts})


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


def _emotion_settings_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
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
        "timer_interval_seconds",
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


def test_emotion_timer_defaults_are_disabled_and_minute_long() -> None:
    settings = EmotionSettings.model_validate(_emotion_settings_payload())

    assert settings.timer_enabled is False
    assert settings.timer_interval_seconds == 60.0


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_emotion_timer_interval_must_be_finite_and_positive(value: float) -> None:
    with pytest.raises(ValidationError):
        EmotionSettings.model_validate(
            _emotion_settings_payload(timer_interval_seconds=value)
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
