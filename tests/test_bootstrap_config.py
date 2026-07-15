from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kagya.api.server import app
from kagya.config import Settings, load_settings
from kagya.config.check import main as config_check_main
from kagya.config.compatibility import (
    COMPATIBILITY_FIELDS,
    compatibility_report,
    documented_compatibility_fields,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def read_raw_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_config_yaml_loads_into_typed_settings() -> None:
    settings = load_settings(CONFIG_PATH)

    assert isinstance(settings, Settings)
    assert settings.project.name == read_raw_config()["project"]["name"]


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


def test_working_memory_capacities_come_from_config() -> None:
    raw_config = read_raw_config()
    settings = load_settings(CONFIG_PATH)

    assert settings.working_memory.item_capacity == raw_config["working_memory"]["item_capacity"]
    assert settings.working_memory.token_capacity == raw_config["working_memory"]["token_capacity"]


def test_working_memory_capacities_must_be_positive() -> None:
    settings = load_settings(CONFIG_PATH)

    with pytest.raises(ValidationError):
        settings.working_memory.__class__(
            item_capacity=0,
            token_capacity=1,
        )


def test_reserved_config_fields_are_documented() -> None:
    readme = (CONFIG_PATH.parent / "README.md").read_text(encoding="utf-8")

    for field in (
        "model.fallback_id",
        "memory.embedding_model_id",
        "adapter_registry.allowed_states",
        "adapter_registry.manual_approval_required",
        "qlora.alpha",
        "qlora.dropout",
        "qlora.max_steps",
    ):
        assert field in readme


def test_compatibility_fields_are_documented() -> None:
    documented = documented_compatibility_fields(CONFIG_PATH.parent / "README.md")

    assert documented == {field.field for field in COMPATIBILITY_FIELDS}


def test_config_compatibility_report_notes_legacy_alias_divergence() -> None:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "qlora": settings.qlora.model_copy(
                update={"alpha": settings.qlora.lora_alpha + 1, "dropout": 0.01}
            ),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"manual_approval_required": False}
            ),
        }
    )

    notes = compatibility_report(settings)

    assert (
        "qlora.alpha differs from qlora.lora_alpha; lora_alpha is training-facing"
        in notes
    )
    assert (
        "qlora.dropout differs from qlora.lora_dropout; lora_dropout is training-facing"
        in notes
    )
    assert (
        "adapter_registry.manual_approval_required=false is reserved; manual approval is still enforced"
        in notes
    )


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    raw_config = read_raw_config()
    raw_config["unknown_section"] = {"enabled": True}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    try:
        load_settings(config_path)
    except ValueError as exc:
        assert "unknown_section" in str(exc)
    else:
        raise AssertionError("unknown config keys should be rejected")


def test_config_check_command_accepts_default_config(capsys) -> None:
    exit_code = config_check_main([str(CONFIG_PATH)])

    assert exit_code == 0
    assert "Config OK" in capsys.readouterr().out


def test_fastapi_app_is_importable() -> None:
    assert app.title == load_settings(CONFIG_PATH).project.name
