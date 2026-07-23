from pathlib import Path
from copy import deepcopy

import pytest
import yaml
from pydantic import ValidationError

from kagya.api.server import app, create_app
from kagya.config import (
    ProjectEnvironment,
    Settings,
    load_settings,
    validate_deployment_hostname,
)
from kagya.config.check import main as config_check_main
from kagya.config.compatibility import (
    COMPATIBILITY_FIELDS,
    compatibility_report,
    documented_compatibility_fields,
)
from kagya.config.settings import load_settings_with_notes
from kagya.config.schema import RemoteWorkerSettings


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def read_raw_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_config_yaml_loads_into_typed_settings() -> None:
    settings = load_settings(CONFIG_PATH)

    assert isinstance(settings, Settings)
    assert settings.project.name == read_raw_config()["project"]["name"]
    assert settings.adapter_registry.behavioral_activation_policy.value == (
        "deterministic_runtime_only"
    )


def test_production_requires_real_model_behavioral_policy() -> None:
    raw = deepcopy(read_raw_config())
    raw["project"]["environment"] = "production"

    with pytest.raises(ValidationError, match="production requires"):
        Settings.model_validate(raw)


def test_unknown_environment_fails_closed() -> None:
    raw = deepcopy(read_raw_config())
    raw["project"]["environment"] = "staging"

    with pytest.raises(ValidationError, match="environment"):
        Settings.model_validate(raw)


def test_disabled_behavioral_policy_is_test_only() -> None:
    raw = deepcopy(read_raw_config())
    raw["adapter_registry"]["behavioral_activation_policy"] = "disabled"

    with pytest.raises(ValidationError, match="limited to test"):
        Settings.model_validate(raw)


def test_startup_revalidates_copied_production_policy() -> None:
    settings = load_settings(CONFIG_PATH)
    invalid = settings.model_copy(
        update={
            "project": settings.project.model_copy(
                update={"environment": ProjectEnvironment.PRODUCTION}
            )
        }
    )

    with pytest.raises(ValidationError, match="production requires"):
        create_app(invalid)


def test_model_ids_come_from_config() -> None:
    raw_config = read_raw_config()
    settings = load_settings(CONFIG_PATH)

    assert settings.model.primary_id == raw_config["model"]["primary_id"]
    assert settings.model.fallback_id == raw_config["model"]["fallback_id"]
    assert settings.model.primary_id == "google/gemma-4-12B-it"
    assert settings.model.revision == raw_config["model"]["revision"]


def test_agent_state_and_journal_paths_must_be_distinct() -> None:
    raw = deepcopy(read_raw_config())
    raw["agent_state"] = {"path": ".kagya/shared-state"}
    raw["agent_journal"] = {
        "path": ".kagya/shared-state",
        "max_bytes": 1024,
        "retained_files": 2,
    }

    with pytest.raises(ValidationError, match="must be distinct"):
        Settings.model_validate(raw)


def test_legacy_config_migrates_explicitly_to_standalone(tmp_path: Path) -> None:
    raw = read_raw_config()
    del raw["deployment"]
    del raw["model"]["revision"]
    del raw["model"]["processor_revision"]
    del raw["model"]["fallback_revision"]
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    settings, notes = load_settings_with_notes(path)

    assert settings.deployment.mode.value == "standalone"
    assert settings.deployment.node.role.value == "all"
    assert settings.deployment.training.backend.value == "local"
    assert any("standalone/all/local" in note for note in notes)
    assert any("model.revision" in note for note in notes)


@pytest.mark.parametrize(
    ("legacy_value", "expected"),
    [(True, "real_model_required"), (False, "deterministic_runtime_only")],
)
def test_legacy_behavioral_gate_config_migrates_explicitly(
    tmp_path: Path, legacy_value: bool, expected: str
) -> None:
    raw = read_raw_config()
    del raw["adapter_registry"]["behavioral_activation_policy"]
    raw["adapter_registry"]["require_real_model_behavioral_gate"] = legacy_value
    path = tmp_path / "legacy-policy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    settings, notes = load_settings_with_notes(path)

    assert settings.adapter_registry.behavioral_activation_policy.value == expected
    assert any("require_real_model_behavioral_gate" in note for note in notes)


def test_valid_split_inference_topology_requires_matching_exact_model() -> None:
    raw = deepcopy(read_raw_config())
    raw["model"]["revision"] = "model-commit-123"
    raw["model"]["processor_revision"] = "processor-commit-123"
    raw["deployment"] = {
        "mode": "split",
        "node": {"id": "inference-01", "role": "inference"},
        "training": {
            "backend": "ssh",
            "remote_worker": {
                "node_id": "worker-01",
                "host": "10.0.0.22",
                "port": 22,
                "user": "kagya-worker",
                "identity_file": "/etc/kagya/id_worker",
                "known_hosts_file": "/etc/kagya/known_hosts",
                "remote_inbox": "/var/lib/kagya/inbox",
                "remote_results": "/var/lib/kagya/results",
                "command": "/opt/kagya/bin/kagya-worker",
                "expected_worker_model": {
                    "model_id": raw["model"]["primary_id"],
                    "revision": raw["model"]["revision"],
                    "processor_revision": raw["model"]["processor_revision"],
                },
            },
        },
    }

    settings = Settings.model_validate(raw)

    assert settings.deployment.node.role.value == "inference"
    assert settings.deployment.training.remote_worker is not None
    assert settings.deployment.training.remote_worker.worker_token_env is None


def test_valid_split_training_worker_topology() -> None:
    raw = deepcopy(read_raw_config())
    raw["model"]["revision"] = "model-commit-123"
    raw["model"]["processor_revision"] = "processor-commit-123"
    raw["deployment"] = {
        "mode": "split",
        "node": {"id": "training-01", "role": "training_worker"},
        "training": {
            "backend": "worker",
            "worker": {
                "inbox_directory": "/var/lib/kagya/inbox",
                "work_directory": "/var/lib/kagya/work",
                "result_directory": "/var/lib/kagya/results",
                "max_concurrent_jobs": 1,
                "retain_failed_jobs": True,
                "allowed_submitters": ["inference-01"],
                "worker_token_env": "KAGYA_WORKER_TOKEN",
            },
        },
    }

    settings = Settings.model_validate(raw)

    assert settings.deployment.training.worker is not None
    assert settings.deployment.training.worker.allowed_submitters == ["inference-01"]


@pytest.mark.parametrize(
    ("mode", "role", "backend"),
    [
        ("standalone", "inference", "local"),
        ("split", "all", "ssh"),
        ("split", "training_worker", "local"),
    ],
)
def test_invalid_deployment_combinations_are_rejected(
    mode: str, role: str, backend: str
) -> None:
    raw = deepcopy(read_raw_config())
    raw["deployment"] = {
        "mode": mode,
        "node": {"id": "node-01", "role": role},
        "training": {"backend": backend},
    }

    with pytest.raises(ValidationError, match="invalid deployment"):
        Settings.model_validate(raw)


def test_split_inference_rejects_model_revision_mismatch() -> None:
    raw = deepcopy(read_raw_config())
    raw["model"]["revision"] = "model-commit-123"
    raw["model"]["processor_revision"] = "processor-commit-123"
    raw["deployment"] = {
        "mode": "split",
        "node": {"id": "inference-01", "role": "inference"},
        "training": {
            "backend": "ssh",
            "remote_worker": {
                "node_id": "worker-01",
                "host": "worker.local",
                "user": "worker",
                "identity_file": "/keys/id",
                "known_hosts_file": "/keys/known_hosts",
                "remote_inbox": "/inbox",
                "remote_results": "/results",
                "command": "/bin/kagya-worker",
                "expected_worker_model": {
                    "model_id": raw["model"]["primary_id"],
                    "revision": "wrong-revision",
                    "processor_revision": raw["model"]["processor_revision"],
                },
            },
        },
    }

    with pytest.raises(ValidationError, match="must match inference model"):
        Settings.model_validate(raw)


def test_deployment_schema_rejects_plaintext_worker_secret() -> None:
    fields = RemoteWorkerSettings.model_fields

    assert "password" not in fields
    assert "worker_token" not in fields
    assert "identity_file" in fields
    assert "known_hosts_file" in fields
    assert "worker_token_env" in fields


def test_hostname_enforcement_requires_expected_hostname() -> None:
    raw = deepcopy(read_raw_config())
    raw["deployment"]["node"]["enforce_hostname_match"] = True

    with pytest.raises(ValidationError, match="expected_hostname"):
        Settings.model_validate(raw)


def test_hostname_validation_can_be_enforced_or_disabled() -> None:
    settings = load_settings(CONFIG_PATH)
    enforced = settings.model_copy(
        update={
            "deployment": settings.deployment.model_copy(
                update={
                    "node": settings.deployment.node.model_copy(
                        update={
                            "expected_hostname": "expected-host",
                            "enforce_hostname_match": True,
                        }
                    )
                }
            )
        }
    )

    validate_deployment_hostname(enforced, actual_hostname="expected-host")
    with pytest.raises(RuntimeError, match="startup host"):
        validate_deployment_hostname(enforced, actual_hostname="other-host")
    validate_deployment_hostname(settings, actual_hostname="other-host")


def test_api_settings_come_from_config() -> None:
    raw_config = read_raw_config()
    settings = load_settings(CONFIG_PATH)

    assert settings.api.host == raw_config["api"]["host"]
    assert settings.api.port == raw_config["api"]["port"]
    assert settings.api.admin_token_env == raw_config["api"]["admin_token_env"]
    assert settings.api.admin_auth.enabled is False
    assert settings.api.admin_auth.session_cookie_name == "kagya_admin_session"
    assert settings.api.cors_origins == raw_config["api"]["cors_origins"]


def test_working_memory_capacities_come_from_config() -> None:
    raw_config = read_raw_config()
    settings = load_settings(CONFIG_PATH)

    assert (
        settings.working_memory.item_capacity
        == raw_config["working_memory"]["item_capacity"]
    )
    assert (
        settings.working_memory.token_capacity
        == raw_config["working_memory"]["token_capacity"]
    )


def test_working_memory_capacities_must_be_positive() -> None:
    settings = load_settings(CONFIG_PATH)

    with pytest.raises(ValidationError):
        settings.working_memory.__class__(
            item_capacity=0,
            token_capacity=1,
        )


def test_appraisal_and_recovery_settings_load_from_config() -> None:
    raw_config = read_raw_config()
    settings = load_settings(CONFIG_PATH)

    assert (
        settings.appraisal.timer_interval_seconds
        == raw_config["appraisal"]["timer_interval_seconds"]
    )
    assert (
        settings.emotion.appraisal_response_rate
        == raw_config["emotion"]["appraisal_response_rate"]
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
