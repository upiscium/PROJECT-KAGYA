"""Configuration loading utilities."""

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import yaml

from kagya.config.schema import Settings


CONFIG_PATH_ENV = "KAGYA_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_settings(path: str | Path | None = None) -> Settings:
    """Load typed settings from a YAML configuration file."""

    settings, _ = load_settings_with_notes(path)
    return settings


def load_settings_with_notes(
    path: str | Path | None = None,
) -> tuple[Settings, list[str]]:
    """Load settings and return explicit compatibility migration notes."""

    config_path = Path(path or os.getenv(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH))
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config: dict[str, Any] = yaml.safe_load(config_file) or {}
    migrated, notes = migrate_config(raw_config)
    return Settings.model_validate(migrated), notes


def migrate_config(raw_config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Materialize legacy defaults without weakening the strict schema."""

    migrated = dict(raw_config)
    notes: list[str] = []
    model = dict(migrated.get("model", {}))
    if "revision" not in model:
        model["revision"] = "main"
        notes.append("model.revision migrated to main; pin an exact revision for split mode")
    if "processor_revision" not in model:
        model["processor_revision"] = model["revision"]
        notes.append("model.processor_revision migrated from model.revision")
    if "fallback_revision" not in model:
        model["fallback_revision"] = "main"
        notes.append("model.fallback_revision migrated to main")
    migrated["model"] = model
    if "deployment" not in migrated:
        migrated["deployment"] = {
            "mode": "standalone",
            "node": {
                "id": "kagya-standalone-legacy",
                "role": "all",
                "expected_hostname": None,
                "enforce_hostname_match": False,
            },
            "training": {"backend": "local"},
        }
        notes.append(
            "deployment section migrated explicitly to standalone/all/local"
        )
    return migrated, notes


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return load_settings()
