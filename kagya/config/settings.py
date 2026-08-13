"""Configuration loading utilities."""

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]  # PyYAML has no bundled stubs in the frozen baseline.

from kagya.config.schema import Settings


CONFIG_PATH_ENV = "KAGYA_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_settings(path: str | Path | None = None) -> Settings:
    """Load typed settings from a YAML configuration file."""

    if path is None:
        configured_path = os.getenv(CONFIG_PATH_ENV)
        config_path = (
            DEFAULT_CONFIG_PATH
            if configured_path is None
            else Path(configured_path)
        )
    else:
        config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config: dict[str, Any] = yaml.safe_load(config_file) or {}
    return Settings.model_validate(raw_config)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return load_settings()
