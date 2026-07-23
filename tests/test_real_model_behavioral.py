import os
from pathlib import Path

import pytest

from kagya.config import load_settings
from kagya.learning.real_model_behavioral import (
    parse_behavior_class,
    run_real_model_behavioral_suite,
)
from kagya.models import load_model_provider


def test_behavior_class_parser_ignores_non_exact_prose() -> None:
    assert parse_behavior_class('Result:\n```json\n{"behavior_class":"defer"}\n```').value == "defer"


@pytest.mark.real_model
def test_real_model_emits_required_structured_behavior_classes() -> None:
    if os.environ.get("KAGYA_RUN_REAL_MODEL_BEHAVIORAL") != "1":
        pytest.skip("set KAGYA_RUN_REAL_MODEL_BEHAVIORAL=1 to load the real model")
    config_path = Path(os.environ.get("KAGYA_CONFIG_PATH", "config.yaml"))
    settings = load_settings(config_path)
    assert settings.model.provider == "transformers", (
        "real-model behavioral integration requires model.provider=transformers"
    )

    result = run_real_model_behavioral_suite(load_model_provider(settings))

    assert set(result.values()) == {"respond", "defer", "no_op"}
