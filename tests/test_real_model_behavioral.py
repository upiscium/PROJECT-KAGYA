import os
from pathlib import Path

import pytest

from kagya.config import load_settings
from kagya.learning.adapter_registry import AdapterRegistry
from kagya.learning.real_model_runtime_behavioral import (
    run_real_model_runtime_evaluation,
)


@pytest.mark.real_model
def test_real_model_runtime_loads_registered_candidate_adapter() -> None:
    if os.environ.get("KAGYA_RUN_REAL_MODEL_BEHAVIORAL") != "1":
        pytest.skip("set KAGYA_RUN_REAL_MODEL_BEHAVIORAL=1 to load the real model")
    adapter_id = os.environ.get("KAGYA_REAL_MODEL_ADAPTER_ID")
    if not adapter_id:
        pytest.skip("set KAGYA_REAL_MODEL_ADAPTER_ID to a registered candidate")
    settings = load_settings(Path(os.environ.get("KAGYA_CONFIG_PATH", "config.yaml")))
    assert settings.model.provider == "transformers"
    entry = AdapterRegistry(settings).lookup(adapter_id)
    assert entry is not None and entry.adapter_hash and entry.base_model_revision

    result, _ = run_real_model_runtime_evaluation(
        settings,
        "pytest-real-model-runtime",
        baseline_id="base-model",
        candidate_id=entry.adapter_id,
        candidate_adapter_path=Path(entry.path),
        candidate_adapter_hash=entry.adapter_hash,
        base_model_revision=entry.base_model_revision,
    )

    assert result.runtime_kind.value == "real_model_runtime"
    assert result.manifest is not None
    assert result.manifest.candidate_adapter_path_hash == entry.adapter_hash
