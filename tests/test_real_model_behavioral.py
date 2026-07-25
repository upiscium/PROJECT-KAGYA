import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from kagya.api.routes.evaluations import _rerun_runtime_evaluation
from kagya.api.schemas.evaluation import BehavioralRerunRequest
from kagya.config import load_settings
from kagya.learning import BehavioralArtifactStore, BehavioralEvaluationState
from kagya.learning.adapter_registry import AdapterRegistry
from kagya.learning.behavioral_evaluation import (
    BehavioralRuntimeKind,
    PairedBehavioralEvaluationResult,
)
from kagya.learning.real_model_runtime_behavioral import (
    run_real_model_runtime_evaluation,
)
from tests.adapter_behavioral_helpers import (
    register_runtime_candidate,
    write_runtime_behavioral_result,
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


def test_real_model_rerun_reserves_adapter_and_records_failure_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = load_settings(Path(__file__).resolve().parents[1] / "config.yaml")
    settings = base.model_copy(
        update={
            "adapter_registry": base.adapter_registry.model_copy(
                update={
                    "path": tmp_path / "registry.json",
                    "eval_result_dir": tmp_path / "evaluations",
                }
            )
        }
    )
    registry = AdapterRegistry(settings)
    register_runtime_candidate(registry, tmp_path, "candidate")
    source_path = write_runtime_behavioral_result(
        registry,
        tmp_path,
        "candidate",
        evaluation_id="real-source",
        runtime_kind=BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    final_source = (
        settings.adapter_registry.eval_result_dir / "behavioral" / "real-source.json"
    )
    final_source.parent.mkdir(parents=True)
    final_source.write_text(json.dumps(source), encoding="utf-8")
    original = PairedBehavioralEvaluationResult.model_validate(source)
    store = BehavioralArtifactStore(settings.adapter_registry.eval_result_dir)

    def rerun(*args, **kwargs):
        rerun_id = args[1]
        result = original.model_copy(update={"evaluation_id": rerun_id})
        artifact = store.prepare(rerun_id, result.model_dump(mode="json"))
        return result, artifact.status.value

    monkeypatch.setattr(
        "kagya.api.routes.evaluations.run_real_model_runtime_evaluation", rerun
    )
    response = _rerun_runtime_evaluation(
        source,
        BehavioralRerunRequest(rerun_id="real-rerun"),
        settings,
        registry,
    )
    assert response.evaluation_id == "real-rerun"

    with store.adapter_lock("candidate", blocking=False):
        with pytest.raises(HTTPException) as busy:
            _rerun_runtime_evaluation(
                source,
                BehavioralRerunRequest(rerun_id="busy-rerun"),
                settings,
                registry,
            )
    assert busy.value.status_code == 409

    def provenance_mismatch(*args, **kwargs):
        rerun_id = args[1]
        assert original.manifest is not None
        changed_manifest = original.manifest.model_copy(
            update={"tool_registry_hash": "e" * 64}
        )
        result = original.model_copy(
            update={"evaluation_id": rerun_id, "manifest": changed_manifest}
        )
        artifact = store.prepare(rerun_id, result.model_dump(mode="json"))
        return result, artifact.status.value

    monkeypatch.setattr(
        "kagya.api.routes.evaluations.run_real_model_runtime_evaluation",
        provenance_mismatch,
    )
    with pytest.raises(HTTPException, match="Immutable rerun manifest differs"):
        _rerun_runtime_evaluation(
            source,
            BehavioralRerunRequest(rerun_id="provenance-mismatch"),
            settings,
            registry,
        )
    mismatch = next(
        item
        for item in store.reconcile()
        if item.evaluation_id == "provenance-mismatch"
    )
    assert mismatch.state == BehavioralEvaluationState.FAILED
    assert not store.prepared_path("provenance-mismatch").exists()

    monkeypatch.setattr(
        "kagya.api.routes.evaluations.run_real_model_runtime_evaluation",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("load failed")),
    )
    with pytest.raises(HTTPException):
        _rerun_runtime_evaluation(
            source,
            BehavioralRerunRequest(rerun_id="failed-rerun"),
            settings,
            registry,
        )
    failed = next(
        item for item in store.reconcile() if item.evaluation_id == "failed-rerun"
    )
    assert failed.state == BehavioralEvaluationState.FAILED
    assert failed.status.value == "orphan_registry_reference"
