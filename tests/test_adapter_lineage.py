import json
from pathlib import Path

import pytest

from kagya.config import load_settings
from kagya.learning import (
    AdapterEvaluator,
    AdapterRegistry,
    AdapterRuntimeManager,
    RuntimeAdapterState,
)
from kagya.models import DummyProvider
from kagya.runtime import AgentEventType, AgentRuntime
from tests.adapter_behavioral_helpers import (
    bind_runtime_behavioral_result,
    register_runtime_candidate,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class _Responses(DummyProvider):
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def test_lineage_requires_known_compatible_parent_hash_and_revision(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    dataset = _dataset(tmp_path / "parent.jsonl", [{"input": "one"}])
    registry.register_candidate(
        adapter_id="parent",
        adapter_path=tmp_path / "parent",
        dataset_path=dataset,
        dataset_hash="dataset-parent",
        base_model_revision="revision-a",
        adapter_hash="hash-parent",
    )

    for parent_id, parent_hash, revision, expected in (
        ("missing", "hash", "revision-a", "Unknown parent"),
        ("parent", "wrong", "revision-a", "hash mismatch"),
        ("parent", "hash-parent", "revision-b", "base revision mismatch"),
    ):
        with pytest.raises(ValueError, match=expected):
            registry.validate_continuation(
                adapter_id="child",
                base_model=registry.settings.model.primary_id,
                base_model_revision=revision,
                parent_adapter_id=parent_id,
                parent_adapter_hash=parent_hash,
            )


def test_lineage_rejects_cycles_and_tracks_dataset_repetition_and_overlap(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    parent_dataset = _dataset(
        tmp_path / "parent.jsonl", [{"input": "same"}, {"input": "old"}]
    )
    registry.register_candidate(
        adapter_id="parent",
        adapter_path=tmp_path / "parent",
        dataset_path=parent_dataset,
        dataset_hash="parent-data",
        base_model_revision="revision-a",
        adapter_hash="hash-parent",
    )
    child_dataset = _dataset(
        tmp_path / "child.jsonl",
        [{"input": "same"}, {"input": "same"}, {"input": "new"}],
    )
    child = registry.register_candidate(
        adapter_id="child",
        adapter_path=tmp_path / "child",
        dataset_path=child_dataset,
        dataset_hash="child-data",
        base_model_revision="revision-a",
        adapter_hash="hash-child",
        parent_adapter_id="parent",
        parent_adapter_hash="hash-parent",
    )

    assert child.dataset_repetition_count == 1
    assert child.dataset_overlap_count == 1
    assert child.dataset_overlap_ratio == pytest.approx(0.5)
    assert [entry.adapter_id for entry in registry.lineage("child")] == [
        "child",
        "parent",
    ]

    payload = json.loads(registry.path.read_text("utf-8"))
    payload["adapters"][0]["parent_adapter_id"] = "child"
    payload["adapters"][0]["parent_adapter_hash"] = "hash-child"
    registry.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Cyclic adapter lineage"):
        registry.lineage("child")


def test_holdout_and_identity_value_behavior_regressions_block_activation(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    _candidate(registry, tmp_path, "candidate")

    entry = registry.apply_evaluation(
        "candidate",
        score=0.95,
        result_path=tmp_path / "fixture-evaluation.json",
        holdout_score=0.7,
        holdout_baseline_score=0.9,
        drift_scores={"identity": -0.2, "value": 0.05, "behavior": 0.05},
        quality_gate_passed=True,
        holdout_gate_passed=False,
        drift_gate_passed=False,
    )

    assert entry.activation_gate_passed is False
    assert entry.holdout_score == 0.7
    assert entry.drift_scores == pytest.approx(
        {"identity": -0.2, "value": 0.05, "behavior": 0.05}
    )
    assert entry.status.value == "trial_active"
    assert entry.holdout_regression is True


def test_lineage_holdout_is_evaluated_against_baseline(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    holdout = _dataset(
        tmp_path / "evaluation_set.jsonl",
        [{"input": "past capability", "output": "retained answer"}],
    )
    dataset = _dataset(tmp_path / "candidate.jsonl", [{"input": "new"}])
    registry.register_candidate(
        adapter_id="candidate",
        adapter_path=tmp_path / "candidate",
        dataset_path=dataset,
        dataset_hash="dataset-candidate",
        base_model_revision="revision-a",
        adapter_hash="hash-candidate",
        evaluation_set_hashes=("holdout-hash",),
        evaluation_dataset_path=holdout,
    )

    result = AdapterEvaluator(registry.settings, registry).evaluate(
        "candidate",
        _Responses("lost answer"),
        baseline_provider=_Responses("retained answer"),
    )

    assert result.holdout_baseline_score == 1.0
    assert result.holdout_score < result.holdout_baseline_score
    assert result.activation_gate_passed is False
    assert registry.lookup("candidate").status.value == "candidate"


def test_canary_failure_limit_accepts_multiple_observations(tmp_path: Path) -> None:
    registry = _registry(tmp_path, canary_failure_limit=2)
    _candidate(registry, tmp_path, "candidate")
    registry.apply_evaluation(
        "candidate", score=0.95, result_path=tmp_path / "eval.json"
    )
    bind_runtime_behavioral_result(registry, tmp_path, "candidate")
    registry.approve("candidate")
    registry.activate("candidate")

    first = registry.record_canary("candidate", success=False)
    second = registry.record_canary("candidate", success=False)

    assert first.rollout_state == "canary"
    assert second.rollout_state == "canary_failed"
    assert second.canary_failures == 2


def test_canary_failure_automatically_rolls_back_without_real_model(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    entry = _candidate(registry, tmp_path, "candidate")
    registry.apply_evaluation(
        "candidate", score=0.95, result_path=tmp_path / "eval.json"
    )
    bind_runtime_behavioral_result(registry, tmp_path, "candidate")
    registry.approve("candidate")
    state = [RuntimeAdapterState(None, None, None, DummyProvider())]

    def switch(provider, selected, sequence):
        state[0] = RuntimeAdapterState(
            None if selected is None else selected.adapter_id,
            None if selected is None else selected.adapter_hash,
            sequence,
            provider,
        )

    manager = AdapterRuntimeManager(
        registry,
        provider_loader=lambda _entry: DummyProvider(),
        runtime_switch=switch,
        runtime_snapshot=lambda: state[0],
        history_path=tmp_path / "activations.json",
    )
    manager.stage(entry.adapter_id)
    manager.verify(entry.adapter_id)
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    runtime.execute(
        AgentEventType.ADAPTER_UPDATE,
        source="test.activate",
        handler=lambda: manager.activate_at_event_boundary(entry.adapter_id),
    )
    rollback = runtime.execute(
        AgentEventType.ADAPTER_UPDATE,
        source="test.canary",
        handler=lambda: manager.report_canary(success=False),
    ).value
    runtime.shutdown()

    assert rollback is not None and rollback.action == "rollback"
    assert state[0].adapter_id is None
    restored = registry.lookup(entry.adapter_id)
    assert restored is not None and restored.rollout_state == "rolled_back"
    assert restored.rollback_target_id is None


def _registry(tmp_path: Path, *, canary_failure_limit: int = 1) -> AdapterRegistry:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "model": settings.model.model_copy(update={"revision": "revision-a"}),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "path": tmp_path / "registry.json",
                    "eval_result_dir": tmp_path / "eval_results",
                    "eval_sets": [],
                    "canary_failure_limit": canary_failure_limit,
                }
            ),
        }
    )
    return AdapterRegistry(settings)


def _candidate(registry: AdapterRegistry, tmp_path: Path, adapter_id: str):
    return register_runtime_candidate(registry, tmp_path, adapter_id)


def _dataset(path: Path, records: list[dict[str, str]]) -> Path:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path
