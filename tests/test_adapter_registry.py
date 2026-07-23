from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import pytest

from kagya.config import Settings, load_settings
from kagya.learning import AdapterRegistry, AdapterStatus
import kagya.learning.adapter_registry as adapter_registry_module
from tests.adapter_behavioral_helpers import (
    bind_runtime_behavioral_result,
    register_runtime_candidate,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_candidate_registration_writes_registry_entry(tmp_path: Path) -> None:
    registry = AdapterRegistry(_settings_for_tmp_registry(tmp_path))

    entry = registry.register_candidate(
        adapter_id="adapter-a",
        adapter_path=tmp_path / "adapter-a",
        dataset_path=tmp_path / "dataset.jsonl",
        dataset_hash="hash-a",
    )

    assert entry.status == AdapterStatus.CANDIDATE
    assert registry.path.exists()
    assert registry.lookup("adapter-a") == entry


def test_candidate_can_become_trial_active_when_score_is_high_enough(tmp_path: Path) -> None:
    registry = _registry_with_candidate(tmp_path, "adapter-a")

    entry = registry.apply_evaluation("adapter-a", score=0.9, result_path=tmp_path / "eval.json")

    assert entry.status == AdapterStatus.TRIAL_ACTIVE
    assert entry.eval_score == 0.9


def test_candidate_below_reject_threshold_becomes_rejected(tmp_path: Path) -> None:
    registry = _registry_with_candidate(tmp_path, "adapter-a")

    entry = registry.apply_evaluation("adapter-a", score=0.1, result_path=tmp_path / "eval.json")

    assert entry.status == AdapterStatus.REJECTED


def test_mid_range_score_leaves_adapter_as_candidate(tmp_path: Path) -> None:
    registry = _registry_with_candidate(tmp_path, "adapter-a")

    entry = registry.apply_evaluation("adapter-a", score=0.6, result_path=tmp_path / "eval.json")

    assert entry.status == AdapterStatus.CANDIDATE


def test_manual_approval_changes_trial_active_to_approved(tmp_path: Path) -> None:
    registry = _registry_with_candidate(tmp_path, "adapter-a")
    registry.apply_evaluation("adapter-a", score=0.9, result_path=tmp_path / "eval.json")

    entry = registry.approve("adapter-a", notes="manual approval")

    assert entry.status == AdapterStatus.APPROVED
    assert entry.notes == "manual approval"


def test_activation_changes_approved_to_active(tmp_path: Path) -> None:
    registry = _registry_with_candidate(tmp_path, "adapter-a")
    registry.apply_evaluation("adapter-a", score=0.9, result_path=tmp_path / "eval.json")
    bind_runtime_behavioral_result(registry, tmp_path, "adapter-a")
    registry.approve("adapter-a")

    entry = registry.activate("adapter-a")

    assert entry.status == AdapterStatus.ACTIVE


def test_existing_active_adapter_becomes_archived_when_new_adapter_activates(tmp_path: Path) -> None:
    registry = _registry_with_candidate(tmp_path, "adapter-a")
    registry.apply_evaluation("adapter-a", score=0.9, result_path=tmp_path / "eval-a.json")
    bind_runtime_behavioral_result(registry, tmp_path, "adapter-a")
    registry.approve("adapter-a")
    registry.activate("adapter-a")
    register_runtime_candidate(registry, tmp_path, "adapter-b")
    registry.apply_evaluation("adapter-b", score=0.9, result_path=tmp_path / "eval-b.json")
    bind_runtime_behavioral_result(registry, tmp_path, "adapter-b")
    registry.approve("adapter-b")

    registry.activate("adapter-b")

    assert registry.lookup("adapter-a").status == AdapterStatus.ARCHIVED
    assert registry.lookup("adapter-b").status == AdapterStatus.ACTIVE


def test_invalid_transitions_raise_errors(tmp_path: Path) -> None:
    registry = _registry_with_candidate(tmp_path, "adapter-a")

    with pytest.raises(ValueError, match="Invalid adapter status transition"):
        registry.activate("adapter-a")

    with pytest.raises(ValueError, match="Invalid adapter status transition"):
        registry.approve("adapter-a")


def test_concurrent_candidate_registrations_do_not_lose_updates(tmp_path: Path) -> None:
    registry = AdapterRegistry(_settings_for_tmp_registry(tmp_path))
    adapter_ids = [f"adapter-{index}" for index in range(20)]

    def register(adapter_id: str) -> None:
        registry.register_candidate(
            adapter_id=adapter_id,
            adapter_path=tmp_path / adapter_id,
            dataset_path=tmp_path / f"{adapter_id}.jsonl",
            dataset_hash=adapter_id,
        )

    with ThreadPoolExecutor(max_workers=len(adapter_ids)) as executor:
        list(executor.map(register, adapter_ids))

    assert {entry.adapter_id for entry in registry.list()} == set(adapter_ids)


def test_concurrent_activation_leaves_exactly_one_active_adapter(tmp_path: Path) -> None:
    registry = AdapterRegistry(_settings_for_tmp_registry(tmp_path))
    for adapter_id in ("adapter-a", "adapter-b"):
        register_runtime_candidate(registry, tmp_path, adapter_id)
        registry.apply_evaluation(adapter_id, score=0.9, result_path=tmp_path / "eval.json")
        bind_runtime_behavioral_result(registry, tmp_path, adapter_id)
        registry.approve(adapter_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(registry.activate, adapter_id) for adapter_id in ("adapter-a", "adapter-b")]
        for future in futures:
            future.result()

    assert sum(entry.status == AdapterStatus.ACTIVE for entry in registry.list()) == 1


def test_replace_failure_preserves_old_registry_and_cleans_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry_with_candidate(tmp_path, "adapter-a")
    old_contents = registry.path.read_bytes()

    def fail_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        assert Path(source).stat().st_mode & 0o777 == 0o600
        assert Path(target) == registry.path
        raise OSError("injected replace failure")

    monkeypatch.setattr(adapter_registry_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        registry.register_candidate(
            adapter_id="adapter-b",
            adapter_path=tmp_path / "adapter-b",
            dataset_path=tmp_path / "dataset-b.jsonl",
            dataset_hash="hash-b",
        )

    assert registry.path.read_bytes() == old_contents
    assert json.loads(old_contents)["adapters"][0]["adapter_id"] == "adapter-a"
    assert not list(tmp_path.glob(".adapter_registry.json.*.tmp"))


def _registry_with_candidate(tmp_path: Path, adapter_id: str) -> AdapterRegistry:
    registry = AdapterRegistry(_settings_for_tmp_registry(tmp_path))
    register_runtime_candidate(registry, tmp_path, adapter_id)
    return registry


def _settings_for_tmp_registry(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "path": tmp_path / "adapter_registry.json",
                    "eval_result_dir": tmp_path / "eval_results",
                    "eval_sets": [],
                    "trial_threshold": 0.8,
                    "reject_threshold": 0.4,
                }
            )
        }
    )
