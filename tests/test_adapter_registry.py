from pathlib import Path

import pytest

from kagya.config import Settings, load_settings
from kagya.learning import AdapterRegistry, AdapterStatus


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
    registry.approve("adapter-a")

    entry = registry.activate("adapter-a")

    assert entry.status == AdapterStatus.ACTIVE


def test_existing_active_adapter_becomes_archived_when_new_adapter_activates(tmp_path: Path) -> None:
    registry = _registry_with_candidate(tmp_path, "adapter-a")
    registry.apply_evaluation("adapter-a", score=0.9, result_path=tmp_path / "eval-a.json")
    registry.approve("adapter-a")
    registry.activate("adapter-a")
    registry.register_candidate(
        adapter_id="adapter-b",
        adapter_path=tmp_path / "adapter-b",
        dataset_path=tmp_path / "dataset-b.jsonl",
        dataset_hash="hash-b",
    )
    registry.apply_evaluation("adapter-b", score=0.9, result_path=tmp_path / "eval-b.json")
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


def _registry_with_candidate(tmp_path: Path, adapter_id: str) -> AdapterRegistry:
    registry = AdapterRegistry(_settings_for_tmp_registry(tmp_path))
    registry.register_candidate(
        adapter_id=adapter_id,
        adapter_path=tmp_path / adapter_id,
        dataset_path=tmp_path / "dataset.jsonl",
        dataset_hash="hash",
    )
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
