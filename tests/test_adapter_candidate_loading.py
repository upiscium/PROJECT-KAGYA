from pathlib import Path

from kagya.config import load_settings
from kagya.learning import AdapterRegistry
from kagya.models.transformers_provider import is_registry_approved_adapter


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_candidate_adapter_is_loadable_only_for_evaluation(tmp_path: Path) -> None:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"path": tmp_path / "registry.json"}
            )
        }
    )


def test_archived_adapter_is_loadable_only_for_rollback(tmp_path: Path) -> None:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"path": tmp_path / "registry.json"}
            )
        }
    )
    adapter_path = tmp_path / "archived"
    adapter_path.mkdir()
    registry = AdapterRegistry(settings)
    registry.register_candidate(
        adapter_id="archived",
        adapter_path=adapter_path,
        dataset_path=tmp_path / "dataset.jsonl",
        dataset_hash="hash",
    )
    registry.apply_evaluation("archived", score=0.9, result_path=tmp_path / "eval")
    registry.approve("archived")
    registry.activate("archived", activation_sequence=1)
    registry.restore_active(None, activation_sequence=2)

    assert is_registry_approved_adapter(settings, adapter_path) is False
    assert (
        is_registry_approved_adapter(settings, adapter_path, allow_archived=True)
        is True
    )
    adapter_path = tmp_path / "candidate"
    adapter_path.mkdir()
    AdapterRegistry(settings).register_candidate(
        adapter_id="candidate",
        adapter_path=adapter_path,
        dataset_path=tmp_path / "dataset.jsonl",
        dataset_hash="hash",
    )

    assert is_registry_approved_adapter(settings, adapter_path) is False
    assert (
        is_registry_approved_adapter(
            settings, adapter_path, allow_candidate=True
        )
        is True
    )
