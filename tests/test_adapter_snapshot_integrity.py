import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from kagya.artifact_provenance import build_adapter_artifact_manifest
from kagya.config import load_settings
from kagya.models import load_model_provider
from kagya.models.transformers_provider import TransformersProvider
from kagya.training.artifacts import sha256_file_map


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_restart_rejects_mutated_active_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, adapter, adapter_hash, manifest = _adapter_fixture(tmp_path, "active")
    provider = load_model_provider(
        settings,
        adapter_path=adapter,
        expected_adapter_hash=adapter_hash,
        expected_adapter_manifest=manifest,
    )
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(provider, "_load_model", lambda model_id: object())

    with pytest.raises(RuntimeError, match="manifest mismatch|registry hash mismatch"):
        provider.get_model()


def test_source_mutation_during_peft_load_cannot_change_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, adapter, adapter_hash, manifest = _adapter_fixture(tmp_path, "approved")
    original = (adapter / "adapter_config.json").read_bytes()
    loaded: dict[str, bytes | Path] = {}

    def from_pretrained(model: object, path: str) -> object:
        snapshot = Path(path)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        loaded["content"] = (snapshot / "adapter_config.json").read_bytes()
        loaded["path"] = snapshot
        (adapter / "adapter_config.json").write_bytes(original)
        return _attached_model(settings)

    monkeypatch.setattr(
        "kagya.models.transformers_provider.PeftModel.from_pretrained", from_pretrained
    )
    provider = TransformersProvider(
        settings,
        adapter_path=adapter,
        model=object(),
        expected_adapter_hash=adapter_hash,
        expected_adapter_manifest=manifest,
    )

    assert loaded["content"] == original
    assert loaded["path"] != adapter
    assert provider.adapter_artifact_manifest_hash == manifest.sha256
    assert provider.adapter_snapshot_manifest_hash == manifest.sha256
    assert provider.adapter_snapshot_hash == adapter_hash


def test_snapshot_path_replacement_during_peft_load_uses_pinned_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, adapter, adapter_hash, manifest = _adapter_fixture(tmp_path, "approved")
    original = (adapter / "adapter_config.json").read_bytes()
    runtime_snapshot = (
        settings.adapter_registry.path.parent / ".adapter-runtime" / adapter_hash
    )
    loaded: dict[str, object] = {}

    def from_pretrained(model: object, path: str) -> object:
        proc_snapshot = Path(path)
        detached = runtime_snapshot.with_name(f"{adapter_hash}.detached")
        replacement = runtime_snapshot.with_name(f"{adapter_hash}.replacement")
        replacement.mkdir()
        (replacement / "adapter_config.json").write_text("{}", encoding="utf-8")
        runtime_snapshot.rename(detached)
        replacement.rename(runtime_snapshot)
        try:
            loaded["path"] = path
            loaded["content"] = (proc_snapshot / "adapter_config.json").read_bytes()
        finally:
            runtime_snapshot.rename(replacement)
            detached.rename(runtime_snapshot)
            (replacement / "adapter_config.json").unlink()
            replacement.rmdir()
        return _attached_model(settings)

    monkeypatch.setattr(
        "kagya.models.transformers_provider.PeftModel.from_pretrained", from_pretrained
    )
    provider = TransformersProvider(
        settings,
        adapter_path=adapter,
        model=object(),
        expected_adapter_hash=adapter_hash,
        expected_adapter_manifest=manifest,
    )

    assert str(loaded["path"]).startswith("/proc/self/fd/")
    assert loaded["content"] == original
    assert provider.adapter_artifact_manifest == manifest


def test_snapshot_fd_is_retained_through_attached_config_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, adapter, adapter_hash, manifest = _adapter_fixture(tmp_path, "approved")
    loaded_path: Path | None = None

    def from_pretrained(model: object, path: str) -> object:
        nonlocal loaded_path
        loaded_path = Path(path)
        return _attached_model(settings)

    def verify_config(model: object, expected: object) -> None:
        assert loaded_path is not None
        assert (loaded_path / "adapter_config.json").is_file()

    monkeypatch.setattr(
        "kagya.models.transformers_provider.PeftModel.from_pretrained", from_pretrained
    )
    monkeypatch.setattr(
        "kagya.models.transformers_provider.verify_attached_adapter_config",
        verify_config,
    )
    TransformersProvider(
        settings,
        adapter_path=adapter,
        model=object(),
        expected_adapter_hash=adapter_hash,
        expected_adapter_manifest=manifest,
    )

    assert loaded_path is not None
    assert not loaded_path.exists()


def test_fd_backed_manifest_rejects_symlink_added_during_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, adapter, adapter_hash, manifest = _adapter_fixture(tmp_path, "approved")

    def from_pretrained(model: object, path: str) -> object:
        snapshot = Path(path)
        os.chmod(snapshot, 0o700)
        (snapshot / "link.bin").symlink_to(snapshot / "adapter_config.json")
        return _attached_model(settings)

    monkeypatch.setattr(
        "kagya.models.transformers_provider.PeftModel.from_pretrained", from_pretrained
    )

    with pytest.raises(ValueError, match="symbolic link"):
        TransformersProvider(
            settings,
            adapter_path=adapter,
            model=object(),
            expected_adapter_hash=adapter_hash,
            expected_adapter_manifest=manifest,
        )


def test_adapter_load_fails_closed_without_linux_procfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, adapter, adapter_hash, manifest = _adapter_fixture(tmp_path, "approved")
    monkeypatch.setattr("kagya.models.transformers_provider.sys.platform", "darwin")

    with pytest.raises(RuntimeError, match="requires Linux procfs"):
        TransformersProvider(
            settings,
            adapter_path=adapter,
            model=object(),
            expected_adapter_hash=adapter_hash,
            expected_adapter_manifest=manifest,
        )


def test_source_mutation_during_snapshot_is_rejected_and_temp_is_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, adapter, adapter_hash, manifest = _adapter_fixture(tmp_path, "approved")
    from kagya.models import transformers_provider as provider_module

    original_copy = provider_module.shutil.copyfileobj
    mutated = False

    def mutating_copy(source, target) -> None:
        nonlocal mutated
        original_copy(source, target)
        if not mutated:
            mutated = True
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(provider_module.shutil, "copyfileobj", mutating_copy)

    with pytest.raises(RuntimeError, match="changed while creating"):
        TransformersProvider(
            settings,
            adapter_path=adapter,
            model=object(),
            expected_adapter_hash=adapter_hash,
            expected_adapter_manifest=manifest,
        )

    runtime_root = settings.adapter_registry.path.parent / ".adapter-runtime"
    assert not list(runtime_root.glob("*.tmp"))
    assert not list(runtime_root.glob(".*.tmp"))


def test_concurrent_snapshots_for_same_hash_are_not_corrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, adapter, adapter_hash, manifest = _adapter_fixture(tmp_path, "approved")
    loaded_paths: list[Path] = []

    def from_pretrained(model: object, path: str) -> object:
        loaded_paths.append(Path(path))
        return _attached_model(settings)

    monkeypatch.setattr(
        "kagya.models.transformers_provider.PeftModel.from_pretrained", from_pretrained
    )

    def load() -> TransformersProvider:
        return TransformersProvider(
            settings,
            adapter_path=adapter,
            model=object(),
            expected_adapter_hash=adapter_hash,
            expected_adapter_manifest=manifest,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        providers = list(executor.map(lambda _: load(), range(8)))

    assert all(str(path).startswith("/proc/self/fd/") for path in loaded_paths)
    assert all(
        item.adapter_artifact_manifest_hash == manifest.sha256 for item in providers
    )


def test_successful_restart_load_reports_snapshot_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, adapter, adapter_hash, manifest = _adapter_fixture(tmp_path, "active")
    monkeypatch.setattr(
        "kagya.models.transformers_provider.PeftModel.from_pretrained",
        lambda model, path: _attached_model(settings),
    )
    provider = load_model_provider(
        settings,
        adapter_path=adapter,
        expected_adapter_hash=adapter_hash,
        expected_adapter_manifest=manifest,
    )
    monkeypatch.setattr(provider, "_load_model", lambda model_id: object())

    provider.get_model()

    assert provider.adapter_artifact_manifest == manifest
    assert provider.adapter_artifact_manifest_hash == manifest.sha256
    assert provider.adapter_snapshot_manifest_hash == manifest.sha256
    assert provider.adapter_snapshot_hash == adapter_hash
    assert str(adapter.resolve()) not in json.dumps(
        provider.adapter_artifact_manifest.model_dump(mode="json")
    )


def _adapter_fixture(tmp_path: Path, state: str) -> tuple[object, Path, str, object]:
    settings = load_settings(CONFIG_PATH)
    registry_path = tmp_path / "registry.json"
    settings = settings.model_copy(
        update={
            "model": settings.model.model_copy(update={"provider": "transformers"}),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"path": registry_path}
            ),
        }
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = json.dumps(
        {
            "peft_type": "LORA",
            "base_model_name_or_path": settings.model.primary_id,
            "revision": settings.model.revision,
            "target_modules": ["q_proj"],
            "r": 4,
            "lora_alpha": 8,
        },
        sort_keys=True,
    ).encode()
    (adapter / "adapter_config.json").write_bytes(config)
    adapter_hash = sha256_file_map({"adapter/adapter_config.json": config})
    registry_path.write_text(
        json.dumps({"adapters": [{"path": str(adapter), "state": state}]}),
        encoding="utf-8",
    )
    manifest = build_adapter_artifact_manifest(adapter)
    return settings, adapter, adapter_hash, manifest


def _attached_model(settings: object) -> object:
    model = settings.model
    config = SimpleNamespace(
        peft_type="LORA",
        base_model_name_or_path=model.primary_id,
        revision=model.revision,
        target_modules={"q_proj"},
        r=4,
        lora_alpha=8,
    )
    return SimpleNamespace(peft_config={"default": config})
