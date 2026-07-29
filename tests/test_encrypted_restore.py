from base64 import b64encode
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kagya.config import Settings, load_settings
from kagya.runtime import AgentStateStore, EventJournal, StateWAL
from kagya.runtime.agent_state import default_agent_state_snapshot
from kagya.security import EncryptedCodec, EncryptionError, build_live_codecs
from kagya.security.backup import (
    BackupError,
    BackupManager,
    assert_no_incomplete_restore,
)
from kagya.security.crypto import KeyRing
from kagya.security.migration import reencrypt_live_state
from kagya.chat_jobs import resolve_chat_job_registry_path
from kagya.security.generation import (
    initialize_encrypted_state,
    require_encrypted_generation,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
SENTINEL = b"plaintext-secret-sentinel"


def _key(value: int) -> str:
    return b64encode(bytes([value]) * 32).decode("ascii")


def test_aead_nonce_uniqueness_tamper_and_key_rotation() -> None:
    old = bytes([1]) * 32
    current = bytes([2]) * 32
    old_codec = EncryptedCodec(
        enabled=True,
        purpose="live-state",
        context="agent-snapshot",
        key_ring=KeyRing("old", {"old": old}),
    )
    first = old_codec.encode(SENTINEL, metadata={"sequence": 1})
    second = old_codec.encode(SENTINEL, metadata={"sequence": 1})

    assert first != second
    assert SENTINEL not in first
    rotated = EncryptedCodec(
        enabled=True,
        purpose="live-state",
        context="agent-snapshot",
        key_ring=KeyRing("current", {"current": current, "old": old}),
    )
    assert rotated.decode(first, expected_metadata={"sequence": 1}) == SENTINEL
    denied = EncryptedCodec(
        enabled=True,
        purpose="live-state",
        context="agent-snapshot",
        key_ring=KeyRing("current", {"current": current}),
    )
    with pytest.raises(EncryptionError, match="not allowed"):
        denied.decode(first)
    envelope = json.loads(first)
    envelope["ciphertext"] = envelope["ciphertext"][:-2] + "AA"
    with pytest.raises(EncryptionError):
        rotated.decode(json.dumps(envelope).encode())


def test_live_files_are_encrypted_and_mixed_plaintext_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch, live=True)
    codecs = build_live_codecs(settings)
    snapshot = default_agent_state_snapshot(1.0)
    store = AgentStateStore(settings.agent_state.path, codec=codecs.snapshot)
    store.save(snapshot)
    wal = StateWAL(settings.agent_state_wal.path, codec=codecs.wal)
    wal.bootstrap(snapshot)
    journal = EventJournal(settings.agent_journal.path, codec=codecs.journal)
    journal.reconcile(snapshot)

    for path in (
        settings.agent_state.path,
        settings.agent_state_wal.path,
    ):
        assert b"optimal_loss" not in path.read_bytes()
    assert store.load(1.0) == snapshot
    settings.agent_state.path.write_bytes(snapshot.model_dump_json().encode())
    with pytest.raises(EncryptionError, match="plaintext"):
        AgentStateStore(settings.agent_state.path, codec=codecs.snapshot).load(1.0)


def test_live_rotation_rewrites_old_generation_and_old_key_can_be_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_settings = _settings(tmp_path, monkeypatch, live=True)
    old_codecs = build_live_codecs(old_settings)
    snapshot = default_agent_state_snapshot(1.0)
    AgentStateStore(old_settings.agent_state.path, codec=old_codecs.snapshot).save(
        snapshot
    )
    StateWAL(old_settings.agent_state_wal.path, codec=old_codecs.wal).bootstrap(
        snapshot
    )
    old_settings.agent_journal.path.touch(mode=0o600)

    monkeypatch.setenv("TEST_CURRENT_LIVE_KEY", _key(8))
    rotated_keys = old_settings.at_rest.live.keys.model_copy(
        update={
            "current_key_id": "current",
            "current_key_env": "TEST_CURRENT_LIVE_KEY",
            "allowed_old_key_envs": {"live-v1": "TEST_LIVE_KEY"},
        }
    )
    rotated = old_settings.model_copy(
        update={
            "at_rest": old_settings.at_rest.model_copy(
                update={
                    "live": old_settings.at_rest.live.model_copy(
                        update={"keys": rotated_keys}
                    )
                }
            )
        }
    )
    assert reencrypt_live_state(rotated) == 3
    denied_old = rotated.model_copy(
        update={
            "at_rest": rotated.at_rest.model_copy(
                update={
                    "live": rotated.at_rest.live.model_copy(
                        update={
                            "keys": rotated_keys.model_copy(
                                update={"allowed_old_key_envs": {}}
                            )
                        }
                    )
                }
            )
        }
    )
    current_codecs = build_live_codecs(denied_old)
    assert (
        AgentStateStore(
            denied_old.agent_state.path, codec=current_codecs.snapshot
        ).load(1.0)
        == snapshot
    )
    StateWAL(denied_old.agent_state_wal.path, codec=current_codecs.wal).verify()
    EventJournal(denied_old.agent_journal.path, codec=current_codecs.journal).verify()


def test_encrypted_backup_round_trip_incremental_and_public_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _authoritative_graph(settings, SENTINEL)
    chat_jobs = resolve_chat_job_registry_path(settings)
    chat_jobs.write_bytes(b"[]")
    adjacent_key = chat_jobs.with_suffix(chat_jobs.suffix + ".key")
    adjacent_key.write_bytes(b"LEGACY_RAW_KEY_SENTINEL")
    manager = BackupManager(settings)

    full = manager.create()
    bundle = settings.at_rest.backup.directory / f"{full.backup_id}.kgb"
    sidecar = bundle.with_suffix(".status.json")
    assert SENTINEL not in bundle.read_bytes()
    assert b"agent_state" not in sidecar.read_bytes()
    inventory = manager._inventory(None)
    assert any(entry.root == "chat_jobs" for entry in inventory)
    assert all(entry.relative_path != adjacent_key.name for entry in inventory)
    assert manager.verify(full.backup_id) == full

    settings.tools.path.write_bytes(b"changed")
    incremental = manager.create(base_backup_id=full.backup_id)
    assert incremental.base_backup_id == full.backup_id
    assert manager.verify(incremental.backup_id) == incremental

    monkeypatch.setenv("TEST_NEW_BACKUP_KEY", _key(7))
    rotated_key_settings = settings.at_rest.backup.keys.model_copy(
        update={
            "current_key_id": "backup-v2",
            "current_key_env": "TEST_NEW_BACKUP_KEY",
            "allowed_old_key_envs": {"backup-v1": "TEST_BACKUP_KEY"},
        }
    )
    rotated_settings = settings.model_copy(
        update={
            "at_rest": settings.at_rest.model_copy(
                update={
                    "backup": settings.at_rest.backup.model_copy(
                        update={"keys": rotated_key_settings}
                    )
                }
            )
        }
    )
    rotated = BackupManager(rotated_settings).rotate(full.backup_id)
    denied_old_settings = rotated_settings.model_copy(
        update={
            "at_rest": rotated_settings.at_rest.model_copy(
                update={
                    "backup": rotated_settings.at_rest.backup.model_copy(
                        update={
                            "keys": rotated_key_settings.model_copy(
                                update={"allowed_old_key_envs": {}}
                            )
                        }
                    )
                }
            )
        }
    )
    assert BackupManager(denied_old_settings).verify(rotated.backup_id) == rotated

    settings.tools.path.write_bytes(b"authoritative-newer")
    chat_jobs.write_bytes(b"[ ]")
    restored = manager.restore(
        full.backup_id, expected_manifest_hash=full.manifest_hash
    )
    assert restored == full
    assert settings.tools.path.read_bytes() == SENTINEL
    assert chat_jobs.read_bytes() == b"[]"
    assert adjacent_key.read_bytes() == b"LEGACY_RAW_KEY_SENTINEL"
    assert (settings.at_rest.backup.directory / "previous-generation").exists()


def test_backup_rejects_symlink_wrong_key_tamper_and_bad_incremental_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _authoritative_graph(settings, SENTINEL)
    linked = settings.actions.document_root / "linked"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(settings.tools.path)
    manager = BackupManager(settings)
    with pytest.raises(BackupError, match="symlinks"):
        manager.create()
    linked.unlink()
    full = manager.create()
    bundle = settings.at_rest.backup.directory / f"{full.backup_id}.kgb"

    monkeypatch.setenv("TEST_BACKUP_KEY", _key(9))
    with pytest.raises((BackupError, EncryptionError)):
        manager.verify(full.backup_id)
    monkeypatch.setenv("TEST_BACKUP_KEY", _key(2))
    original = bundle.read_bytes()
    lines = original.splitlines(keepends=True)
    header = json.loads(lines[0])
    header["key_ids"] = ["forged", "forged"]
    lines[0] = (
        json.dumps(
            header, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
        + b"\n"
    )
    bundle.write_bytes(b"".join(lines))
    with pytest.raises(BackupError):
        manager.verify(full.backup_id)
    bundle.write_bytes(original)
    data = bytearray(bundle.read_bytes())
    data[-20] ^= 1
    bundle.write_bytes(data)
    with pytest.raises(BackupError):
        manager.verify(full.backup_id)


def test_restore_crash_checkpoints_never_publish_incomplete_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _authoritative_graph(settings, SENTINEL)
    preview = BackupManager(settings).create()
    settings.tools.path.write_bytes(b"newer-authoritative-value")

    def fail_before(checkpoint: str) -> None:
        if checkpoint == "restore_before_swap":
            raise RuntimeError("simulated crash before swap")

    with pytest.raises(RuntimeError, match="before swap"):
        BackupManager(settings, failure_injector=fail_before).restore(
            preview.backup_id, expected_manifest_hash=preview.manifest_hash
        )
    assert settings.tools.path.read_bytes() == b"newer-authoritative-value"

    def fail_after(checkpoint: str) -> None:
        if checkpoint == "restore_after_swap":
            raise RuntimeError("simulated crash after swap")

    with pytest.raises(RuntimeError, match="after swap"):
        BackupManager(settings, failure_injector=fail_after).restore(
            preview.backup_id, expected_manifest_hash=preview.manifest_hash
        )
    assert settings.tools.path.read_bytes() == b"newer-authoritative-value"
    assert not (settings.at_rest.backup.directory / ".restore-in-progress").exists()


def test_incomplete_restore_marker_blocks_startup_and_status_redacts_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _authoritative_graph(settings, SENTINEL)
    manager = BackupManager(settings)
    manager.create()
    status = manager.list(1)[0].model_dump(mode="json")
    assert "path" not in json.dumps(status).lower()
    marker = settings.at_rest.backup.directory / ".restore-in-progress"
    marker.write_text("{}", encoding="ascii")
    with pytest.raises(BackupError, match="incomplete authoritative restore"):
        assert_no_incomplete_restore(settings)


def test_failed_rollback_preserves_marker_and_offline_retry_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _authoritative_graph(settings, SENTINEL)
    preview = BackupManager(settings).create()
    settings.tools.path.write_bytes(b"prior-generation")

    def fail(checkpoint: str) -> None:
        if checkpoint in {"restore_after_swap", "restore_rollback_replace"}:
            raise RuntimeError(checkpoint)

    with pytest.raises(BackupError, match="rollback failed"):
        BackupManager(settings, failure_injector=fail).restore(
            preview.backup_id, expected_manifest_hash=preview.manifest_hash
        )
    manager = BackupManager(settings)
    status = manager.recovery_status()
    assert status["phase"] == "rollback_failed"
    assert "pending_generation" not in status
    with pytest.raises(BackupError, match="incomplete authoritative restore"):
        assert_no_incomplete_restore(settings)

    assert manager.recover()["status"] == "rolled_back"
    assert settings.tools.path.read_bytes() == b"prior-generation"
    assert manager.recovery_status() == {"status": "clean"}


@pytest.mark.parametrize(
    "checkpoint", ["backup_after_inventory", "backup_stream_chunk"]
)
def test_backup_aborts_when_source_mutates_during_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint: str
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _authoritative_graph(settings, SENTINEL)
    mutated = False

    def mutate(current: str) -> None:
        nonlocal mutated
        if current == checkpoint and not mutated:
            mutated = True
            settings.tools.path.write_bytes(b"concurrently-mutated")

    with pytest.raises(BackupError, match="changed while streaming"):
        BackupManager(settings, failure_injector=mutate).create()
    assert BackupManager(settings).list() == []
    assert list(settings.at_rest.backup.directory.glob("*.kgb")) == []


@pytest.mark.parametrize("mutation", ["change_streamed", "add_file"])
def test_backup_requires_identical_complete_post_stream_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _authoritative_graph(settings, SENTINEL)
    mutated = False

    def mutate(checkpoint: str) -> None:
        nonlocal mutated
        if checkpoint != "backup_before_second_inventory" or mutated:
            return
        mutated = True
        if mutation == "change_streamed":
            settings.tools.path.write_bytes(b"changed-after-stream")
        else:
            added = settings.actions.document_root / "added-after-stream.txt"
            added.parent.mkdir(parents=True, exist_ok=True)
            added.write_bytes(b"new path")

    with pytest.raises(BackupError, match="inventory changed while streaming"):
        BackupManager(settings, failure_injector=mutate).create()
    assert BackupManager(settings).list() == []


@pytest.mark.parametrize(
    "checkpoint,occurrence",
    [
        ("restore_forward_planned", 1),
        ("restore_forward_replace", 1),
        ("restore_forward_replaced", 1),
        ("restore_forward_completed", 1),
        ("restore_forward_planned", 2),
        ("restore_forward_replace", 2),
        ("restore_forward_replaced", 2),
        ("restore_forward_completed", 2),
    ],
)
def test_write_ahead_restore_journal_recovers_every_rename_instruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str,
    occurrence: int,
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _authoritative_graph(settings, SENTINEL)
    preview = BackupManager(settings).create()
    settings.tools.path.write_bytes(b"verified-prior-generation")
    seen = 0

    class SimulatedProcessCrash(BaseException):
        pass

    def crash(current: str) -> None:
        nonlocal seen
        if current == checkpoint:
            seen += 1
            if seen == occurrence:
                raise SimulatedProcessCrash

    with pytest.raises(SimulatedProcessCrash):
        BackupManager(settings, failure_injector=crash).restore(
            preview.backup_id, expected_manifest_hash=preview.manifest_hash
        )
    manager = BackupManager(settings)
    assert manager.recovery_status()["status"] == "recovery_required"
    assert manager.recover()["status"] == "rolled_back"
    assert settings.tools.path.read_bytes() == b"verified-prior-generation"


def test_restore_offline_failure_never_activates_and_activation_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _authoritative_graph(settings, SENTINEL)
    preview = BackupManager(settings).create()
    settings.tools.path.write_bytes(b"prior")
    activated = False

    def activate() -> None:
        nonlocal activated
        activated = True

    with pytest.raises(RuntimeError, match="offline build"):
        BackupManager(settings).restore(
            preview.backup_id,
            expected_manifest_hash=preview.manifest_hash,
            after_publish=lambda: (_ for _ in ()).throw(RuntimeError("offline build")),
            activate=activate,
        )
    assert activated is False
    assert settings.tools.path.read_bytes() == b"prior"

    def fail_activation() -> None:
        raise RuntimeError("activation")

    with pytest.raises(BackupError, match="activation failed"):
        BackupManager(settings).restore(
            preview.backup_id,
            expected_manifest_hash=preview.manifest_hash,
            activate=fail_activation,
        )
    manager = BackupManager(settings)
    assert manager.recovery_status()["phase"] == "activation_failed"
    assert settings.tools.path.read_bytes() == SENTINEL
    assert manager.recover()["status"] == "rolled_back"
    assert settings.tools.path.read_bytes() == b"prior"


def test_retention_keeps_complete_transitive_chains_and_prunes_by_full_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    settings = settings.model_copy(
        update={
            "at_rest": settings.at_rest.model_copy(
                update={
                    "backup": settings.at_rest.backup.model_copy(
                        update={"retention_count": 1}
                    )
                }
            )
        }
    )
    _authoritative_graph(settings, SENTINEL)
    manager = BackupManager(settings)
    full = manager.create()
    settings.tools.path.write_bytes(b"incremental-one")
    first = manager.create(base_backup_id=full.backup_id)
    settings.tools.path.write_bytes(b"incremental-two")
    second = manager.create(base_backup_id=first.backup_id)
    assert {item.backup_id for item in manager.list()} == {
        full.backup_id,
        first.backup_id,
        second.backup_id,
    }

    newer_full = manager.create()
    assert [item.backup_id for item in manager.list()] == [newer_full.backup_id]
    for expired in (full, first, second):
        assert not (
            settings.at_rest.backup.directory / f"{expired.backup_id}.kgb"
        ).exists()


def test_missing_incremental_base_is_never_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _authoritative_graph(settings, SENTINEL)
    manager = BackupManager(settings)
    full = manager.create()
    settings.tools.path.write_bytes(b"incremental")
    incremental = manager.create(base_backup_id=full.backup_id)
    (settings.at_rest.backup.directory / f"{full.backup_id}.kgb").unlink()

    assert incremental.backup_id not in {item.backup_id for item in manager.list()}


def test_encrypted_production_requires_explicit_init_and_detects_file_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch, live=True)
    settings = settings.model_copy(
        update={
            "project": settings.project.model_copy(
                update={"environment": "production"}
            ),
            "at_rest": settings.at_rest.model_copy(
                update={
                    "live": settings.at_rest.live.model_copy(
                        update={
                            "generation_marker": tmp_path / "sealed-generation.json"
                        }
                    ),
                    "memory_encrypted_filesystem_attested": True,
                    "backup": settings.at_rest.backup.model_copy(
                        update={"encrypted_filesystem_attested": True}
                    ),
                }
            ),
        }
    )
    with pytest.raises(EncryptionError, match="not initialized"):
        require_encrypted_generation(settings)

    initialize_encrypted_state(settings)
    require_encrypted_generation(settings)
    settings.agent_state_wal.path.unlink()
    with pytest.raises(EncryptionError, match="incomplete"):
        require_encrypted_generation(settings)


@pytest.mark.parametrize(
    "required",
    ["snapshot", "wal", "journal", "generation_marker"],
)
def test_backup_rejects_each_missing_initialized_authoritative_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    required: str,
) -> None:
    settings = _settings(tmp_path, monkeypatch, live=True)
    initialize_encrypted_state(settings)
    paths = {
        "snapshot": settings.agent_state.path,
        "wal": settings.agent_state_wal.path,
        "journal": settings.agent_journal.path,
        "generation_marker": settings.at_rest.live.generation_marker,
    }
    paths[required].unlink()

    with pytest.raises(BackupError, match="incomplete|marker is missing"):
        BackupManager(settings).create()


def test_restore_staging_is_dedicated_private_cleaned_and_attested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _authoritative_graph(settings, SENTINEL)
    manager = BackupManager(settings)
    preview = manager.create()
    staging = settings.at_rest.backup.restore_staging_directory
    assert (
        staging.parent == Path(".kagya") or staging != settings.at_rest.backup.directory
    )
    assert manager.verify(preview.backup_id) == preview
    assert staging.stat().st_mode & 0o777 == 0o700
    assert list(staging.iterdir()) == []

    production = settings.model_copy(
        update={
            "project": settings.project.model_copy(
                update={"environment": "production"}
            ),
            "at_rest": settings.at_rest.model_copy(
                update={
                    "backup": settings.at_rest.backup.model_copy(
                        update={"encrypted_filesystem_attested": False}
                    )
                }
            ),
        }
    )
    with pytest.raises(BackupError, match="not attested"):
        BackupManager(production).verify(preview.backup_id)


def test_production_requires_encryption_and_memory_attestation() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["project"]["environment"] = "production"
    raw["adapter_registry"]["behavioral_activation_policy"] = "real_model_required"
    with pytest.raises(ValidationError, match="live authoritative encryption"):
        Settings.model_validate(deepcopy(raw))
    raw["at_rest"]["live"]["enabled"] = True
    with pytest.raises(ValidationError, match="encrypted filesystem attestation"):
        Settings.model_validate(raw)
    raw["at_rest"]["memory_encrypted_filesystem_attested"] = True
    with pytest.raises(ValidationError, match="restore staging"):
        Settings.model_validate(raw)


def test_key_loading_is_strict_and_missing_key_fails_startup_codec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch, live=True)
    monkeypatch.delenv("TEST_LIVE_KEY")
    with pytest.raises(EncryptionError, match="unavailable"):
        build_live_codecs(settings)
    monkeypatch.setenv("TEST_LIVE_KEY", b64encode(b"short").decode())
    with pytest.raises(EncryptionError, match="exactly 32 bytes"):
        build_live_codecs(settings)


def _settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    live: bool = False,
) -> Settings:
    settings = load_settings(CONFIG_PATH)
    monkeypatch.setenv("TEST_LIVE_KEY", _key(1))
    monkeypatch.setenv("TEST_BACKUP_KEY", _key(2))
    monkeypatch.setenv("TEST_ADAPTER_KEY", _key(3))
    at_rest = settings.at_rest.model_copy(
        update={
            "live": settings.at_rest.live.model_copy(
                update={
                    "enabled": live,
                    "generation_marker": tmp_path / "sealed-generation.json",
                    "keys": settings.at_rest.live.keys.model_copy(
                        update={"current_key_env": "TEST_LIVE_KEY"}
                    ),
                }
            ),
            "backup": settings.at_rest.backup.model_copy(
                update={
                    "directory": tmp_path / "backups",
                    "restore_staging_directory": tmp_path / "restore-staging",
                    "keys": settings.at_rest.backup.keys.model_copy(
                        update={"current_key_env": "TEST_BACKUP_KEY"}
                    ),
                    "adapter_keys": settings.at_rest.backup.adapter_keys.model_copy(
                        update={"current_key_env": "TEST_ADAPTER_KEY"}
                    ),
                }
            ),
        }
    )
    return settings.model_copy(
        update={
            "at_rest": at_rest,
            "agent_state": settings.agent_state.model_copy(
                update={"path": tmp_path / "state.json"}
            ),
            "agent_journal": settings.agent_journal.model_copy(
                update={"path": tmp_path / "journal.jsonl"}
            ),
            "agent_state_wal": settings.agent_state_wal.model_copy(
                update={"path": tmp_path / "private" / "wal.jsonl"}
            ),
            "memory": settings.memory.model_copy(
                update={"persist_directory": tmp_path / "memory"}
            ),
            "sleep": settings.sleep.model_copy(
                update={
                    "dream_dataset_path": tmp_path / "dreams.jsonl",
                    "job_registry_path": tmp_path / "jobs.json",
                    "training_artifact_directory": tmp_path / "training",
                }
            ),
            "qlora": settings.qlora.model_copy(
                update={"output_dir": tmp_path / "adapters"}
            ),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "path": tmp_path / "adapters.json",
                    "eval_result_dir": tmp_path / "evals",
                }
            ),
            "tools": settings.tools.model_copy(
                update={
                    "path": tmp_path / "tools.json",
                    "audit_path": tmp_path / "tools.jsonl",
                }
            ),
            "actions": settings.actions.model_copy(
                update={
                    "document_root": tmp_path / "documents",
                    "calendar_path": tmp_path / "calendar.json",
                }
            ),
        }
    )


def _authoritative_graph(settings: Settings, content: bytes) -> None:
    codecs = build_live_codecs(settings)
    snapshot = default_agent_state_snapshot(1.0)
    AgentStateStore(settings.agent_state.path, codec=codecs.snapshot).save(snapshot)
    StateWAL(settings.agent_state_wal.path, codec=codecs.wal).bootstrap(snapshot)
    settings.agent_journal.path.touch(mode=0o600)
    settings.tools.path.write_bytes(content)
