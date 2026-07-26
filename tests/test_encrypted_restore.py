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
from kagya.security.backup import BackupError, BackupManager
from kagya.security.crypto import KeyRing
from kagya.security.migration import reencrypt_live_state


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
        AgentStateStore(
            settings.agent_state.path, codec=codecs.snapshot
        ).load(1.0)


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
    assert AgentStateStore(
        denied_old.agent_state.path, codec=current_codecs.snapshot
    ).load(1.0) == snapshot
    StateWAL(denied_old.agent_state_wal.path, codec=current_codecs.wal).verify()
    EventJournal(
        denied_old.agent_journal.path, codec=current_codecs.journal
    ).verify()


def test_encrypted_backup_round_trip_incremental_and_public_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _authoritative_graph(settings, SENTINEL)
    manager = BackupManager(settings)

    full = manager.create()
    bundle = settings.at_rest.backup.directory / f"{full.backup_id}.kgb"
    sidecar = bundle.with_suffix(".status.json")
    assert SENTINEL not in bundle.read_bytes()
    assert b"agent_state" not in sidecar.read_bytes()
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
    restored = manager.restore(
        full.backup_id, expected_manifest_hash=full.manifest_hash
    )
    assert restored == full
    assert settings.tools.path.read_bytes() == SENTINEL
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
        json.dumps(header, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        .encode("ascii")
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


def test_production_requires_encryption_and_memory_attestation() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["project"]["environment"] = "production"
    raw["adapter_registry"]["behavioral_activation_policy"] = "real_model_required"
    with pytest.raises(ValidationError, match="live authoritative encryption"):
        Settings.model_validate(deepcopy(raw))
    raw["at_rest"]["live"]["enabled"] = True
    with pytest.raises(ValidationError, match="encrypted filesystem attestation"):
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
                    "keys": settings.at_rest.live.keys.model_copy(
                        update={"current_key_env": "TEST_LIVE_KEY"}
                    ),
                }
            ),
            "backup": settings.at_rest.backup.model_copy(
                update={
                    "directory": tmp_path / "backups",
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
