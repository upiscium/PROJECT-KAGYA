from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from kagya.chat_jobs import (
    ChatJobRegistry,
    ChatJobIdempotencyTombstone,
    ChatJobRecord,
    encode_chat_request,
    parse_chat_job_registry,
    resolve_chat_job_registry_path,
    serialize_chat_job_registry,
    validate_chat_job_registry,
)
from kagya.operation_status import OperationState, OperationStatus, operation_now
from kagya.runtime import AgentRuntime
from kagya.security import EncryptionError, build_live_codecs
from kagya.security.migration import (
    migrate_chat_request_spool,
    reencrypt_live_state,
)
from tests.chat_job_helpers import request_codec
from tests.test_encrypted_restore import _authoritative_graph, _key, _settings


def _record(*, sealed_request: str, state: OperationState = OperationState.QUEUED):
    now = operation_now()
    operation_id = str(uuid4())
    event_id = str(uuid4())
    return ChatJobRecord(
        status=OperationStatus(
            operation_id=operation_id,
            event_id=event_id,
            status=state,
            status_sequence=1,
            queue_position=1 if state == OperationState.QUEUED else None,
            submitted_at=now,
            updated_at=now,
            completed_at=now if state == OperationState.COMPLETED else None,
            result_available=state == OperationState.COMPLETED,
        ),
        enqueue_sequence=1,
        client_id="client",
        idempotency_key=operation_id,
        correlation_id="context",
        sealed_request=sealed_request,
    )


def _seal_record(record: ChatJobRecord, text: str, codec=None) -> None:
    codec = codec or request_codec()
    record.sealed_request = encode_chat_request(
        {"text": text},
        codec,
        operation_id=record.status.operation_id,
        event_id=record.status.event_id,
    )


def _legacy_seal(value: dict[str, str], key: bytes) -> str:
    plaintext = json.dumps(value, separators=(",", ":")).encode()
    nonce = bytes(range(16))
    ciphertext = _legacy_xor_stream(plaintext, key, nonce)
    signature = hmac.digest(key, nonce + ciphertext, "sha256")
    return (nonce + signature + ciphertext).hex()


def _legacy_xor_stream(value: bytes, key: bytes, nonce: bytes) -> bytes:
    output = bytearray()
    for counter in range((len(value) + 31) // 32):
        output.extend(hmac.digest(key, nonce + counter.to_bytes(8, "big"), "sha256"))
    return bytes(left ^ right for left, right in zip(value, output, strict=False))


def test_request_spool_rejects_tamper_wrong_key_and_moved_ciphertext(
    tmp_path: Path,
) -> None:
    codec = request_codec()
    first = _record(sealed_request="")
    second = _record(sealed_request="")
    second.enqueue_sequence = 2
    _seal_record(first, "first", codec)
    _seal_record(second, "second", codec)
    path = tmp_path / "chat_jobs.json"
    path.write_bytes(serialize_chat_job_registry([first, second], []))
    validate_chat_job_registry(path, codec)

    original = path.read_bytes()
    values = json.loads(original)
    values[0]["sealed_request"], values[1]["sealed_request"] = (
        values[1]["sealed_request"],
        values[0]["sealed_request"],
    )
    path.write_text(json.dumps(values))
    with pytest.raises(EncryptionError, match="metadata"):
        validate_chat_job_registry(path, codec)

    path.write_bytes(original)
    values = json.loads(original)
    envelope = json.loads(values[0]["sealed_request"])
    envelope["ciphertext"] = envelope["ciphertext"][:-2] + "AA"
    values[0]["sealed_request"] = json.dumps(envelope)
    path.write_text(json.dumps(values))
    with pytest.raises(EncryptionError, match="authentication"):
        validate_chat_job_registry(path, codec)

    path.write_bytes(original)
    with pytest.raises(EncryptionError, match="authentication"):
        validate_chat_job_registry(path, request_codec(key=bytes(reversed(range(32)))))


def test_registry_requires_codec_and_rejects_legacy_or_mixed_active_requests(
    tmp_path: Path,
) -> None:
    runtime = AgentRuntime(queue_capacity=1)
    with pytest.raises(TypeError, match="request_codec"):
        ChatJobRegistry(tmp_path / "missing.json", runtime, lambda value: value)

    key = bytes([4]) * 32
    legacy = _record(sealed_request=_legacy_seal({"text": "legacy"}, key))
    encrypted = _record(sealed_request="")
    encrypted.enqueue_sequence = 2
    _seal_record(encrypted, "encrypted")
    path = tmp_path / "mixed.json"
    path.write_bytes(serialize_chat_job_registry([legacy, encrypted], []))
    with pytest.raises(EncryptionError, match="malformed or plaintext"):
        ChatJobRegistry(
            path,
            runtime,
            lambda value: value,
            request_codec=request_codec(),
        )


def test_atomic_legacy_migration_preserves_tombstone_and_removes_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch, live=True)
    path = resolve_chat_job_registry_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = bytes([9]) * 32
    record = _record(sealed_request=_legacy_seal({"text": "private"}, key))
    terminal = _record(sealed_request="", state=OperationState.COMPLETED)
    tombstone = ChatJobIdempotencyTombstone(
        status=terminal.status,
        client_id=terminal.client_id,
        idempotency_key=terminal.idempotency_key,
        expires_at=operation_now(),
    )
    path.write_bytes(serialize_chat_job_registry([record], [tombstone]))
    key_path = path.with_suffix(path.suffix + ".key")
    key_path.write_bytes(key)

    assert migrate_chat_request_spool(settings) == 1
    assert not key_path.exists()
    records, tombstones = parse_chat_job_registry(path.read_bytes())
    assert len(records) == 1
    assert len(tombstones) == 1
    assert "private" not in path.read_text()
    validate_chat_job_registry(path, build_live_codecs(settings).chat_request_spool)
    key_path.write_bytes(key)
    assert migrate_chat_request_spool(settings) == 0
    assert not key_path.exists()


def test_legacy_migration_rejects_mixed_store_and_replace_failure_is_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch, live=True)
    path = resolve_chat_job_registry_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = bytes([7]) * 32
    legacy = _record(sealed_request=_legacy_seal({"text": "legacy"}, key))
    encrypted = _record(sealed_request="")
    encrypted.enqueue_sequence = 2
    _seal_record(encrypted, "encrypted", build_live_codecs(settings).chat_request_spool)
    key_path = path.with_suffix(path.suffix + ".key")
    key_path.write_bytes(key)
    path.write_bytes(serialize_chat_job_registry([legacy, encrypted], []))
    mixed = path.read_bytes()
    with pytest.raises(EncryptionError, match="mixed legacy"):
        migrate_chat_request_spool(settings)
    assert path.read_bytes() == mixed
    assert key_path.read_bytes() == key

    path.write_bytes(serialize_chat_job_registry([legacy], []))
    original = path.read_bytes()
    real_replace = os.replace
    calls = 0

    def fail_publish(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publish failure")
        real_replace(source, destination)

    monkeypatch.setattr("kagya.security.migration.os.replace", fail_publish)
    with pytest.raises(OSError, match="injected"):
        migrate_chat_request_spool(settings)
    assert path.read_bytes() == original
    assert key_path.read_bytes() == key


def test_live_rotation_reencrypts_active_chat_request_and_preserves_tombstone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_settings = _settings(tmp_path, monkeypatch, live=True)
    _authoritative_graph(old_settings, b"state")
    old_codec = build_live_codecs(old_settings).chat_request_spool
    path = resolve_chat_job_registry_path(old_settings)
    record = _record(sealed_request="")
    _seal_record(record, "rotate-private", old_codec)
    terminal = _record(sealed_request="", state=OperationState.COMPLETED)
    tombstone = ChatJobIdempotencyTombstone(
        status=terminal.status,
        client_id=terminal.client_id,
        idempotency_key=terminal.idempotency_key,
        expires_at=operation_now(),
    )
    path.write_bytes(serialize_chat_job_registry([record], [tombstone]))

    monkeypatch.setenv("TEST_CURRENT_LIVE_KEY", _key(8))
    keys = old_settings.at_rest.live.keys.model_copy(
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
                        update={"keys": keys}
                    )
                }
            )
        }
    )
    assert reencrypt_live_state(rotated) == 4

    current_only = rotated.model_copy(
        update={
            "at_rest": rotated.at_rest.model_copy(
                update={
                    "live": rotated.at_rest.live.model_copy(
                        update={
                            "keys": keys.model_copy(
                                update={"allowed_old_key_envs": {}}
                            )
                        }
                    )
                }
            )
        }
    )
    validate_chat_job_registry(
        path, build_live_codecs(current_only).chat_request_spool
    )
    records, tombstones = parse_chat_job_registry(path.read_bytes())
    assert len(records) == 1
    assert len(tombstones) == 1
