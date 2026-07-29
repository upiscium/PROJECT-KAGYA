"""Explicit one-shot plaintext to encrypted live-state migration."""

from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
from uuid import uuid4

from kagya.chat_jobs import (
    ChatJobRecord,
    decode_chat_request,
    encode_chat_request,
    parse_chat_job_registry,
    resolve_chat_job_registry_path,
    serialize_chat_job_registry,
    validate_chat_request_records,
)
from kagya.config.schema import Settings
from kagya.runtime.agent_state import AgentStateSnapshot
from kagya.runtime.event_journal import EventJournal, JournalRecord
from kagya.runtime.state_wal import StateWAL, StateWalRecord
from kagya.security.crypto import EncryptedCodec, EncryptionError
from kagya.security.live import build_live_codecs
from kagya.security.generation import seal_encrypted_generation


def migrate_live_state(settings: Settings) -> int:
    if not settings.at_rest.live.enabled:
        raise EncryptionError("live encryption must be enabled for migration")
    codecs = build_live_codecs(settings)
    StateWAL(settings.agent_state_wal.path).verify()
    EventJournal(
        settings.agent_journal.path,
        retained_files=settings.agent_journal.retained_files,
    ).verify()
    targets: list[tuple[Path, bytes]] = []
    if settings.agent_state.path.exists():
        plaintext = settings.agent_state.path.read_bytes()
        snapshot = AgentStateSnapshot.model_validate_json(plaintext)
        targets.append(
            (
                settings.agent_state.path,
                codecs.snapshot.encode(
                    snapshot.model_dump_json().encode("utf-8"),
                    metadata={"record_type": "snapshot"},
                ),
            )
        )
    targets.extend(
        _migrate_lines(settings.agent_state_wal.path, codecs.wal, StateWalRecord)
    )
    journal_paths = [
        settings.agent_journal.path.with_name(
            f"{settings.agent_journal.path.name}.{index}"
        )
        for index in range(settings.agent_journal.retained_files, 0, -1)
    ] + [settings.agent_journal.path]
    for path in journal_paths:
        targets.extend(_migrate_lines(path, codecs.journal, JournalRecord))
    if (
        len(
            {path for path, _encoded in targets}
            & {
                settings.agent_state.path,
                settings.agent_state_wal.path,
                settings.agent_journal.path,
            }
        )
        != 3
    ):
        raise EncryptionError("plaintext authoritative generation is incomplete")
    return _replace_targets(
        targets, finalize=lambda: seal_encrypted_generation(settings)
    )


def reencrypt_live_state(settings: Settings) -> int:
    """Rewrite allowed old live generations under the current write generation."""

    if not settings.at_rest.live.enabled:
        raise EncryptionError("live encryption must be enabled for rotation")
    codecs = build_live_codecs(settings)
    StateWAL(settings.agent_state_wal.path, codec=codecs.wal).verify()
    EventJournal(
        settings.agent_journal.path,
        retained_files=settings.agent_journal.retained_files,
        codec=codecs.journal,
    ).verify()
    targets: list[tuple[Path, bytes]] = []
    if settings.agent_state.path.exists():
        plaintext = codecs.snapshot.decode(
            settings.agent_state.path.read_bytes(),
            expected_metadata={"record_type": "snapshot"},
        )
        snapshot = AgentStateSnapshot.model_validate_json(plaintext)
        targets.append(
            (
                settings.agent_state.path,
                codecs.snapshot.encode(
                    snapshot.model_dump_json().encode("utf-8"),
                    metadata={"record_type": "snapshot"},
                ),
            )
        )
    targets.extend(
        _reencrypt_lines(settings.agent_state_wal.path, codecs.wal, StateWalRecord)
    )
    journal_paths = [
        settings.agent_journal.path.with_name(
            f"{settings.agent_journal.path.name}.{index}"
        )
        for index in range(settings.agent_journal.retained_files, 0, -1)
    ] + [settings.agent_journal.path]
    for path in journal_paths:
        targets.extend(_reencrypt_lines(path, codecs.journal, JournalRecord))
    targets.extend(
        _reencrypt_chat_request_spool(
            resolve_chat_job_registry_path(settings), codecs.chat_request_spool
        )
    )
    return _replace_targets(
        targets, finalize=lambda: seal_encrypted_generation(settings)
    )


def migrate_chat_request_spool(settings: Settings) -> int:
    """Atomically replace the retired adjacent-key request spool format."""

    if not settings.at_rest.live.enabled:
        raise EncryptionError("live encryption must be enabled for migration")
    path = resolve_chat_job_registry_path(settings)
    if not path.exists():
        return 0
    key_path = path.with_suffix(path.suffix + ".key")
    codec = build_live_codecs(settings).chat_request_spool
    records, tombstones = parse_chat_job_registry(path.read_bytes())
    undecoded: list[ChatJobRecord] = []
    for record in records:
        if not record.sealed_request:
            continue
        try:
            decode_chat_request(
                record.sealed_request,
                codec,
                operation_id=record.status.operation_id,
                event_id=record.status.event_id,
            )
        except EncryptionError:
            undecoded.append(record)
    if not undecoded:
        validate_chat_request_records(records, codec)
        if key_path.exists():
            key_path.unlink()
            _fsync_directory(key_path.parent)
        return 0

    try:
        legacy_key = key_path.read_bytes()
    except FileNotFoundError as exc:
        raise EncryptionError("legacy chat request spool key is unavailable") from exc
    if len(legacy_key) != 32:
        raise EncryptionError("legacy chat request spool key must be exactly 32 bytes")
    legacy = [
        (record, _decode_legacy_chat_request(record.sealed_request, legacy_key))
        for record in undecoded
    ]
    encrypted = sum(bool(record.sealed_request) for record in records) - len(legacy)
    if encrypted:
        raise EncryptionError("mixed legacy and encrypted chat request spool")

    terminal = {"completed", "failed", "canceled"}
    for record, request in legacy:
        if record.status.status.value in terminal:
            record.sealed_request = ""
            continue
        record.sealed_request = encode_chat_request(
            request,
            codec,
            operation_id=record.status.operation_id,
            event_id=record.status.event_id,
        )
    validate_chat_request_records(records, codec)
    encoded = serialize_chat_job_registry(records, tombstones)
    replaced = _replace_targets([(path, encoded)])
    key_path.unlink(missing_ok=True)
    _fsync_directory(key_path.parent)
    return replaced


def _replace_targets(
    targets: list[tuple[Path, bytes]], *, finalize: Callable[[], object] | None = None
) -> int:
    if not targets:
        return 0
    staged: list[tuple[Path, Path]] = []
    replaced: list[tuple[Path, Path]] = []
    try:
        for path, encoded in targets:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".encrypted", dir=path.parent
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as output:
                os.fchmod(output.fileno(), 0o600)
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            staged.append((path, temporary))
        for path, temporary in staged:
            previous = path.with_name(
                f".{path.name}.{uuid4()}.migration-previous"
            )
            os.replace(path, previous)
            try:
                os.replace(temporary, path)
            except Exception:
                os.replace(previous, path)
                raise
            _fsync_directory(path.parent)
            replaced.append((path, previous))
        if finalize is not None:
            finalize()
        return len(replaced)
    except Exception:
        for path, previous in reversed(replaced):
            path.unlink(missing_ok=True)
            os.replace(previous, path)
            _fsync_directory(path.parent)
        raise
    finally:
        for _path, temporary in staged:
            temporary.unlink(missing_ok=True)
        for _path, previous in replaced:
            previous.unlink(missing_ok=True)


def _migrate_lines(
    path: Path, codec: EncryptedCodec, model: type[Any]
) -> list[tuple[Path, bytes]]:
    if not path.exists():
        return []
    encoded_lines: list[bytes] = []
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        record = model.model_validate_json(line)
        metadata = {
            "processing_sequence": record.processing_sequence,
            "record_id": record.record_id,
        }
        encoded_lines.append(codec.encode(line, metadata=metadata))
    encoded = b"\n".join(encoded_lines)
    return [(path, encoded + (b"\n" if encoded else b""))]


def _reencrypt_lines(
    path: Path, codec: EncryptedCodec, model: type[Any]
) -> list[tuple[Path, bytes]]:
    if not path.exists():
        return []
    encoded_lines: list[bytes] = []
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        plaintext = codec.decode(line)
        record = model.model_validate_json(plaintext)
        metadata = {
            "processing_sequence": record.processing_sequence,
            "record_id": record.record_id,
        }
        codec.decode(line, expected_metadata=metadata)
        encoded_lines.append(codec.encode(plaintext, metadata=metadata))
    encoded = b"\n".join(encoded_lines)
    return [(path, encoded + (b"\n" if encoded else b""))]


def _reencrypt_chat_request_spool(
    path: Path, codec: EncryptedCodec
) -> list[tuple[Path, bytes]]:
    if not path.exists():
        return []
    records, tombstones = parse_chat_job_registry(path.read_bytes())
    validate_chat_request_records(records, codec)
    for record in records:
        if not record.sealed_request:
            continue
        request = decode_chat_request(
            record.sealed_request,
            codec,
            operation_id=record.status.operation_id,
            event_id=record.status.event_id,
        )
        record.sealed_request = encode_chat_request(
            request,
            codec,
            operation_id=record.status.operation_id,
            event_id=record.status.event_id,
        )
    return [(path, serialize_chat_job_registry(records, tombstones))]


def _decode_legacy_chat_request(value: str, key: bytes) -> dict[str, Any]:
    try:
        sealed = bytes.fromhex(value)
    except ValueError as exc:
        raise EncryptionError("legacy chat request spool is invalid") from exc
    if len(sealed) < 48:
        raise EncryptionError("legacy chat request spool is invalid")
    nonce, signature, ciphertext = sealed[:16], sealed[16:48], sealed[48:]
    if not hmac.compare_digest(
        signature, hmac.digest(key, nonce + ciphertext, "sha256")
    ):
        raise EncryptionError("legacy chat request spool authentication failed")
    try:
        opened = json.loads(_legacy_xor_stream(ciphertext, key, nonce))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EncryptionError("legacy chat request spool is invalid") from exc
    if not isinstance(opened, dict):
        raise EncryptionError("legacy chat request spool is invalid")
    return opened


def _legacy_xor_stream(value: bytes, key: bytes, nonce: bytes) -> bytes:
    output = bytearray()
    for counter in range((len(value) + 31) // 32):
        output.extend(hmac.digest(key, nonce + counter.to_bytes(8, "big"), "sha256"))
    return bytes(left ^ right for left, right in zip(value, output, strict=False))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
