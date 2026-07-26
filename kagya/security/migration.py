"""Explicit one-shot plaintext to encrypted live-state migration."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Callable
from uuid import uuid4

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
    return _replace_targets(
        targets, finalize=lambda: seal_encrypted_generation(settings)
    )


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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
