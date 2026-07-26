"""Sealed authoritative live-state generation lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from kagya.config.schema import ProjectEnvironment, Settings
from kagya.runtime.agent_state import AgentStateStore, default_agent_state_snapshot
from kagya.runtime.event_journal import EventJournal, hash_snapshot
from kagya.runtime.state_wal import StateWAL
from kagya.security.crypto import EncryptionError
from kagya.security.live import build_live_codecs


_MARKER_VERSION = 1


def initialize_encrypted_state(settings: Settings) -> str:
    """Create one coherent encrypted empty generation and seal it last."""

    if not settings.at_rest.live.enabled:
        raise EncryptionError("live encryption must be enabled for initialization")
    marker = settings.at_rest.live.generation_marker
    if marker.exists() or any(path.exists() for path in _authoritative_paths(settings)):
        raise EncryptionError("authoritative encrypted state is already initialized")
    codecs = build_live_codecs(settings)
    snapshot = default_agent_state_snapshot(settings.emotion.baseline_surprisal)
    try:
        AgentStateStore(settings.agent_state.path, codec=codecs.snapshot).save(snapshot)
        StateWAL(settings.agent_state_wal.path, codec=codecs.wal).bootstrap(snapshot)
        settings.agent_journal.path.parent.mkdir(
            parents=True, exist_ok=True, mode=0o700
        )
        journal_descriptor = os.open(
            settings.agent_journal.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            os.fsync(journal_descriptor)
        finally:
            os.close(journal_descriptor)
        _fsync_directory(settings.agent_journal.path.parent)
        journal = EventJournal(
            settings.agent_journal.path,
            retained_files=settings.agent_journal.retained_files,
            codec=codecs.journal,
        )
        journal.reconcile(snapshot)
        generation_id = str(uuid4())
        _write_marker(settings, generation_id)
        verify_encrypted_generation(settings)
        return generation_id
    except Exception:
        if not marker.exists():
            for path in _authoritative_paths(settings):
                path.unlink(missing_ok=True)
        raise


def seal_encrypted_generation(settings: Settings) -> str:
    """Seal a fully migrated generation after verifying its continuity."""

    generation_id = str(uuid4())
    _verify_files(settings)
    _write_marker(settings, generation_id)
    verify_encrypted_generation(settings)
    return generation_id


def require_encrypted_generation(settings: Settings) -> None:
    if (
        settings.project.environment == ProjectEnvironment.PRODUCTION
        and settings.at_rest.live.enabled
    ):
        verify_encrypted_generation(settings)


def verify_encrypted_generation(settings: Settings) -> None:
    marker = settings.at_rest.live.generation_marker
    try:
        payload = json.loads(marker.read_bytes())
    except FileNotFoundError as exc:
        raise EncryptionError(
            "encrypted state is not initialized; run state-encryption-init"
        ) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EncryptionError("encrypted state generation marker is invalid") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "generation_id",
        "created_at",
    }:
        raise EncryptionError("encrypted state generation marker schema is invalid")
    if payload["schema_version"] != _MARKER_VERSION:
        raise EncryptionError("encrypted state generation marker is unsupported")
    _verify_files(settings)


def _verify_files(settings: Settings) -> None:
    missing = [
        path.name for path in _authoritative_paths(settings) if not path.is_file()
    ]
    if missing:
        raise EncryptionError("sealed encrypted state generation is incomplete")
    codecs = build_live_codecs(settings)
    snapshot = AgentStateStore(settings.agent_state.path, codec=codecs.snapshot).load(
        settings.emotion.baseline_surprisal
    )
    wal = StateWAL(settings.agent_state_wal.path, codec=codecs.wal).reconstruct()
    if (
        wal.sequence != snapshot.last_processed_event_sequence
        or wal.snapshot_hash != hash_snapshot(snapshot)
    ):
        raise EncryptionError("sealed snapshot and WAL are inconsistent")
    EventJournal(
        settings.agent_journal.path,
        retained_files=settings.agent_journal.retained_files,
        codec=codecs.journal,
    ).verify()


def _write_marker(settings: Settings, generation_id: str) -> None:
    path = settings.at_rest.live.generation_marker
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        payload = {
            "schema_version": _MARKER_VERSION,
            "generation_id": generation_id,
            "created_at": datetime.now(UTC).isoformat(),
        }
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), 0o600)
            output.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                    "ascii"
                )
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _authoritative_paths(settings: Settings) -> tuple[Path, Path, Path]:
    return (
        settings.agent_state.path,
        settings.agent_state_wal.path,
        settings.agent_journal.path,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
