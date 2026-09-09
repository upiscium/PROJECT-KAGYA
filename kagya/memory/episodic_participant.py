"""Durable staged participant for coordinated episodic Memory writes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import UUID, uuid4, uuid5

from kagya.memory.dual_memory_system import DualMemorySystem
from kagya.memory.memory_schema import EpisodicMemoryRecord, MemoryRecordType
from kagya.runtime.event_journal import (
    AbortOutcome,
    ParticipantCapability,
    ParticipantOutcome,
    StartupParticipantOutcome,
)
from kagya.runtime.transaction_coordinator import (
    ParticipantDivergedError,
    ParticipantUnavailableError,
    TransactionBinding,
    UnsupportedParticipantReconciliationError,
    validate_transaction_binding,
)


MEMORY_EPISODIC_PARTICIPANT_ID = "memory.episodic"
_OPERATION_SCHEMA_VERSION = 1
_PENDING_SCHEMA_VERSION = 1
_OPERATION_HASH_DOMAIN = b"PROJECT-KAGYA:R07:MEMORY-EPISODIC:V1\x00"
_EPISODE_ID_NAMESPACE = UUID("f0ced3ab-ad3f-5acb-b190-edea7fff6aff")
_STAGING_DIRECTORY = ".r07-episodic-pending"
_MAX_PENDING_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class EpisodicWrite:
    user_input: str
    response: str
    loss: float
    emotion_valence: float
    emotion_arousal: float
    record_type: MemoryRecordType
    created_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.user_input, str) or not isinstance(self.response, str):
            raise ValueError("Episodic text is invalid")
        for value in (self.loss, self.emotion_valence, self.emotion_arousal):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Episodic numeric value is invalid")
        if self.record_type is not MemoryRecordType.EPISODIC_LOG:
            raise ValueError("Coordinated record type is unsupported")
        try:
            timestamp = datetime.fromisoformat(self.created_at)
        except (TypeError, ValueError):
            raise ValueError("Episodic timestamp is invalid") from None
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("Episodic timestamp must be timezone-aware")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "schema_version": _OPERATION_SCHEMA_VERSION,
            "user_input": self.user_input,
            "response": self.response,
            "loss": float(self.loss),
            "emotion_valence": float(self.emotion_valence),
            "emotion_arousal": float(self.emotion_arousal),
            "record_type": self.record_type.value,
            "created_at": self.created_at,
        }


def episodic_operation_digest(operation: EpisodicWrite) -> str:
    canonical = _canonical_json_bytes(operation.canonical_dict())
    return hashlib.sha256(_OPERATION_HASH_DOMAIN + canonical).hexdigest()


class MemoryEpisodicParticipant:
    """Memory-owned prepare/finalize/abort primitives for one episodic write."""

    participant_id = MEMORY_EPISODIC_PARTICIPANT_ID
    capabilities = (
        ParticipantCapability.ABORT,
        ParticipantCapability.IDEMPOTENT_FINALIZE,
        ParticipantCapability.INSPECT_RECONCILE,
        ParticipantCapability.PREPARE,
    )

    def __init__(self, memory: DualMemorySystem, operation: EpisodicWrite) -> None:
        self.memory = memory
        self.operation = operation
        self.operation_digest = episodic_operation_digest(operation)
        self._lock = RLock()

    @classmethod
    def from_pending(
        cls,
        memory: DualMemorySystem,
        transaction_id: str,
        participant_id: str,
        operation_digest: str,
    ) -> MemoryEpisodicParticipant:
        """Rebuild the domain participant from Memory-owned pending evidence."""

        try:
            parsed = UUID(transaction_id)
        except (TypeError, ValueError):
            raise ParticipantDivergedError("Transaction identity is invalid") from None
        if (
            str(parsed) != transaction_id
            or participant_id != MEMORY_EPISODIC_PARTICIPANT_ID
            or re.fullmatch(r"[0-9a-f]{64}", operation_digest) is None
        ):
            raise ParticipantDivergedError("Pending Memory identity is invalid")
        loaded = cls._load_memory_optional(memory, f"{transaction_id}.json")
        if loaded is None:
            episode_id = _episode_id(transaction_id, participant_id, operation_digest)
            committed = memory.get_episodic_record(episode_id)
            if committed is None:
                raise UnsupportedParticipantReconciliationError(
                    "Memory operation evidence is absent"
                )
            operation = _operation_from_record(committed)
        else:
            operation = _operation_from_dict(loaded.get("operation"))
        participant = cls(memory, operation)
        if participant.operation_digest != operation_digest:
            raise ParticipantDivergedError("Pending Memory digest conflicts")
        if loaded is None:
            assert committed is not None
            if not participant._record_matches(committed, episode_id):
                raise ParticipantDivergedError("Committed Memory record conflicts")
            return participant
        expected = {
            "schema_version": _PENDING_SCHEMA_VERSION,
            "transaction_id": transaction_id,
            "participant_id": participant_id,
            "operation_digest": operation_digest,
            "episode_id": participant.episode_id(transaction_id),
            "operation": operation.canonical_dict(),
        }
        if loaded != expected:
            raise ParticipantDivergedError("Pending Memory artifact conflicts")
        return participant

    def episode_id(self, transaction_id: str) -> str:
        try:
            parsed = UUID(transaction_id)
        except (TypeError, ValueError):
            raise ParticipantDivergedError("Transaction identity is invalid") from None
        if str(parsed) != transaction_id:
            raise ParticipantDivergedError("Transaction identity is invalid")
        return _episode_id(transaction_id, self.participant_id, self.operation_digest)

    def pending_path(self, binding: TransactionBinding) -> Path:
        self._validate_binding(binding)
        return self._staging_directory() / f"{binding.transaction_id}.json"

    def prepare(self, binding: TransactionBinding) -> None:
        with self._lock:
            expected = self._artifact(binding)
            path = self.pending_path(binding)
            existing = self._load_optional(path)
            committed = self.memory.get_episodic_record(str(expected["episode_id"]))
            if committed is not None:
                if not self._record_matches(committed, str(expected["episode_id"])):
                    raise ParticipantDivergedError("Committed Memory record conflicts")
                if existing is not None and existing != expected:
                    raise ParticipantDivergedError("Pending Memory record conflicts")
                return
            if existing is not None:
                if existing != expected:
                    raise ParticipantDivergedError("Pending Memory record conflicts")
                return
            self._write_artifact(path, expected)

    def finalize(self, binding: TransactionBinding) -> ParticipantOutcome:
        with self._lock:
            expected = self._artifact(binding)
            path = self.pending_path(binding)
            existing = self._load_optional(path)
            committed = self.memory.get_episodic_record(str(expected["episode_id"]))
            if committed is not None:
                if not self._record_matches(committed, str(expected["episode_id"])):
                    raise ParticipantDivergedError("Committed Memory record conflicts")
                if existing is not None:
                    if existing != expected:
                        raise ParticipantDivergedError("Pending Memory record conflicts")
                    self._remove_artifact(path)
                return ParticipantOutcome.ALREADY_CONSISTENT
            if existing != expected:
                if existing is None:
                    raise ParticipantUnavailableError("Pending Memory record is absent")
                raise ParticipantDivergedError("Pending Memory record conflicts")
            self.memory.publish_coordinated_episodic(
                str(expected["episode_id"]),
                self.operation.user_input,
                self.operation.response,
                loss=self.operation.loss,
                emotion_valence=self.operation.emotion_valence,
                emotion_arousal=self.operation.emotion_arousal,
                record_type=self.operation.record_type,
                created_at=self.operation.created_at,
            )
            committed = self.memory.get_episodic_record(str(expected["episode_id"]))
            if committed is None or not self._record_matches(
                committed, str(expected["episode_id"])
            ):
                raise ParticipantUnavailableError("Memory publication is unverified")
            self._remove_artifact(path)
            return ParticipantOutcome.FINALIZED

    def abort(self, binding: TransactionBinding) -> AbortOutcome:
        with self._lock:
            expected = self._artifact(binding)
            path = self.pending_path(binding)
            existing = self._load_optional(path)
            if existing is None:
                return AbortOutcome.ALREADY_ABSENT
            if existing != expected:
                raise ParticipantDivergedError("Pending Memory record conflicts")
            self._remove_artifact(path)
            return AbortOutcome.ABORTED

    def inspect_reconciliation(
        self, binding: TransactionBinding
    ) -> StartupParticipantOutcome:
        with self._lock:
            expected = self._artifact(binding)
            committed = self.memory.get_episodic_record(str(expected["episode_id"]))
            if committed is not None:
                if not self._record_matches(committed, str(expected["episode_id"])):
                    raise ParticipantDivergedError("Committed Memory record conflicts")
                return StartupParticipantOutcome.VERIFIED_CONSISTENT
            pending = self._load_optional(self.pending_path(binding))
            if pending == expected:
                raise ParticipantUnavailableError("Memory roll-forward is required")
            if pending is None:
                raise UnsupportedParticipantReconciliationError(
                    "Memory operation evidence is absent"
                )
            raise ParticipantDivergedError("Pending Memory record conflicts")

    def reconcile(self, binding: TransactionBinding) -> StartupParticipantOutcome:
        expected = self._artifact(binding)
        committed = self.memory.get_episodic_record(str(expected["episode_id"]))
        if committed is not None:
            if not self._record_matches(committed, str(expected["episode_id"])):
                raise ParticipantDivergedError("Committed Memory record conflicts")
            pending = self._load_optional(self.pending_path(binding))
            if pending is not None:
                if pending != expected:
                    raise ParticipantDivergedError("Pending Memory record conflicts")
                self._remove_artifact(self.pending_path(binding))
            return StartupParticipantOutcome.VERIFIED_CONSISTENT
        self.finalize(binding)
        return StartupParticipantOutcome.ROLLED_FORWARD

    def _validate_binding(self, binding: TransactionBinding) -> None:
        if (
            not validate_transaction_binding(binding)
            or binding.participant_id != self.participant_id
            or binding.operation_digest != self.operation_digest
        ):
            raise ParticipantDivergedError("Transaction binding is invalid")

    def _artifact(self, binding: TransactionBinding) -> dict[str, object]:
        self._validate_binding(binding)
        return {
            "schema_version": _PENDING_SCHEMA_VERSION,
            "transaction_id": binding.transaction_id,
            "participant_id": self.participant_id,
            "operation_digest": self.operation_digest,
            "episode_id": self.episode_id(binding.transaction_id),
            "operation": self.operation.canonical_dict(),
        }

    def _record_matches(
        self, record: EpisodicMemoryRecord, expected_episode_id: str
    ) -> bool:
        return (
            record.id == expected_episode_id
            and record.user_input == self.operation.user_input
            and record.response == self.operation.response
            and record.loss == float(self.operation.loss)
            and record.emotion_valence == float(self.operation.emotion_valence)
            and record.emotion_arousal == float(self.operation.emotion_arousal)
            and record.record_type is self.operation.record_type
            and not record.archived
            and record.created_at == self.operation.created_at
            and record.metadata == {}
        )

    def _staging_directory(self) -> Path:
        directory = self.memory.settings.memory.persist_directory / _STAGING_DIRECTORY
        descriptor = -1
        try:
            descriptor = self._open_staging_descriptor(self.memory, create=True)
        except OSError:
            raise ParticipantUnavailableError("Memory staging is unavailable") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        return directory

    def _load_optional(self, path: Path) -> dict[str, Any] | None:
        return self._load_memory_optional(self.memory, path.name)

    @classmethod
    def _load_memory_optional(
        cls, memory: DualMemorySystem, name: str
    ) -> dict[str, Any] | None:
        parent_descriptor = -1
        descriptor = -1
        try:
            parent_descriptor = cls._open_staging_descriptor(memory, create=False)
            if parent_descriptor < 0:
                return None
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError:
            return None
        except OSError:
            raise ParticipantUnavailableError("Memory staging is unavailable") from None
        try:
            with os.fdopen(descriptor, "rb") as source:
                descriptor = -1
                status = os.fstat(source.fileno())
                if (
                    not stat.S_ISREG(status.st_mode)
                    or status.st_uid != os.geteuid()
                    or stat.S_IMODE(status.st_mode) != 0o600
                ):
                    raise OSError
                payload = source.read(_MAX_PENDING_BYTES + 1)
            if len(payload) > _MAX_PENDING_BYTES:
                raise ValueError
            loaded = json.loads(payload)
            if not isinstance(loaded, dict):
                raise ValueError
            return loaded
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            raise ParticipantDivergedError("Pending Memory artifact is invalid") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    def _write_artifact(self, path: Path, artifact: dict[str, object]) -> None:
        payload = _canonical_json_bytes(artifact) + b"\n"
        parent_descriptor = -1
        descriptor = -1
        temporary = f".pending-{uuid4()}.tmp"
        try:
            parent_descriptor = self._open_staging_descriptor(self.memory, create=True)
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            with os.fdopen(descriptor, "wb") as target:
                descriptor = -1
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            os.replace(
                temporary,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            temporary = ""
            os.fsync(parent_descriptor)
        except OSError:
            raise ParticipantUnavailableError("Memory staging is unavailable") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary:
                try:
                    os.unlink(temporary, dir_fd=parent_descriptor)
                except OSError:
                    pass
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    def _remove_artifact(self, path: Path) -> None:
        parent_descriptor = -1
        try:
            parent_descriptor = self._open_staging_descriptor(self.memory, create=False)
            if parent_descriptor < 0:
                return
            os.unlink(path.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        except FileNotFoundError:
            return
        except OSError:
            raise ParticipantUnavailableError("Memory staging is unavailable") from None
        finally:
            if parent_descriptor >= 0:
                os.close(parent_descriptor)

    @staticmethod
    def _open_staging_descriptor(
        memory: DualMemorySystem, *, create: bool
    ) -> int:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        absolute = memory.settings.memory.persist_directory.absolute()
        descriptor = os.open(absolute.anchor, flags)
        try:
            for component in absolute.parts[1:]:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            status = os.fstat(descriptor)
            if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.geteuid():
                raise OSError
            try:
                staging_descriptor = os.open(
                    _STAGING_DIRECTORY, flags, dir_fd=descriptor
                )
            except FileNotFoundError:
                if not create:
                    return -1
                os.mkdir(_STAGING_DIRECTORY, 0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                staging_descriptor = os.open(
                    _STAGING_DIRECTORY, flags, dir_fd=descriptor
                )
            staging_status = os.fstat(staging_descriptor)
            if (
                not stat.S_ISDIR(staging_status.st_mode)
                or staging_status.st_uid != os.geteuid()
            ):
                os.close(staging_descriptor)
                raise OSError
            os.fchmod(staging_descriptor, 0o700)
            return staging_descriptor
        except Exception:
            os.close(descriptor)
            raise
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _episode_id(
    transaction_id: str, participant_id: str, operation_digest: str
) -> str:
    canonical = json.dumps(
        [transaction_id, participant_id, operation_digest],
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    return f"episode-{uuid5(_EPISODE_ID_NAMESPACE, canonical)}"


def _operation_from_record(record: EpisodicMemoryRecord) -> EpisodicWrite:
    if record.metadata or record.archived:
        raise ParticipantDivergedError("Committed Memory record conflicts")
    return EpisodicWrite(
        user_input=record.user_input,
        response=record.response,
        loss=record.loss,
        emotion_valence=record.emotion_valence,
        emotion_arousal=record.emotion_arousal,
        record_type=record.record_type,
        created_at=record.created_at,
    )


def _operation_from_dict(value: object) -> EpisodicWrite:
    expected_keys = {
        "schema_version",
        "user_input",
        "response",
        "loss",
        "emotion_valence",
        "emotion_arousal",
        "record_type",
        "created_at",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ParticipantDivergedError("Pending Memory operation is invalid")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != _OPERATION_SCHEMA_VERSION
        or not isinstance(value["user_input"], str)
        or not isinstance(value["response"], str)
        or not isinstance(value["loss"], float)
        or not isinstance(value["emotion_valence"], float)
        or not isinstance(value["emotion_arousal"], float)
        or not isinstance(value["record_type"], str)
        or not isinstance(value["created_at"], str)
    ):
        raise ParticipantDivergedError("Pending Memory operation is invalid")
    try:
        return EpisodicWrite(
            user_input=value["user_input"],
            response=value["response"],
            loss=value["loss"],
            emotion_valence=value["emotion_valence"],
            emotion_arousal=value["emotion_arousal"],
            record_type=MemoryRecordType(value["record_type"]),
            created_at=value["created_at"],
        )
    except (TypeError, ValueError):
        raise ParticipantDivergedError("Pending Memory operation is invalid") from None
