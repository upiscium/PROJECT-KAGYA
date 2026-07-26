"""Streaming encrypted backups and fail-closed isolated restoration."""

from __future__ import annotations

from base64 import b64decode, b64encode
import builtins
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, BinaryIO, Callable, Iterator, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from kagya.artifact_provenance import build_adapter_artifact_manifest
from kagya.config.schema import ProjectEnvironment, Settings
from kagya.runtime.agent_state import AgentStateStore
from kagya.runtime.event_journal import EventJournal, hash_snapshot
from kagya.runtime.state_wal import StateWAL
from kagya.security.crypto import (
    EncryptedCodec,
    EncryptionError,
    KeyRing,
    load_key_ring,
)
from kagya.security.live import build_live_codecs


BACKUP_FORMAT_VERSION = 1
_CHUNK_SIZE = 1024 * 1024
_RESTORE_MARKER = ".restore-in-progress"


class BackupError(RuntimeError):
    """Raised when a backup or restore cannot be trusted."""


class BackupFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root: str = Field(pattern=r"^[a-z0-9_]+$")
    relative_path: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_artifact: bool = False
    included: bool = True


class BackupManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    backup_id: str
    created_at: datetime
    source_revision: str
    model_id: str
    model_revision: str
    processor_revision: str
    adapter_revision: str | None = None
    base_backup_id: str | None = None
    base_manifest_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    roots: dict[str, bool]
    files: list[BackupFile]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.model_dump(mode="json"))).hexdigest()


class BackupStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format_version: Literal[1] = 1
    backup_id: str
    created_at: datetime
    encrypted_size: int = Field(ge=0)
    encrypted_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    key_ids: list[str]


class RestorePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backup_id: str
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    file_count: int = Field(ge=0)
    total_plaintext_bytes: int = Field(ge=0)
    base_backup_id: str | None = None
    model_revision: str
    adapter_revision: str | None = None


@dataclass(frozen=True)
class _Source:
    label: str
    path: Path
    directory: bool
    adapter: bool = False
    allowed_names: frozenset[str] | None = None


class BackupManager:
    def __init__(
        self,
        settings: Settings,
        *,
        failure_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.directory = settings.at_rest.backup.directory
        self._failure_injector = failure_injector

    def create(self, *, base_backup_id: str | None = None) -> RestorePreview:
        backup_ring, adapter_ring = self._rings()
        backup_id = str(uuid4())
        created_at = datetime.now(UTC)
        base_manifest: BackupManifest | None = None
        if base_backup_id is not None:
            base_manifest = self._read_manifest(self._bundle_path(base_backup_id))[0]
            if base_manifest.backup_id != base_backup_id:
                raise BackupError("incremental base backup ID does not match")
        files = self._inventory(base_manifest)
        self._inject("backup_after_inventory")
        manifest = BackupManifest(
            backup_id=backup_id,
            created_at=created_at,
            source_revision=_source_revision(),
            model_id=self.settings.model.primary_id,
            model_revision=self.settings.model.revision,
            processor_revision=self.settings.model.processor_revision,
            adapter_revision=self._active_adapter_revision(),
            base_backup_id=None if base_manifest is None else base_manifest.backup_id,
            base_manifest_hash=None if base_manifest is None else base_manifest.sha256,
            roots={
                source.label: _source_exists(source) for source in self._source_list()
            },
            files=files,
        )
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        final_path = self._bundle_path(backup_id)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{backup_id}.", suffix=".tmp", dir=self.directory
        )
        temporary = Path(temporary_name)
        codec = EncryptedCodec(
            enabled=True,
            purpose="backup",
            context="bundle-record",
            key_ring=backup_ring,
        )
        adapter_codec = EncryptedCodec(
            enabled=True,
            purpose="adapter-artifact",
            context="backup-chunk",
            key_ring=adapter_ring,
        )
        published = False
        try:
            with os.fdopen(descriptor, "wb") as output:
                os.fchmod(output.fileno(), 0o600)
                header = {
                    "format": "kagya-encrypted-backup",
                    "format_version": BACKUP_FORMAT_VERSION,
                    "backup_id": backup_id,
                    "key_ids": [
                        backup_ring.current_key_id,
                        adapter_ring.current_key_id,
                    ],
                }
                header_bytes = _canonical(header)
                header_hash = hashlib.sha256(header_bytes).hexdigest()
                output.write(header_bytes + b"\n")
                record_index = 0
                source_by_label = {item.label: item for item in self._source_list()}
                for entry in files:
                    if not entry.included:
                        continue
                    source = source_by_label[entry.root]
                    path = _source_file(source, entry.relative_path)
                    wrote = False
                    with _open_regular_nofollow(path) as stream:
                        initial_stat = os.fstat(stream.fileno())
                        offset = 0
                        streamed_hash = hashlib.sha256()
                        while chunk := stream.read(_CHUNK_SIZE):
                            self._inject("backup_stream_chunk")
                            streamed_hash.update(chunk)
                            payload_data = chunk
                            if entry.adapter_artifact:
                                payload_data = adapter_codec.encode(
                                    chunk,
                                    metadata={
                                        "backup_id": backup_id,
                                        "record_index": record_index,
                                    },
                                )
                            payload = {
                                "type": "file_chunk",
                                "root": entry.root,
                                "relative_path": entry.relative_path,
                                "offset": offset,
                                "adapter_artifact": entry.adapter_artifact,
                                "data": b64encode(payload_data).decode("ascii"),
                            }
                            _write_record(
                                output,
                                codec,
                                record_index,
                                payload,
                                backup_id=backup_id,
                                header_hash=header_hash,
                            )
                            offset += len(chunk)
                            record_index += 1
                            wrote = True
                        final_stat = os.fstat(stream.fileno())
                    if (
                        offset != entry.size
                        or streamed_hash.hexdigest() != entry.sha256
                        or _changed_while_open(initial_stat, final_stat)
                    ):
                        raise BackupError("backup source changed while streaming")
                    if not wrote:
                        payload = {
                            "type": "file_chunk",
                            "root": entry.root,
                            "relative_path": entry.relative_path,
                            "offset": 0,
                            "adapter_artifact": entry.adapter_artifact,
                            "data": "",
                        }
                        _write_record(
                            output,
                            codec,
                            record_index,
                            payload,
                            backup_id=backup_id,
                            header_hash=header_hash,
                        )
                        record_index += 1
                _write_record(
                    output,
                    codec,
                    record_index,
                    {"type": "manifest", "manifest": manifest.model_dump(mode="json")},
                    backup_id=backup_id,
                    header_hash=header_hash,
                )
                output.flush()
                os.fsync(output.fileno())
            self._verify_created_bundle(temporary, manifest)
            self._inject("backup_before_replace")
            os.replace(temporary, final_path)
            _fsync_directory(self.directory)
            status = _bundle_status(final_path, header, created_at)
            _atomic_json(self._sidecar_path(backup_id), status.model_dump(mode="json"))
            published = True
            self.enforce_retention()
            return _preview(manifest)
        except Exception:
            temporary.unlink(missing_ok=True)
            if not published:
                final_path.unlink(missing_ok=True)
                self._sidecar_path(backup_id).unlink(missing_ok=True)
            raise

    def list(self, limit: int = 50) -> builtins.list[BackupStatus]:
        if limit < 1 or limit > 100:
            raise BackupError("backup status limit must be between 1 and 100")
        statuses: builtins.list[BackupStatus] = []
        if not self.directory.exists():
            return statuses
        valid_ids = set(self._restorable_manifests())
        for path in self.directory.glob("*.status.json"):
            try:
                status = BackupStatus.model_validate_json(path.read_bytes())
                if status.backup_id in valid_ids:
                    statuses.append(status)
            except (OSError, ValueError):
                continue
        statuses.sort(key=lambda item: item.created_at, reverse=True)
        return statuses[:limit]

    def preview(self, backup_id: str) -> RestorePreview:
        manifest, _header = self._read_manifest(self._bundle_path(backup_id))
        return _preview(manifest)

    def scheduled_base_backup_id(self) -> str | None:
        manifests = self._restorable_manifests()
        if not manifests:
            return None
        latest = max(manifests.values(), key=lambda item: item.created_at)
        chain_length = 1
        current = latest
        while current.base_backup_id is not None:
            chain_length += 1
            current = manifests[current.base_backup_id]
        if chain_length >= self.settings.at_rest.backup.incremental_full_every:
            return None
        return latest.backup_id

    def verify(self, backup_id: str) -> RestorePreview:
        self._require_staging_attestation()
        staging = self._staging_directory()
        with tempfile.TemporaryDirectory(
            prefix="kagya-restore-verify-", dir=staging
        ) as temporary:
            root = Path(temporary)
            os.chmod(root, 0o700)
            manifest = self._extract_chain(backup_id, root)
            self._verify_staged(root, manifest)
            return _preview(manifest)

    def restore(
        self,
        backup_id: str,
        *,
        expected_manifest_hash: str,
        prepare: Callable[[Path], None] | None = None,
        after_publish: Callable[[], None] | None = None,
        after_rollback: Callable[[], None] | None = None,
    ) -> RestorePreview:
        self._require_staging_attestation()
        staging = self._staging_directory()
        temporary = Path(tempfile.mkdtemp(prefix="kagya-restore-", dir=staging))
        os.chmod(temporary, 0o700)
        try:
            manifest = self._extract_chain(backup_id, temporary)
            if manifest.backup_id != backup_id:
                raise BackupError("restore backup ID does not match")
            if manifest.sha256 != expected_manifest_hash:
                raise BackupError("restore manifest hash does not match expectation")
            self._verify_staged(temporary, manifest)
            if prepare is not None:
                prepare(temporary)
            self._inject("restore_before_swap")
            self._commit_staged(
                temporary,
                manifest,
                after_publish=after_publish,
                after_rollback=after_rollback,
            )
            return _preview(manifest)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def rotate(self, backup_id: str) -> RestorePreview:
        """Re-encrypt an allowed old generation as a new full current generation."""

        self._require_staging_attestation()
        staging = self._staging_directory()
        with tempfile.TemporaryDirectory(
            prefix="kagya-rotate-", dir=staging
        ) as temporary:
            root = Path(temporary)
            os.chmod(root, 0o700)
            manifest = self._extract_chain(backup_id, root)
            self._verify_staged(root, manifest)
            return self._create_from_staged(root)

    def enforce_retention(self) -> None:
        if not self.directory.exists():
            return
        manifests = self._restorable_manifests()
        chains: dict[str, set[str]] = {}
        newest: dict[str, datetime] = {}
        for backup_id, manifest in manifests.items():
            root_id = self._chain_root(backup_id, manifests)
            chains.setdefault(root_id, set()).add(backup_id)
            newest[root_id] = max(
                newest.get(root_id, manifest.created_at), manifest.created_at
            )
        ordered_roots = sorted(chains, key=lambda item: newest[item], reverse=True)
        retained_roots = set(
            ordered_roots[: self.settings.at_rest.backup.retention_count]
        )
        # A chain is removed as one unit, and only when a newer verified full chain exists.
        for root_id in ordered_roots:
            if root_id in retained_roots or not any(
                newest[other] > newest[root_id] for other in retained_roots
            ):
                continue
            for backup_id in chains[root_id]:
                self._bundle_path(backup_id).unlink(missing_ok=True)
                self._sidecar_path(backup_id).unlink(missing_ok=True)
        # Orphaned incrementals are never advertised and can only be removed by
        # retention once at least one independently restorable full chain exists.
        if retained_roots:
            known = set(manifests)
            for path in self.directory.glob("*.kgb"):
                if path.stem not in known:
                    path.unlink(missing_ok=True)
                    path.with_suffix(".status.json").unlink(missing_ok=True)
        _fsync_directory(self.directory)

    def _restorable_manifests(self) -> dict[str, BackupManifest]:
        candidates: dict[str, BackupManifest] = {}
        if not self.directory.exists():
            return candidates
        for path in self.directory.glob("*.kgb"):
            try:
                manifest, _header = self._read_manifest(path)
                if path == self._bundle_path(manifest.backup_id):
                    candidates[manifest.backup_id] = manifest
            except (BackupError, EncryptionError, OSError, ValueError):
                continue
        valid: dict[str, BackupManifest] = {}
        visiting: set[str] = set()

        def accept(backup_id: str) -> bool:
            if backup_id in valid:
                return True
            manifest = candidates.get(backup_id)
            if manifest is None or backup_id in visiting:
                return False
            if manifest.base_backup_id is None:
                valid[backup_id] = manifest
                return True
            visiting.add(backup_id)
            base = candidates.get(manifest.base_backup_id)
            accepted = (
                base is not None
                and accept(base.backup_id)
                and base.sha256 == manifest.base_manifest_hash
            )
            visiting.discard(backup_id)
            if accepted:
                valid[backup_id] = manifest
            return accepted

        for backup_id in candidates:
            accept(backup_id)
        return valid

    def _chain_root(self, backup_id: str, manifests: dict[str, BackupManifest]) -> str:
        current = manifests[backup_id]
        while current.base_backup_id is not None:
            current = manifests[current.base_backup_id]
        return current.backup_id

    def _inventory(self, base: BackupManifest | None) -> builtins.list[BackupFile]:
        self._validate_inventory_sources()
        previous = {
            (item.root, item.relative_path): item.sha256
            for item in ([] if base is None else base.files)
        }
        files: builtins.list[BackupFile] = []
        seen_destinations: set[Path] = set()
        for source in self._source_list():
            resolved = source.path.resolve(strict=False)
            if resolved in seen_destinations:
                raise BackupError("configured backup roots overlap exactly")
            seen_destinations.add(resolved)
            for path, relative in _walk_source(source):
                digest, size = _hash_file(path)
                files.append(
                    BackupFile(
                        root=source.label,
                        relative_path=relative,
                        size=size,
                        sha256=digest,
                        adapter_artifact=source.adapter,
                        included=previous.get((source.label, relative)) != digest,
                    )
                )
        return sorted(files, key=lambda item: (item.root, item.relative_path))

    def _read_manifest(self, path: Path) -> tuple[BackupManifest, dict[str, Any]]:
        manifest: BackupManifest | None = None
        header: dict[str, Any] | None = None
        for payload, current_header in self._read_records(path):
            header = current_header
            if payload.get("type") == "manifest":
                if manifest is not None:
                    raise BackupError("backup contains multiple manifests")
                manifest = BackupManifest.model_validate(payload.get("manifest"))
        if header is None or manifest is None:
            raise BackupError("backup manifest is missing")
        if manifest.backup_id != header.get("backup_id"):
            raise BackupError("backup header and manifest IDs disagree")
        return manifest, header

    def _read_records(
        self, path: Path
    ) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        backup_ring, _adapter_ring = self._rings()
        codec = EncryptedCodec(
            enabled=True,
            purpose="backup",
            context="bundle-record",
            key_ring=backup_ring,
        )
        try:
            source = _open_regular_nofollow(path)
        except OSError as exc:
            raise BackupError("backup is unavailable") from exc
        with source:
            header_line = source.readline()
            try:
                header = json.loads(header_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BackupError("backup header is invalid") from exc
            if not isinstance(header, dict) or set(header) != {
                "format",
                "format_version",
                "backup_id",
                "key_ids",
            }:
                raise BackupError("backup header schema is invalid")
            if (
                header["format"] != "kagya-encrypted-backup"
                or header["format_version"] != BACKUP_FORMAT_VERSION
                or not isinstance(header["backup_id"], str)
                or not isinstance(header["key_ids"], list)
            ):
                raise BackupError("backup header is unsupported")
            header_hash = hashlib.sha256(header_line.rstrip(b"\n")).hexdigest()
            found_manifest = False
            for index, line in enumerate(source):
                if not line.endswith(b"\n"):
                    raise BackupError("backup record is truncated")
                if found_manifest:
                    raise BackupError("backup contains records after its manifest")
                try:
                    plaintext = codec.decode(
                        line.rstrip(b"\n"),
                        expected_metadata={
                            "record_index": index,
                            "backup_id": header["backup_id"],
                            "header_hash": header_hash,
                        },
                    )
                    payload = json.loads(plaintext)
                except (
                    EncryptionError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ) as exc:
                    raise BackupError("backup record authentication failed") from exc
                if not isinstance(payload, dict):
                    raise BackupError("backup record payload is invalid")
                found_manifest = payload.get("type") == "manifest"
                yield payload, header
            if not found_manifest:
                raise BackupError("backup is truncated before its manifest")

    def _extract_chain(self, backup_id: str, root: Path) -> BackupManifest:
        manifest, _header = self._read_manifest(self._bundle_path(backup_id))
        if manifest.base_backup_id is not None:
            base = self._extract_chain(manifest.base_backup_id, root)
            if base.sha256 != manifest.base_manifest_hash:
                raise BackupError("incremental base manifest hash is invalid")
        self._extract_one(self._bundle_path(backup_id), root, manifest)
        self._shape_staged(root, manifest)
        return manifest

    def _shape_staged(self, root: Path, manifest: BackupManifest) -> None:
        known_roots = {source.label for source in _sources(self.settings)}
        if set(manifest.roots) != known_roots:
            raise BackupError("backup authoritative root set is invalid")
        for label, exists in manifest.roots.items():
            label_root = root / label
            if exists:
                label_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            elif label_root.exists():
                shutil.rmtree(label_root)
        expected = {(item.root, item.relative_path) for item in manifest.files}
        for label_root in root.iterdir():
            if not label_root.is_dir():
                raise BackupError("isolated restore root contains an invalid entry")
            for path in sorted(label_root.rglob("*"), reverse=True):
                if (
                    path.is_file()
                    and (label_root.name, path.relative_to(label_root).as_posix())
                    not in expected
                ):
                    path.unlink()
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass

    def _verify_created_bundle(self, path: Path, manifest: BackupManifest) -> None:
        self._require_staging_attestation()
        staging = self._staging_directory()
        with tempfile.TemporaryDirectory(
            prefix="kagya-backup-self-verify-", dir=staging
        ) as temporary:
            root = Path(temporary)
            os.chmod(root, 0o700)
            if manifest.base_backup_id is not None:
                base = self._extract_chain(manifest.base_backup_id, root)
                if base.sha256 != manifest.base_manifest_hash:
                    raise BackupError("incremental base manifest hash is invalid")
            candidate, _header = self._read_manifest(path)
            if candidate != manifest:
                raise BackupError("temporary backup manifest changed")
            self._extract_one(path, root, manifest)
            self._shape_staged(root, manifest)
            self._verify_staged(root, manifest)

    def _extract_one(self, path: Path, root: Path, manifest: BackupManifest) -> None:
        _backup_ring, adapter_ring = self._rings()
        adapter_codec = EncryptedCodec(
            enabled=True,
            purpose="adapter-artifact",
            context="backup-chunk",
            key_ring=adapter_ring,
        )
        expected = {(item.root, item.relative_path): item for item in manifest.files}
        handles: dict[tuple[str, str], Any] = {}
        try:
            for index, (payload, _header) in enumerate(self._read_records(path)):
                if payload.get("type") == "manifest":
                    continue
                if (
                    set(payload)
                    != {
                        "type",
                        "root",
                        "relative_path",
                        "offset",
                        "adapter_artifact",
                        "data",
                    }
                    or payload.get("type") != "file_chunk"
                ):
                    raise BackupError("backup file record schema is invalid")
                key = (str(payload["root"]), str(payload["relative_path"]))
                entry = expected.get(key)
                if entry is None or not entry.included:
                    raise BackupError("backup contains an unmanifested file")
                destination = _staged_file(root, *key)
                handle = handles.get(key)
                if handle is None:
                    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    handle = destination.open("wb")
                    os.chmod(destination, 0o600)
                    handles[key] = handle
                if payload["offset"] != handle.tell():
                    raise BackupError("backup file chunk offset is invalid")
                try:
                    data = b64decode(payload["data"], validate=True)
                except (TypeError, ValueError) as exc:
                    raise BackupError("backup file chunk encoding is invalid") from exc
                if bool(payload["adapter_artifact"]) != entry.adapter_artifact:
                    raise BackupError("backup adapter classification is invalid")
                if entry.adapter_artifact and data:
                    try:
                        data = adapter_codec.decode(
                            data,
                            expected_metadata={
                                "backup_id": manifest.backup_id,
                                "record_index": index,
                            },
                        )
                    except EncryptionError as exc:
                        raise BackupError(
                            "adapter artifact authentication failed"
                        ) from exc
                handle.write(data)
        finally:
            for handle in handles.values():
                handle.close()
        for entry in manifest.files:
            destination = _staged_file(root, entry.root, entry.relative_path)
            if not entry.included and destination.exists():
                continue
            if not destination.exists():
                if entry.size == 0 and entry.included:
                    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    destination.touch(mode=0o600)
                else:
                    raise BackupError("manifested backup file is missing")
            digest, size = _hash_file(destination)
            if digest != entry.sha256 or size != entry.size:
                raise BackupError("restored file checksum or size mismatch")

    def _verify_staged(self, root: Path, manifest: BackupManifest) -> None:
        for entry in manifest.files:
            path = _staged_file(root, entry.root, entry.relative_path)
            digest, size = _hash_file(path)
            if digest != entry.sha256 or size != entry.size:
                raise BackupError("isolated restore checksum validation failed")
        by_root = {source.label: source for source in _sources(self.settings)}
        live = build_live_codecs(self.settings)
        state_source = by_root["agent_state"]
        state_path = _staged_file(root, "agent_state", state_source.path.name)
        if not state_path.exists():
            return
        snapshot = AgentStateStore(state_path, codec=live.snapshot).load(
            self.settings.emotion.baseline_surprisal
        )
        wal_source = by_root["state_wal"]
        wal_path = _staged_file(root, "state_wal", wal_source.path.name)
        journal_path = _staged_file(
            root, "journal", self.settings.agent_journal.path.name
        )
        if not wal_path.exists() or not journal_path.exists():
            raise BackupError("authoritative restore graph is incomplete")
        wal = StateWAL(wal_path, codec=live.wal).reconstruct()
        if wal.sequence != snapshot.last_processed_event_sequence or (
            wal.snapshot_hash != hash_snapshot(snapshot)
        ):
            raise BackupError("snapshot and WAL do not describe the same state")
        journal = EventJournal(
            journal_path,
            max_bytes=self.settings.agent_journal.max_bytes,
            retained_files=self.settings.agent_journal.retained_files,
            codec=live.journal,
        )
        records = journal.verify()
        if snapshot.last_processed_event_sequence and not any(
            record.snapshot_sequence == snapshot.last_processed_event_sequence
            and record.snapshot_hash == hash_snapshot(snapshot)
            for record in records
        ):
            raise BackupError("snapshot and Journal do not describe the same state")
        if manifest.model_id != self.settings.model.primary_id:
            raise BackupError("backup model identity does not match this runtime")
        if (
            manifest.model_revision != self.settings.model.revision
            or manifest.processor_revision != self.settings.model.processor_revision
        ):
            raise BackupError("backup model revisions do not match this runtime")
        current_source = _source_revision()
        if manifest.source_revision != current_source:
            raise BackupError("backup source revision does not match this runtime")
        registry_entry = next(
            (item for item in manifest.files if item.root == "adapter_registry"), None
        )
        if registry_entry is not None:
            registry_path = _staged_file(
                root, registry_entry.root, registry_entry.relative_path
            )
            registry_hash, _size = _hash_file(registry_path)
            if manifest.adapter_revision != registry_hash:
                raise BackupError("backup adapter registry revision is invalid")
            _validate_staged_adapter_hashes(root, registry_path, self.settings)

    def _commit_staged(
        self,
        root: Path,
        manifest: BackupManifest,
        *,
        after_publish: Callable[[], None] | None,
        after_rollback: Callable[[], None] | None,
    ) -> None:
        assert_no_incomplete_restore(self.settings)
        sources = {source.label: source for source in _sources(self.settings)}
        labels = set(manifest.roots)
        rollback = self.directory / "previous-generation"
        pending_rollback = self.directory / f".previous-{uuid4()}"
        pending_rollback.mkdir(mode=0o700)
        marker = self.directory / _RESTORE_MARKER
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "backup_id": manifest.backup_id,
            "pending_generation": pending_rollback.name,
            "phase": "forward",
            "completed_operations": 0,
            "previous_operations": [],
        }
        _atomic_json(marker, metadata)
        swapped: builtins.list[tuple[Path, Path | None]] = []
        try:
            for label in sorted(labels):
                source = sources[label]
                staged_root = root / label
                operations: builtins.list[tuple[Path | None, Path, str]]
                root_exists = manifest.roots[label]
                if not root_exists and source.directory and source.allowed_names:
                    operations = [
                        (None, source.path / name, f"{label}-{name}")
                        for name in sorted(source.allowed_names)
                    ]
                elif not root_exists:
                    operations = [(None, source.path, label)]
                elif source.directory and source.allowed_names is None:
                    operations = [(staged_root, source.path, label)]
                elif source.directory:
                    operations = [
                        (
                            staged_root / name
                            if (staged_root / name).exists()
                            else None,
                            source.path / name,
                            f"{label}-{name.replace('/', '_')}",
                        )
                        for name in sorted(source.allowed_names or ())
                    ]
                else:
                    operations = [(staged_root / source.path.name, source.path, label)]
                for staged, destination, rollback_name in operations:
                    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    previous: Path | None = None
                    if destination.exists():
                        previous = pending_rollback / rollback_name
                        previous.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                        self._restore_replace(destination, previous, "forward")
                    swapped.append((destination, previous))
                    metadata["completed_operations"] = len(swapped)
                    metadata["previous_operations"].append(previous is not None)
                    _atomic_json(marker, metadata)
                    if staged is not None:
                        self._restore_replace(staged, destination, "forward")
                    self._restore_fsync(destination.parent, "forward")
            self._inject("restore_after_swap")
            if after_publish is not None:
                after_publish()
            self._verify_live_generation()
            if rollback.exists():
                shutil.rmtree(rollback)
            self._restore_replace(pending_rollback, rollback, "forward")
            self._restore_fsync(self.directory, "forward")
            marker.unlink(missing_ok=True)
            _fsync_directory(self.directory)
        except Exception as forward_error:
            metadata["phase"] = "rollback"
            _atomic_json(marker, metadata)
            try:
                for index, (destination, previous) in enumerate(reversed(swapped)):
                    if destination.exists():
                        failed = pending_rollback / f"failed-{index}"
                        self._restore_replace(destination, failed, "rollback")
                    if previous is not None and previous.exists():
                        self._restore_replace(previous, destination, "rollback")
                    self._restore_fsync(destination.parent, "rollback")
                self._verify_live_generation()
                if after_rollback is not None:
                    after_rollback()
                shutil.rmtree(pending_rollback)
                marker.unlink()
                _fsync_directory(self.directory)
            except Exception as rollback_error:
                metadata["phase"] = "rollback_failed"
                _atomic_json(marker, metadata)
                raise BackupError(
                    "restore rollback failed; recovery marker was preserved"
                ) from rollback_error
            raise forward_error

    def _restore_replace(self, source: Path, destination: Path, phase: str) -> None:
        self._inject(f"restore_{phase}_replace")
        os.replace(source, destination)

    def _restore_fsync(self, path: Path, phase: str) -> None:
        self._inject(f"restore_{phase}_fsync")
        _fsync_directory(path)

    def _verify_live_generation(self) -> None:
        codecs = build_live_codecs(self.settings)
        snapshot = AgentStateStore(
            self.settings.agent_state.path, codec=codecs.snapshot
        ).load(self.settings.emotion.baseline_surprisal)
        wal = StateWAL(
            self.settings.agent_state_wal.path, codec=codecs.wal
        ).reconstruct()
        if wal.sequence != snapshot.last_processed_event_sequence or (
            wal.snapshot_hash != hash_snapshot(snapshot)
        ):
            raise BackupError("live snapshot and WAL are inconsistent after restore")
        EventJournal(
            self.settings.agent_journal.path,
            retained_files=self.settings.agent_journal.retained_files,
            codec=codecs.journal,
        ).verify()

    def recovery_status(self) -> dict[str, Any]:
        marker = self.directory / _RESTORE_MARKER
        if not marker.exists():
            return {"status": "clean"}
        try:
            payload = json.loads(marker.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackupError("restore recovery marker is invalid") from exc
        allowed = {
            "schema_version",
            "backup_id",
            "pending_generation",
            "phase",
            "completed_operations",
            "previous_operations",
        }
        if not isinstance(payload, dict) or set(payload) != allowed:
            raise BackupError("restore recovery marker schema is invalid")
        return {
            "status": "recovery_required",
            "backup_id": payload["backup_id"],
            "phase": payload["phase"],
            "completed_operations": payload["completed_operations"],
        }

    def recover(self) -> dict[str, Any]:
        marker = self.directory / _RESTORE_MARKER
        if not marker.exists():
            return {"status": "clean"}
        try:
            payload = json.loads(marker.read_bytes())
            backup_id = str(payload["backup_id"])
            pending_name = str(payload["pending_generation"])
            completed = int(payload["completed_operations"])
            previous_flags = payload["previous_operations"]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise BackupError("restore recovery marker is invalid") from exc
        if (
            not pending_name.startswith(".previous-")
            or Path(pending_name).name != pending_name
            or not isinstance(previous_flags, list)
            or len(previous_flags) != completed
            or not all(isinstance(item, bool) for item in previous_flags)
        ):
            raise BackupError("restore recovery marker schema is invalid")
        manifest, _header = self._read_manifest(self._bundle_path(backup_id))
        operations = self._manifest_operation_destinations(manifest)
        if completed < 0 or completed > len(operations):
            raise BackupError("restore recovery operation count is invalid")
        pending = self.directory / pending_name
        if not pending.is_dir():
            raise BackupError("restore recovery generation is unavailable")
        payload["phase"] = "rollback"
        _atomic_json(marker, payload)
        try:
            for index in range(completed - 1, -1, -1):
                destination, rollback_name = operations[index]
                previous = pending / rollback_name
                if previous.exists():
                    if destination.exists():
                        self._restore_replace(
                            destination,
                            pending / f"recovery-failed-{index}",
                            "rollback",
                        )
                    self._restore_replace(previous, destination, "rollback")
                elif not previous_flags[index] and destination.exists():
                    failed = pending / f"recovery-failed-{index}"
                    self._restore_replace(destination, failed, "rollback")
                self._restore_fsync(destination.parent, "rollback")
            self._verify_live_generation()
            shutil.rmtree(pending)
            marker.unlink()
            _fsync_directory(self.directory)
            return {"status": "rolled_back", "backup_id": backup_id}
        except Exception as exc:
            payload["phase"] = "rollback_failed"
            _atomic_json(marker, payload)
            raise BackupError(
                "restore recovery retry failed; recovery marker was preserved"
            ) from exc

    def _manifest_operation_destinations(
        self, manifest: BackupManifest
    ) -> builtins.list[tuple[Path, str]]:
        sources = {source.label: source for source in _sources(self.settings)}
        operations: builtins.list[tuple[Path, str]] = []
        for label in sorted(manifest.roots):
            source = sources[label]
            if source.directory and source.allowed_names:
                operations.extend(
                    (source.path / name, f"{label}-{name}")
                    for name in sorted(source.allowed_names)
                )
            else:
                operations.append((source.path, label))
        return operations

    def _create_from_staged(self, root: Path) -> RestorePreview:
        # Rotation is deliberately implemented through a temporary settings graph,
        # preserving the normal streaming writer and current key generations.
        original = _sources(self.settings)
        overrides = {source.label: root / source.label for source in original}
        return _StagedBackupManager(self, overrides).create()

    def _rings(self) -> tuple[KeyRing, KeyRing]:
        return (
            load_key_ring(self.settings.at_rest.backup.keys),
            load_key_ring(self.settings.at_rest.backup.adapter_keys),
        )

    def _source_list(self) -> builtins.list[_Source]:
        return _sources(self.settings)

    def _inject(self, checkpoint: str) -> None:
        if self._failure_injector is not None:
            self._failure_injector(checkpoint)

    def _require_staging_attestation(self) -> None:
        if (
            self.settings.project.environment == ProjectEnvironment.PRODUCTION
            and not self.settings.at_rest.backup.encrypted_filesystem_attested
        ):
            raise BackupError(
                "production restore staging encrypted filesystem is not attested"
            )

    def _staging_directory(self) -> Path:
        path = self.settings.at_rest.backup.restore_staging_directory
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path, 0o700)
        return path

    def _validate_inventory_sources(self) -> None:
        _validate_adapter_registry_paths(self.settings)

    def _bundle_path(self, backup_id: str) -> Path:
        _validate_backup_id(backup_id)
        return self.directory / f"{backup_id}.kgb"

    def _sidecar_path(self, backup_id: str) -> Path:
        _validate_backup_id(backup_id)
        return self.directory / f"{backup_id}.status.json"

    def _active_adapter_revision(self) -> str | None:
        path = self.settings.adapter_registry.path
        if not path.exists():
            return None
        digest, _size = _hash_file(path)
        return digest


class _StagedBackupManager(BackupManager):
    def __init__(self, parent: BackupManager, overrides: dict[str, Path]) -> None:
        super().__init__(parent.settings)
        self._parent = parent
        self._overrides = overrides

    def _source_list(self) -> builtins.list[_Source]:
        return [
            _Source(
                source.label,
                self._overrides[source.label],
                True,
                source.adapter,
            )
            for source in _sources(self.settings)
        ]

    def _active_adapter_revision(self) -> str | None:
        registry = (
            self._overrides["adapter_registry"]
            / self.settings.adapter_registry.path.name
        )
        if not registry.exists():
            return None
        digest, _size = _hash_file(registry)
        return digest

    def _validate_inventory_sources(self) -> None:
        # The staged graph was already registry/artifact-validated before rotation.
        return


def _sources(settings: Settings) -> list[_Source]:
    journal_names = frozenset(
        [settings.agent_journal.path.name]
        + [
            f"{settings.agent_journal.path.name}.{index}"
            for index in range(1, settings.agent_journal.retained_files + 1)
        ]
    )
    return [
        _Source("agent_state", settings.agent_state.path, False),
        _Source(
            "journal",
            settings.agent_journal.path.parent,
            True,
            allowed_names=journal_names,
        ),
        _Source("state_wal", settings.agent_state_wal.path, False),
        _Source("memory", settings.memory.persist_directory, True),
        _Source("dreams", settings.sleep.dream_dataset_path, False),
        _Source("training_jobs", settings.sleep.job_registry_path, False),
        _Source("training_artifacts", settings.sleep.training_artifact_directory, True),
        _Source("adapters", settings.qlora.output_dir, True, adapter=True),
        _Source("adapter_registry", settings.adapter_registry.path, False),
        _Source("eval_results", settings.adapter_registry.eval_result_dir, True),
        _Source("tool_registry", settings.tools.path, False),
        _Source("tool_audit", settings.tools.audit_path, False),
        _Source("documents", settings.actions.document_root, True),
        _Source("calendar", settings.actions.calendar_path, False),
    ]


def assert_no_incomplete_restore(settings: Settings) -> None:
    """Prevent startup from publishing a generation interrupted during swap."""

    if (settings.at_rest.backup.directory / _RESTORE_MARKER).exists():
        raise BackupError("an incomplete authoritative restore requires recovery")


def _validate_adapter_registry_paths(settings: Settings) -> None:
    path = settings.adapter_registry.path
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackupError("adapter registry is unreadable") from exc
    adapters = payload.get("adapters", []) if isinstance(payload, dict) else None
    if not isinstance(adapters, list):
        raise BackupError("adapter registry schema is invalid")
    root = settings.qlora.output_dir.resolve()
    for item in adapters:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise BackupError("adapter registry entry is invalid")
        artifact = Path(item["path"]).resolve()
        if not artifact.is_relative_to(root):
            raise BackupError("adapter artifact is outside the approved adapter root")


def _validate_staged_adapter_hashes(
    root: Path, registry_path: Path, settings: Settings
) -> None:
    try:
        payload = json.loads(registry_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackupError("restored adapter registry is unreadable") from exc
    adapters = payload.get("adapters", []) if isinstance(payload, dict) else []
    configured_root = settings.qlora.output_dir.resolve()
    for item in adapters:
        if not isinstance(item, dict):
            raise BackupError("restored adapter registry entry is invalid")
        expected_hash = item.get("adapter_hash")
        artifact_value = item.get("path")
        if expected_hash is None:
            continue
        if not isinstance(expected_hash, str) or not isinstance(artifact_value, str):
            raise BackupError("restored adapter hash binding is invalid")
        artifact = Path(artifact_value).resolve()
        if not artifact.is_relative_to(configured_root):
            raise BackupError("restored adapter path is outside its approved root")
        relative = artifact.relative_to(configured_root).as_posix()
        staged = root / "adapters" / relative
        try:
            actual = build_adapter_artifact_manifest(
                staged,
                base_model_name=str(item.get("base_model", settings.model.primary_id)),
                base_model_revision=str(
                    item.get("base_model_revision", settings.model.revision)
                ),
            ).sha256
        except (OSError, ValueError) as exc:
            raise BackupError("restored adapter artifact is invalid") from exc
        if actual != expected_hash:
            raise BackupError("restored adapter artifact hash does not match registry")


def _walk_source(source: _Source) -> Iterator[tuple[Path, str]]:
    if not source.path.exists():
        return
    try:
        root_stat = source.path.lstat()
    except OSError as exc:
        raise BackupError("configured backup source cannot be inspected") from exc
    if stat.S_ISLNK(root_stat.st_mode):
        raise BackupError("backup source symlinks are forbidden")
    if source.directory:
        if not stat.S_ISDIR(root_stat.st_mode):
            raise BackupError("configured backup directory is not a directory")
        for path in sorted(source.path.rglob("*")):
            relative = path.relative_to(source.path)
            if (
                source.allowed_names is not None
                and relative.as_posix() not in source.allowed_names
            ):
                continue
            item_stat = path.lstat()
            if stat.S_ISLNK(item_stat.st_mode):
                raise BackupError("backup source symlinks are forbidden")
            if stat.S_ISDIR(item_stat.st_mode):
                continue
            if not stat.S_ISREG(item_stat.st_mode):
                raise BackupError("backup source special files are forbidden")
            yield path, _safe_relative(relative.as_posix())
    else:
        if not stat.S_ISREG(root_stat.st_mode):
            raise BackupError("configured backup source is not a regular file")
        yield source.path, source.path.name


def _source_exists(source: _Source) -> bool:
    if source.allowed_names is None:
        return source.path.exists()
    return any((source.path / name).exists() for name in source.allowed_names)


def _source_file(source: _Source, relative: str) -> Path:
    relative = _safe_relative(relative)
    candidate = source.path / relative if source.directory else source.path
    if (
        source.directory
        and candidate.resolve().is_relative_to(source.path.resolve()) is False
    ):
        raise BackupError("backup source path escapes its configured root")
    if candidate.is_symlink() or not candidate.is_file():
        raise BackupError("backup source changed to an unsafe file")
    return candidate


def _staged_file(root: Path, label: str, relative: str) -> Path:
    if not label or not label.replace("_", "").isalnum():
        raise BackupError("backup root label is invalid")
    relative = _safe_relative(relative)
    destination = root / label / relative
    if not destination.resolve().is_relative_to((root / label).resolve()):
        raise BackupError("restored path escapes isolation root")
    return destination


def _safe_relative(value: str) -> str:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise BackupError("backup relative path is unsafe")
    return value


def _write_record(
    output: Any,
    codec: EncryptedCodec,
    index: int,
    payload: dict[str, Any],
    *,
    backup_id: str,
    header_hash: str,
) -> None:
    encoded = codec.encode(
        _canonical(payload),
        metadata={
            "record_index": index,
            "backup_id": backup_id,
            "header_hash": header_hash,
        },
    )
    output.write(encoded + b"\n")


def _preview(manifest: BackupManifest) -> RestorePreview:
    return RestorePreview(
        backup_id=manifest.backup_id,
        manifest_hash=manifest.sha256,
        created_at=manifest.created_at,
        file_count=len(manifest.files),
        total_plaintext_bytes=sum(item.size for item in manifest.files),
        base_backup_id=manifest.base_backup_id,
        model_revision=manifest.model_revision,
        adapter_revision=manifest.adapter_revision,
    )


def _bundle_status(
    path: Path, header: dict[str, Any], created_at: datetime
) -> BackupStatus:
    digest, size = _hash_file(path)
    return BackupStatus(
        backup_id=str(header["backup_id"]),
        created_at=created_at,
        encrypted_size=size,
        encrypted_sha256=digest,
        key_ids=[str(item) for item in header["key_ids"]],
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), 0o600)
            output.write(_canonical(payload))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with _open_regular_nofollow(path) as source:
        while chunk := source.read(_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _open_regular_nofollow(path: Path) -> BinaryIO:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise BackupError("backup file cannot be opened safely") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise BackupError("backup source is not a regular file")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def _changed_while_open(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _source_revision() -> str:
    from kagya._build_info import resolve_source_build_info

    info = resolve_source_build_info()
    return info.commit_sha or "unknown"


def _validate_backup_id(backup_id: str) -> None:
    try:
        if str(UUID(backup_id)) != backup_id:
            raise ValueError
    except ValueError as exc:
        raise BackupError("backup ID is invalid") from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
