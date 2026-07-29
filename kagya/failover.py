"""Linearizable etcd v3 fencing and immutable RPO-0 core-state publication."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from typing import Any, Callable, Protocol
from urllib import error, request
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kagya.chat_jobs import validate_chat_job_registry
from kagya.config.schema import EtcdFailoverSettings, Settings
from kagya.learning import AdapterRegistry, AdapterStatus
from kagya.memory.dual_memory_system import configured_memory_identity
from kagya.runtime.agent_state import AgentStateSnapshot
from kagya.runtime.event_journal import (
    EventJournal,
    JournalLifecycle,
    JournalRecord,
    hash_snapshot,
)
from kagya.runtime.state_wal import StateWAL
from kagya.security import build_live_codecs


NOT_AUTHORITATIVE = "not_authoritative"


class FailoverError(RuntimeError):
    """Raised when authority or replica continuity cannot be established."""


class NotAuthoritativeError(FailoverError):
    """Stable failure for a node that cannot authoritatively mutate state."""


class EtcdTransport(Protocol):
    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]: ...


class HttpEtcdTransport:
    """Minimal etcd v3 JSON-gateway client using the standard library."""

    def __init__(self, settings: EtcdFailoverSettings) -> None:
        self._endpoints = tuple(endpoint.rstrip("/") for endpoint in settings.endpoints)
        self._timeout = settings.request_timeout_seconds
        self._next_endpoint = 0
        self._lock = Lock()
        self._auth_token = (
            None
            if settings.auth_token_env is None
            else os.environ.get(settings.auth_token_env)
        )
        if settings.auth_token_env is not None and not self._auth_token:
            raise FailoverError("etcd authentication token is unavailable")

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, separators=(",", ":")).encode("ascii")
        with self._lock:
            start = self._next_endpoint
            self._next_endpoint = (self._next_endpoint + 1) % len(self._endpoints)
        last_error: Exception | None = None
        for offset in range(len(self._endpoints)):
            endpoint = self._endpoints[(start + offset) % len(self._endpoints)]
            headers = {"Content-Type": "application/json"}
            if self._auth_token is not None:
                headers["Authorization"] = self._auth_token
            operation = request.Request(
                endpoint + path,
                data=encoded,
                headers=headers,
                method="POST",
            )
            try:
                with request.urlopen(operation, timeout=self._timeout) as response:
                    value = json.loads(response.read())
                if not isinstance(value, dict):
                    raise FailoverError("etcd returned a non-object response")
                return value
            except (OSError, ValueError, error.HTTPError) as exc:
                last_error = exc
        raise FailoverError("all etcd endpoints are unavailable") from last_error


class FailoverManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 3
    node_id: str
    fencing_token: int = Field(gt=0)
    captured_at: datetime
    processing_sequence: int = Field(ge=0)
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    wal_record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    journal_record_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_id: str
    model_revision: str
    processor_revision: str
    fallback_id: str
    fallback_revision: str
    memory_host: str
    memory_port: int = Field(gt=0, le=65535)
    memory_ssl: bool
    memory_tenant: str
    memory_database: str
    memory_db1_collection: str
    memory_db2_collection: str
    memory_embedding_identity: str
    memory_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    memory_transaction_head: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_id: str | None = None
    adapter_hash: str | None = None
    adapter_activation_sequence: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_manifest(self) -> "FailoverManifest":
        if self.schema_version != 3:
            raise ValueError("unsupported failover manifest schema")
        if self.captured_at.tzinfo is None:
            raise ValueError("failover capture time must include a timezone")
        adapter_values = (
            self.adapter_id,
            self.adapter_hash,
            self.adapter_activation_sequence,
        )
        if any(value is not None for value in adapter_values) and not all(
            value is not None for value in adapter_values
        ):
            raise ValueError("adapter failover identity must be complete")
        return self


class ReplicaBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    manifest: FailoverManifest
    files: dict[str, str]
    bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class PublicationWatermark(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    bundle_key: str
    bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    processing_sequence: int = Field(ge=0)
    fencing_token: int = Field(gt=0)
    node_id: str


class EtcdFencingAuthority:
    """Own one leased writer key and CAS-publish immutable generations."""

    def __init__(
        self,
        node_id: str,
        settings: EtcdFailoverSettings,
        transport: EtcdTransport | None = None,
    ) -> None:
        self.node_id = node_id
        self.settings = settings
        self.transport = transport or HttpEtcdTransport(settings)
        self._prefix = settings.key_prefix.rstrip("/")
        self._lease_key = f"{self._prefix}/authority"
        self._watermark_key = f"{self._prefix}/watermark"
        self._lease_id: int | None = None
        self._authority_value: bytes | None = None
        self._fencing_token: int | None = None
        self._watermark_revision = 0
        self._watermark_sequence = 0
        self._watermark_bundle_key: str | None = None
        self._lost = Event()
        self._stop = Event()
        self._renew_stop = Event()
        self._renew_thread: Thread | None = None
        self._campaign_thread: Thread | None = None
        self._acquisition_conflicts = 0
        self._stale_write_rejections = 0

    @property
    def fencing_token(self) -> int:
        if self._fencing_token is None:
            raise NotAuthoritativeError(NOT_AUTHORITATIVE)
        return self._fencing_token

    @property
    def is_authoritative(self) -> bool:
        return self._fencing_token is not None and not self._lost.is_set()

    @property
    def acquisition_conflicts(self) -> int:
        return self._acquisition_conflicts

    @property
    def stale_write_rejections(self) -> int:
        return self._stale_write_rejections

    def acquire(self) -> bool:
        previous_renewal = self._renew_thread
        if previous_renewal is not None and previous_renewal.is_alive():
            self._renew_stop.set()
            if previous_renewal is not current_thread():
                previous_renewal.join(timeout=self.settings.request_timeout_seconds)
            if previous_renewal.is_alive():
                raise FailoverError("previous authority renewal did not stop")
        grant = self.transport.post(
            "/v3/lease/grant", {"TTL": self.settings.lease_ttl_seconds}
        )
        lease_id = _integer(grant.get("ID"), "lease ID")
        authority_value = _canonical_bytes(
            {"node_id": self.node_id, "tenure_id": str(uuid4()), "lease_id": lease_id}
        )
        result = self.transport.post(
            "/v3/kv/txn",
            {
                "compare": [_compare(self._lease_key, "CREATE", "EQUAL", "0")],
                "success": [_put(self._lease_key, authority_value, lease=lease_id)],
                "failure": [],
            },
        )
        if not result.get("succeeded"):
            self._acquisition_conflicts += 1
            self.transport.post("/v3/lease/revoke", {"ID": lease_id})
            return False
        token = _header_revision(result)
        self._lease_id = lease_id
        self._authority_value = authority_value
        self._fencing_token = token
        self._lost.clear()
        self._renew_stop.clear()
        watermark = self._range(self._watermark_key)
        self._watermark_revision = 0 if watermark is None else watermark[1]
        if watermark is not None:
            published = PublicationWatermark.model_validate_json(watermark[0])
            self._watermark_sequence = published.processing_sequence
            self._watermark_bundle_key = published.bundle_key
        return True

    def start_renewal(self, on_loss: Callable[[], None] | None = None) -> None:
        if not self.is_authoritative:
            raise NotAuthoritativeError(NOT_AUTHORITATIVE)
        if self._renew_thread is not None and self._renew_thread.is_alive():
            return

        def renew() -> None:
            while not self._stop.is_set() and not self._renew_stop.wait(
                self.settings.renew_interval_seconds
            ):
                try:
                    self.renew_once()
                except Exception:
                    self._lost.set()
                    if on_loss is not None:
                        on_loss()
                    return

        self._renew_thread = Thread(
            target=renew, name="kagya-etcd-renewal", daemon=True
        )
        self._renew_thread.start()

    def renew_once(self) -> None:
        try:
            self.assert_authoritative()
            assert self._lease_id is not None
            response = self.transport.post(
                "/v3/lease/keepalive", {"ID": self._lease_id}
            )
            result = response.get("result", response)
            if _integer(result.get("TTL"), "lease TTL") <= 0:
                raise NotAuthoritativeError(NOT_AUTHORITATIVE)
            current = self._range(self._lease_key)
            if current is None or current[0] != self._authority_value:
                raise NotAuthoritativeError(NOT_AUTHORITATIVE)
            if current[1] != self.fencing_token or current[2] != self._lease_id:
                raise NotAuthoritativeError(NOT_AUTHORITATIVE)
        except Exception:
            self._lost.set()
            raise

    def start_campaign(self, on_acquired: Callable[[], None]) -> None:
        """Continue standby election attempts until this node wins or closes."""

        if self._campaign_thread is not None and self._campaign_thread.is_alive():
            return

        def campaign() -> None:
            while not self._stop.wait(self.settings.renew_interval_seconds):
                try:
                    if not self.acquire():
                        continue
                    if self._stop.is_set():
                        self.close()
                        return
                    try:
                        on_acquired()
                    except Exception:
                        self.relinquish()
                        continue
                except Exception:
                    self._lost.set()
                    continue
                return

        self._campaign_thread = Thread(
            target=campaign, name="kagya-etcd-campaign", daemon=True
        )
        self._campaign_thread.start()

    def assert_authoritative(self) -> None:
        if not self.is_authoritative:
            raise NotAuthoritativeError(NOT_AUTHORITATIVE)

    def publish(
        self, manifest: FailoverManifest, files: dict[str, bytes]
    ) -> PublicationWatermark:
        self.assert_authoritative()
        if (
            manifest.fencing_token != self.fencing_token
            or manifest.node_id != self.node_id
        ):
            raise NotAuthoritativeError(NOT_AUTHORITATIVE)
        if manifest.processing_sequence < self._watermark_sequence:
            raise FailoverError(
                "replica watermark processing sequence cannot move backward"
            )
        encoded_files = {
            name: base64.b64encode(value).decode("ascii")
            for name, value in sorted(files.items())
        }
        provisional: dict[str, Any] = {
            "schema_version": 1,
            "manifest": manifest.model_dump(mode="json"),
            "files": encoded_files,
        }
        bundle_hash = hashlib.sha256(_canonical_bytes(provisional)).hexdigest()
        bundle = ReplicaBundle(
            manifest=manifest,
            files=encoded_files,
            bundle_hash=bundle_hash,
        )
        bundle_bytes = _canonical_bytes(bundle.model_dump(mode="json"))
        if len(bundle_bytes) > self.settings.max_bundle_bytes:
            raise FailoverError("replica commit bundle exceeds configured etcd limit")
        bundle_key = f"{self._prefix}/bundles/{self.fencing_token}/{manifest.processing_sequence}/{bundle_hash}"
        watermark = PublicationWatermark(
            bundle_key=bundle_key,
            bundle_hash=bundle_hash,
            processing_sequence=manifest.processing_sequence,
            fencing_token=self.fencing_token,
            node_id=self.node_id,
        )
        compares = [
            _compare(self._lease_key, "MOD", "EQUAL", str(self.fencing_token)),
            _compare(
                self._lease_key, "VALUE", "EQUAL", _b64(self._authority_value or b"")
            ),
            _compare(bundle_key, "CREATE", "EQUAL", "0"),
            _compare(
                self._watermark_key, "MOD", "EQUAL", str(self._watermark_revision)
            ),
        ]
        previous_bundle_key = self._watermark_bundle_key
        success_operations = [
            _put(bundle_key, bundle_bytes),
            _put(
                self._watermark_key,
                _canonical_bytes(watermark.model_dump(mode="json")),
            ),
        ]
        if previous_bundle_key is not None and previous_bundle_key != bundle_key:
            success_operations.append(_delete(previous_bundle_key))
        result = self.transport.post(
            "/v3/kv/txn",
            {
                "compare": compares,
                "success": success_operations,
                "failure": [],
            },
        )
        if not result.get("succeeded"):
            self._stale_write_rejections += 1
            self._lost.set()
            raise NotAuthoritativeError(NOT_AUTHORITATIVE)
        self._watermark_revision = _header_revision(result)
        self._watermark_sequence = manifest.processing_sequence
        self._watermark_bundle_key = bundle_key
        return watermark

    def latest_bundle(self) -> ReplicaBundle | None:
        current = self._range(self._watermark_key)
        if current is None:
            self._watermark_revision = 0
            self._watermark_sequence = 0
            self._watermark_bundle_key = None
            return None
        watermark = PublicationWatermark.model_validate_json(current[0])
        bundle_value = self._range(watermark.bundle_key)
        if bundle_value is None:
            raise FailoverError("published replica bundle is missing")
        bundle = ReplicaBundle.model_validate_json(bundle_value[0])
        _verify_bundle(bundle)
        if bundle.bundle_hash != watermark.bundle_hash:
            raise FailoverError("replica bundle does not match its watermark")
        if (
            bundle.manifest.processing_sequence != watermark.processing_sequence
            or bundle.manifest.fencing_token != watermark.fencing_token
            or bundle.manifest.node_id != watermark.node_id
        ):
            raise FailoverError("replica manifest does not match its watermark")
        self._watermark_revision = current[1]
        self._watermark_sequence = watermark.processing_sequence
        self._watermark_bundle_key = watermark.bundle_key
        return bundle

    def close(self) -> None:
        self._stop.set()
        if (
            self._renew_thread is not None
            and self._renew_thread is not current_thread()
        ):
            self._renew_thread.join(timeout=self.settings.request_timeout_seconds)
        if (
            self._campaign_thread is not None
            and self._campaign_thread is not current_thread()
        ):
            self._campaign_thread.join(timeout=self.settings.request_timeout_seconds)
        self.relinquish()

    def relinquish(self) -> None:
        lease_id = self._lease_id
        self._lost.set()
        self._renew_stop.set()
        self._lease_id = None
        self._authority_value = None
        self._fencing_token = None
        if lease_id is not None:
            try:
                self.transport.post("/v3/lease/revoke", {"ID": lease_id})
            except Exception:
                pass
        renewal = self._renew_thread
        if renewal is not None and renewal is not current_thread():
            renewal.join(timeout=self.settings.request_timeout_seconds)

    def _range(self, key: str) -> tuple[bytes, int, int] | None:
        response = self.transport.post(
            "/v3/kv/range", {"key": _b64(key.encode("utf-8"))}
        )
        values = response.get("kvs", [])
        if not values:
            return None
        item = values[0]
        return (
            base64.b64decode(item["value"], validate=True),
            _integer(item.get("mod_revision"), "mod revision"),
            _integer(item.get("lease", 0), "lease ID"),
        )


def build_manifest(
    settings: Settings,
    authority: EtcdFencingAuthority,
    app: Any,
    *,
    journal_records: list[JournalRecord] | None = None,
) -> FailoverManifest:
    snapshot = app.state.agent_state_store.last_snapshot
    if snapshot is None:
        raise FailoverError("authoritative snapshot is unavailable")
    wal_records = app.state.state_wal.verify()
    records = (
        app.state.event_journal.verify() if journal_records is None else journal_records
    )
    active_entries = [
        entry
        for entry in app.state.adapter_registry.list()
        if entry.status == AdapterStatus.ACTIVE
    ]
    if len(active_entries) > 1:
        raise FailoverError("multiple active adapters cannot be published")
    active = active_entries[0] if active_entries else None
    memory = app.state.memory_system
    memory_hash, transaction_head = memory.shared_memory_state()
    return FailoverManifest(
        node_id=settings.deployment.node.id,
        fencing_token=authority.fencing_token,
        captured_at=datetime.now(UTC),
        processing_sequence=snapshot.last_processed_event_sequence,
        snapshot_hash=hash_snapshot(snapshot),
        wal_record_hash=wal_records[-1].record_hash,
        journal_record_hash=records[-1].record_hash if records else None,
        model_id=settings.model.primary_id,
        model_revision=settings.model.revision,
        processor_revision=settings.model.processor_revision,
        fallback_id=settings.model.fallback_id,
        fallback_revision=settings.model.fallback_revision,
        memory_host=settings.memory.http_host,
        memory_port=settings.memory.http_port,
        memory_ssl=settings.memory.http_ssl,
        memory_tenant=settings.memory.http_tenant,
        memory_database=settings.memory.http_database,
        memory_db1_collection=memory.db1_collection_name,
        memory_db2_collection=memory.db2_collection_name,
        memory_embedding_identity=memory.embedding_identity,
        memory_hash=memory_hash,
        memory_transaction_head=transaction_head,
        adapter_id=None if active is None else active.adapter_id,
        adapter_hash=None if active is None else active.adapter_hash,
        adapter_activation_sequence=None
        if active is None
        else active.activation_sequence,
    )


def core_files(
    settings: Settings,
    *,
    journal: EventJournal | None = None,
    journal_files: dict[str, bytes] | None = None,
) -> dict[str, bytes]:
    paths = {
        "snapshot": settings.agent_state.path,
        "wal": settings.agent_state_wal.path,
        "chat_jobs": settings.api.chat_job_registry_path,
        "adapter_registry": settings.adapter_registry.path,
        "adapter_history": settings.adapter_registry.path.with_name(
            f"{settings.adapter_registry.path.stem}_activations.json"
        ),
    }
    files = {name: path.read_bytes() for name, path in paths.items() if path.exists()}
    if journal_files is not None:
        files.update(journal_files)
    elif journal is not None:
        files.update(journal.durable_files())
    else:
        files["journal"] = (
            settings.agent_journal.path.read_bytes()
            if settings.agent_journal.path.exists()
            else b""
        )
        for index in range(1, settings.agent_journal.retained_files + 1):
            path = settings.agent_journal.path.with_name(
                f"{settings.agent_journal.path.name}.{index}"
            )
            if path.exists():
                files[f"journal.{index}"] = path.read_bytes()
    return files


def restore_and_preflight(settings: Settings, bundle: ReplicaBundle) -> None:
    manifest = bundle.manifest
    db1_collection, db2_collection, embedding_identity = configured_memory_identity(
        settings
    )
    expected = (
        settings.model.primary_id,
        settings.model.revision,
        settings.model.processor_revision,
        settings.model.fallback_id,
        settings.model.fallback_revision,
        settings.memory.http_host,
        settings.memory.http_port,
        settings.memory.http_ssl,
        settings.memory.http_tenant,
        settings.memory.http_database,
        db1_collection,
        db2_collection,
        embedding_identity,
    )
    actual = (
        manifest.model_id,
        manifest.model_revision,
        manifest.processor_revision,
        manifest.fallback_id,
        manifest.fallback_revision,
        manifest.memory_host,
        manifest.memory_port,
        manifest.memory_ssl,
        manifest.memory_tenant,
        manifest.memory_database,
        manifest.memory_db1_collection,
        manifest.memory_db2_collection,
        manifest.memory_embedding_identity,
    )
    if actual != expected:
        raise FailoverError("promotion model or shared memory identity mismatch")
    decoded = {
        name: base64.b64decode(value, validate=True)
        for name, value in bundle.files.items()
    }
    required = {"snapshot", "wal", "journal"}
    if not required.issubset(decoded):
        raise FailoverError("replica bundle lacks core state")
    staging = settings.agent_state.path.parent / f".failover-preflight-{uuid4()}"
    staging.mkdir(parents=True)
    try:
        snapshot_path = staging / settings.agent_state.path.name
        wal_path = staging / settings.agent_state_wal.path.name
        journal_path = staging / settings.agent_journal.path.name
        mapping = {"snapshot": snapshot_path, "wal": wal_path, "journal": journal_path}
        mapping["chat_jobs"] = staging / settings.api.chat_job_registry_path.name
        mapping["adapter_registry"] = staging / settings.adapter_registry.path.name
        mapping["adapter_history"] = staging / (
            f"{settings.adapter_registry.path.stem}_activations.json"
        )
        for index in range(1, settings.agent_journal.retained_files + 1):
            mapping[f"journal.{index}"] = journal_path.with_name(
                f"{journal_path.name}.{index}"
            )
        for name, value in decoded.items():
            path = mapping.get(name)
            if path is None:
                raise FailoverError("replica bundle contains an unknown core file")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        codecs = build_live_codecs(settings)
        plaintext = codecs.snapshot.decode(
            snapshot_path.read_bytes(),
            expected_metadata={"record_type": "snapshot"}
            if codecs.snapshot.enabled
            else None,
        )
        snapshot = AgentStateSnapshot.model_validate_json(plaintext)
        wal = StateWAL(wal_path, codec=codecs.wal)
        reconstructed = wal.reconstruct()
        journal = EventJournal(
            journal_path,
            max_bytes=settings.agent_journal.max_bytes,
            retained_files=settings.agent_journal.retained_files,
            codec=codecs.journal,
        )
        journal_records = journal.verify()
        chat_jobs_path = mapping["chat_jobs"]
        if chat_jobs_path.exists():
            validate_chat_job_registry(chat_jobs_path, codecs.chat_request_spool)
        registry_settings = settings.model_copy(
            update={
                "adapter_registry": settings.adapter_registry.model_copy(
                    update={"path": mapping["adapter_registry"]}
                )
            }
        )
        active_entries = [
            entry
            for entry in AdapterRegistry(registry_settings).list()
            if entry.status == AdapterStatus.ACTIVE
        ]
        if len(active_entries) > 1:
            raise FailoverError("promotion has multiple active adapters")
        active = active_entries[0] if active_entries else None
        adapter = (
            None if active is None else active.adapter_id,
            None if active is None else active.adapter_hash,
            None if active is None else active.activation_sequence,
        )
        if adapter != (
            manifest.adapter_id,
            manifest.adapter_hash,
            manifest.adapter_activation_sequence,
        ):
            raise FailoverError("promotion active adapter identity mismatch")
        if (
            reconstructed.sequence != snapshot.last_processed_event_sequence
            or reconstructed.snapshot_hash != hash_snapshot(snapshot)
            or manifest.processing_sequence != reconstructed.sequence
            or manifest.snapshot_hash != reconstructed.snapshot_hash
            or manifest.wal_record_hash != wal.verify()[-1].record_hash
        ):
            raise FailoverError("promotion Journal/WAL/snapshot continuity mismatch")
        if manifest.journal_record_hash != (
            journal_records[-1].record_hash if journal_records else None
        ):
            raise FailoverError("promotion Journal head mismatch")
        terminal = [
            record
            for record in journal_records
            if record.lifecycle in {JournalLifecycle.COMPLETED, JournalLifecycle.FAILED}
            or (
                record.lifecycle == JournalLifecycle.RECOVERY_CLASSIFIED
                and (
                    record.failure_category == "committed_before_crash"
                    or (
                        record.failure_category == "uncommitted_after_crash"
                        and record.processing_sequence == record.snapshot_sequence
                    )
                )
            )
        ]
        if snapshot.last_processed_event_sequence and (
            not terminal
            or terminal[-1].snapshot_sequence != snapshot.last_processed_event_sequence
            or terminal[-1].snapshot_hash != reconstructed.snapshot_hash
        ):
            raise FailoverError("promotion terminal Journal continuity mismatch")
        _install_files(settings, decoded)
    finally:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()


def _install_files(settings: Settings, files: dict[str, bytes]) -> None:
    targets = {
        "snapshot": settings.agent_state.path,
        "wal": settings.agent_state_wal.path,
        "journal": settings.agent_journal.path,
        "chat_jobs": settings.api.chat_job_registry_path,
        "adapter_registry": settings.adapter_registry.path,
        "adapter_history": settings.adapter_registry.path.with_name(
            f"{settings.adapter_registry.path.stem}_activations.json"
        ),
    }
    for index in range(1, settings.agent_journal.retained_files + 1):
        targets[f"journal.{index}"] = settings.agent_journal.path.with_name(
            f"{settings.agent_journal.path.name}.{index}"
        )
    temporaries: dict[str, Path] = {}
    try:
        for name, target in targets.items():
            if name not in files:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid4()}.replica")
            with temporary.open("xb") as output:
                os.fchmod(output.fileno(), 0o600)
                output.write(files[name])
                output.flush()
                os.fsync(output.fileno())
            temporaries[name] = temporary
        for name, target in targets.items():
            if name in temporaries:
                os.replace(temporaries[name], target)
            elif name.startswith("journal.") or name in {
                "chat_jobs",
                "adapter_registry",
                "adapter_history",
            }:
                target.unlink(missing_ok=True)
        for directory in {path.parent for path in targets.values()}:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)


def _verify_bundle(bundle: ReplicaBundle) -> None:
    value = bundle.model_dump(mode="json", exclude={"bundle_hash"})
    if hashlib.sha256(_canonical_bytes(value)).hexdigest() != bundle.bundle_hash:
        raise FailoverError("replica bundle hash mismatch")


def _put(key: str, value: bytes, *, lease: int | None = None) -> dict[str, Any]:
    request_value: dict[str, Any] = {"key": _b64(key.encode()), "value": _b64(value)}
    if lease is not None:
        request_value["lease"] = str(lease)
    return {"request_put": request_value}


def _delete(key: str) -> dict[str, Any]:
    return {"request_delete_range": {"key": _b64(key.encode())}}


def _compare(key: str, target: str, result: str, value: str) -> dict[str, str]:
    field = {"CREATE": "create_revision", "MOD": "mod_revision", "VALUE": "value"}[
        target
    ]
    return {"key": _b64(key.encode()), "target": target, "result": result, field: value}


def _header_revision(response: dict[str, Any]) -> int:
    return _integer(response.get("header", {}).get("revision"), "etcd revision")


def _integer(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise FailoverError(f"etcd response has invalid {label}") from exc
    if result < 0:
        raise FailoverError(f"etcd response has invalid {label}")
    return result


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")
