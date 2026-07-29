"""Durable event transactions over a shared Chroma service."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from threading import RLock
from typing import Any

from kagya.external_transaction import (
    ExternalTransactionAudit,
    ExternalTransactionRecord,
    ExternalTransactionStatus,
)

_LOGICAL_ID = "_kagya_tx_logical_id"
_EVENT_ID = "_kagya_tx_event_id"
_COMMAND_INDEX = "_kagya_tx_command_index"
_PROCESSING_SEQUENCE = "_kagya_tx_processing_sequence"
_RESERVED = {_LOGICAL_ID, _EVENT_ID, _COMMAND_INDEX, _PROCESSING_SEQUENCE}
_CONTINUITY_ID = "transaction-continuity-v1"
_ZERO_HASH = "0" * 64
_MARKER_RETENTION = 64


class ChromaTransactionStore:
    """Keep immutable commands and one visibility marker per runtime event."""

    def __init__(
        self,
        client: Any,
        *,
        collection_name: str,
        node_id: str,
        fencing_token: Callable[[], int],
        embedding_function: Any,
        event_provider: Callable[[], Any | None],
    ) -> None:
        self._collection = client.get_or_create_collection(
            collection_name, embedding_function=embedding_function
        )
        self._node_id = node_id
        self._fencing_token = fencing_token
        self._event_provider = event_provider
        self._collections: dict[str, Any] = {}
        self._next_indices: dict[str, int] = {}
        self._lock = RLock()
        self._materialization_failure_injector: Callable[[str, str], None] | None = None

    def wrap(self, collection: Any, collection_role: str) -> TransactionalCollection:
        self._collections[collection_role] = collection
        return TransactionalCollection(collection, collection_role, self)

    def is_staging(self) -> bool:
        event = self._event_provider()
        return event is not None and event.processing_sequence is not None

    def stage(
        self,
        role: str,
        operation: str,
        ids: list[str],
        versions: list[dict[str, Any]],
    ) -> None:
        event = self._event_provider()
        if event is None or event.processing_sequence is None:
            raise RuntimeError("shared Chroma staging requires an AgentRuntime event")
        with self._lock:
            continuity = self._continuity()
            if (
                self.is_committed(event.event_id)
                or self.is_compensated(event.event_id)
                or event.processing_sequence
                <= int(continuity.get("sealed_through_sequence", -1))
            ):
                raise RuntimeError("external event is already sealed")
            index = self._next_indices.get(event.event_id)
            if index is None:
                index = self._existing_command_count(event.event_id)
            self._next_indices[event.event_id] = index + 1
            command_id = _command_id(event.event_id, index)
            payload = {
                "schema_version": 1,
                "collection": role,
                "operation": operation,
                "ids": ids,
                "version_ids": [item["physical_id"] for item in versions],
            }
            document = _canonical_json(payload)
            existing = self._collection.get(ids=[command_id], include=["documents"])
            if existing.get("ids"):
                if (existing.get("documents") or [None])[0] != document:
                    raise RuntimeError("external command identity was reused")
            else:
                self._collection.add(
                    ids=[command_id],
                    documents=[document],
                    metadatas=[
                        {
                            "kind": "command",
                            "event_id": event.event_id,
                            "command_index": index,
                            "processing_sequence": event.processing_sequence,
                            "fencing_token": self._fencing_token(),
                            "node_id": self._node_id,
                            "collection": role,
                            "operation": operation,
                            "created_at": _now_iso(),
                        }
                    ],
                )
            collection = self._collections[role]
            for version in versions:
                physical_id = version["physical_id"]
                if collection.get(ids=[physical_id]).get("ids"):
                    continue
                kwargs = {
                    key: value
                    for key, value in version.items()
                    if key in {"embeddings", "metadatas", "documents", "uris"}
                    and value is not None
                }
                collection.add(ids=[physical_id], **kwargs)

    def finalize_event(self, event_id: str, processing_sequence: int) -> int:
        with self._lock:
            commands = self._commands(event_id)
            if not commands:
                return 0
            if self.is_compensated(event_id):
                raise RuntimeError("compensated external commands cannot be committed")
            marker_id = _marker_id("committed", event_id)
            existing_marker = self._collection.get(
                ids=[marker_id], include=["metadatas"]
            )
            if existing_marker.get("ids"):
                metadata = (existing_marker.get("metadatas") or [{}])[0] or {}
                if int(metadata.get("processing_sequence", -1)) != processing_sequence:
                    raise ValueError("external commit processing sequence mismatch")
                return 0
            for command in commands:
                if command["processing_sequence"] != processing_sequence:
                    raise ValueError("external command processing sequence mismatch")
                payload = command["payload"]
                collection = self._collections[str(payload["collection"])]
                version_ids = [str(item) for item in payload["version_ids"]]
                if version_ids and len(
                    collection.get(ids=version_ids).get("ids") or []
                ) != len(version_ids):
                    raise RuntimeError(
                        "external command has an incomplete prepared version"
                    )
            self._collection.add(
                ids=[marker_id],
                documents=["committed"],
                metadatas=[
                    {
                        "kind": "committed",
                        "event_id": event_id,
                        "processing_sequence": processing_sequence,
                        "command_count": len(commands),
                        "fencing_token": self._fencing_token(),
                        "node_id": self._node_id,
                        "previous_memory_hash": self.canonical_memory_hash(),
                        "previous_transaction_head": self.transaction_head(),
                        "applied_at": _now_iso(),
                    }
                ],
            )
            self._next_indices.pop(event_id, None)
            return len(commands)

    def materialize_event(self, event_id: str, processing_sequence: int) -> int:
        """Canonicalize one committed overlay and remove its transient payloads."""

        with self._lock:
            commands = self._commands(event_id)
            marker = self._marker("committed", event_id)
            if not commands:
                if marker is not None:
                    self._collection.delete(ids=[_marker_id("committed", event_id)])
                return 0
            if marker is None:
                if self._marker("materialized", event_id) is not None:
                    return 0
                raise RuntimeError("external event has no commit evidence")
            if int(marker.get("processing_sequence", -1)) != processing_sequence:
                raise ValueError("external materialization sequence mismatch")
            self._inject_materialization_failure("before_canonical", event_id)
            affected: dict[str, set[str]] = {}
            deletes: dict[str, set[str]] = {}
            for command in commands:
                payload = command["payload"]
                role = str(payload["collection"])
                affected.setdefault(role, set()).update(
                    str(item) for item in payload["ids"]
                )
                if payload["operation"] == "delete":
                    deletes.setdefault(role, set()).update(
                        str(item) for item in payload["ids"]
                    )
            for role, logical_ids in affected.items():
                wrapper = TransactionalCollection(self._collections[role], role, self)
                raw = self._collections[role].get(
                    include=["metadatas", "documents", "embeddings", "uris"]
                )
                final_rows = {row["id"]: row for row in wrapper._visible_rows(raw)}
                for logical_id in sorted(logical_ids):
                    if logical_id in deletes.get(role, set()):
                        self._delete_logical_history(role, logical_id)
                        continue
                    row = final_rows.get(logical_id)
                    if row is None:
                        raise RuntimeError("materialized record is unavailable")
                    self._upsert_canonical(role, row)
            self._inject_materialization_failure("after_canonical", event_id)
            memory_hash = self.canonical_memory_hash()
            self._seal_continuity(
                event_id=event_id,
                processing_sequence=processing_sequence,
                disposition="materialized",
                previous_memory_hash=str(marker["previous_memory_hash"]),
                previous_transaction_head=str(marker["previous_transaction_head"]),
                memory_hash=memory_hash,
            )
            self._inject_materialization_failure("after_continuity", event_id)
            self._retain_terminal_marker(
                "materialized", event_id, processing_sequence, memory_hash
            )
            self._cleanup_commands(commands)
            self._collection.delete(ids=[_marker_id("committed", event_id)])
            self._prune_terminal_markers()
            self._inject_materialization_failure("after_cleanup", event_id)
            return len(commands)

    def compensate_event(self, event_id: str, reason: str) -> int:
        with self._lock:
            commands = self._commands(event_id)
            if self.is_committed(event_id):
                return 0
            marker_id = _marker_id("compensated", event_id)
            marker = self._marker("compensated", event_id)
            if not commands and marker is not None:
                return 0
            if not commands:
                return 0
            sequence = int(commands[0]["processing_sequence"])
            if marker is None:
                self._collection.add(
                    ids=[marker_id],
                    documents=["compensated"],
                    metadatas=[
                        {
                            "kind": "compensated",
                            "event_id": event_id,
                            "processing_sequence": sequence,
                            "command_count": len(commands),
                            "reason": _safe_reason(reason),
                            "previous_memory_hash": self.canonical_memory_hash(),
                            "previous_transaction_head": self.transaction_head(),
                            "applied_at": _now_iso(),
                        }
                    ],
                )
                marker = self._marker("compensated", event_id)
            assert marker is not None
            self._seal_continuity(
                event_id=event_id,
                processing_sequence=sequence,
                disposition="compensated",
                previous_memory_hash=str(marker["previous_memory_hash"]),
                previous_transaction_head=str(marker["previous_transaction_head"]),
                memory_hash=self.canonical_memory_hash(),
            )
            self._cleanup_commands(commands)
            self._next_indices.pop(event_id, None)
            self._prune_terminal_markers()
            return len(commands)

    def is_committed(self, event_id: str) -> bool:
        return bool(
            self._collection.get(ids=[_marker_id("committed", event_id)]).get("ids")
        )

    def is_compensated(self, event_id: str) -> bool:
        return bool(
            self._collection.get(ids=[_marker_id("compensated", event_id)]).get("ids")
        )

    def visible_events(self) -> set[str]:
        result = self._collection.get(
            where={"kind": "committed"}, include=["metadatas"]
        )
        return {
            str(metadata["event_id"])
            for metadata in result.get("metadatas") or []
            if metadata is not None
        }

    def recover_materialization(self) -> None:
        result = self._collection.get(
            where={"kind": "committed"}, include=["metadatas"]
        )
        for metadata in result.get("metadatas") or []:
            if metadata is not None:
                self.materialize_event(
                    str(metadata["event_id"]),
                    int(metadata["processing_sequence"]),
                )
        compensated = self._collection.get(
            where={"kind": "compensated"}, include=["metadatas"]
        )
        for metadata in compensated.get("metadatas") or []:
            if metadata is not None:
                self.compensate_event(
                    str(metadata["event_id"]), str(metadata.get("reason", "recovered"))
                )

    def canonical_memory_hash(self) -> str:
        records: list[dict[str, Any]] = []
        for role, collection in sorted(self._collections.items()):
            result = collection.get(
                include=["metadatas", "documents", "embeddings", "uris"]
            )
            for index, physical_id in enumerate(result.get("ids") or []):
                metadata = dict(_value_at(result.get("metadatas"), index) or {})
                if _LOGICAL_ID in metadata:
                    continue
                records.append(
                    {
                        "collection": role,
                        "id": str(physical_id),
                        "metadata": metadata,
                        "document": _value_at(result.get("documents"), index),
                        "embedding": _json_value(
                            _value_at(result.get("embeddings"), index)
                        ),
                        "uri": _value_at(result.get("uris"), index),
                    }
                )
        records.sort(key=lambda item: (item["collection"], item["id"]))
        return hashlib.sha256(_canonical_json(records).encode("ascii")).hexdigest()

    def transaction_head(self) -> str:
        return str(self._continuity().get("head_hash", _ZERO_HASH))

    def continuity(self) -> dict[str, Any]:
        return dict(self._continuity())

    def readiness_probe(self) -> None:
        self._collection.get(ids=[_CONTINUITY_ID], include=[])
        for collection in self._collections.values():
            collection.query(
                query_texts=["kagya-readiness-probe"], n_results=1, include=[]
            )

    def set_materialization_failure_injector(
        self, injector: Callable[[str, str], None] | None
    ) -> None:
        self._materialization_failure_injector = injector

    def assert_not_ahead(self, processing_sequence: int) -> None:
        result = self._collection.get(
            where={"kind": "committed"}, include=["metadatas"]
        )
        ahead = [
            int(metadata.get("processing_sequence", -1))
            for metadata in result.get("metadatas") or []
            if metadata is not None
            and int(metadata.get("processing_sequence", -1)) > processing_sequence
        ]
        if ahead:
            raise RuntimeError("shared Chroma commit is ahead of the promoted snapshot")
        sealed_sequence = int(self._continuity().get("sealed_through_sequence", -1))
        if sealed_sequence > processing_sequence:
            raise RuntimeError(
                "shared Chroma continuity is ahead of the promoted snapshot"
            )

    def current_event_id(self) -> str | None:
        event = self._event_provider()
        return event.event_id if self.is_staging() and event is not None else None

    def command_operations(self, role: str) -> list[dict[str, Any]]:
        result = self._collection.get(
            where={"$and": [{"kind": "command"}, {"collection": role}]},
            include=["documents", "metadatas"],
        )
        visible = self.visible_events()
        current = self.current_event_id()
        operations: list[dict[str, Any]] = []
        for document, metadata in zip(
            result.get("documents") or [], result.get("metadatas") or [], strict=False
        ):
            if metadata is None:
                continue
            event_id = str(metadata["event_id"])
            if event_id not in visible and event_id != current:
                continue
            payload = json.loads(str(document))
            payload["event_id"] = event_id
            payload["processing_sequence"] = int(metadata["processing_sequence"])
            payload["command_index"] = int(metadata["command_index"])
            operations.append(payload)
        return sorted(
            operations,
            key=lambda item: (item["processing_sequence"], item["command_index"]),
        )

    def records(self) -> list[ExternalTransactionRecord]:
        result = self._collection.get(
            where={"kind": "command"}, include=["documents", "metadatas"]
        )
        records: list[ExternalTransactionRecord] = []
        for command_id, document, metadata in zip(
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or [],
            strict=False,
        ):
            if metadata is None:
                continue
            event_id = str(metadata["event_id"])
            status = (
                ExternalTransactionStatus.COMMITTED
                if self.is_committed(event_id)
                else ExternalTransactionStatus.COMPENSATED
                if self.is_compensated(event_id)
                else ExternalTransactionStatus.PENDING
            )
            payload = json.loads(str(document))
            created_at = str(metadata["created_at"])
            records.append(
                ExternalTransactionRecord(
                    revision=1,
                    transaction_id=str(command_id),
                    artifact_type=f"chroma_{payload['collection']}_{payload['operation']}",
                    artifact_id=",".join(str(item) for item in payload["ids"]),
                    status=status,
                    event_id=event_id,
                    processing_sequence=int(metadata["processing_sequence"]),
                    source="agent_runtime",
                    created_at=created_at,
                    updated_at=created_at,
                    audit=[
                        ExternalTransactionAudit(
                            revision=1,
                            status=status,
                            timestamp=created_at,
                            reason="event_command",
                        )
                    ],
                )
            )
        return records

    def _existing_command_count(self, event_id: str) -> int:
        return len(self._commands(event_id))

    def _commands(self, event_id: str) -> list[dict[str, Any]]:
        result = self._collection.get(
            where={"$and": [{"kind": "command"}, {"event_id": event_id}]},
            include=["documents", "metadatas"],
        )
        values = []
        for command_id, document, metadata in zip(
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or [],
            strict=False,
        ):
            if metadata is not None:
                values.append(
                    {
                        "command_id": str(command_id),
                        "payload": json.loads(str(document)),
                        "processing_sequence": int(metadata["processing_sequence"]),
                        "command_index": int(metadata["command_index"]),
                    }
                )
        return sorted(values, key=lambda item: item["command_index"])

    def _marker(self, kind: str, event_id: str) -> dict[str, Any] | None:
        result = self._collection.get(
            ids=[_marker_id(kind, event_id)], include=["metadatas"]
        )
        values = result.get("metadatas") or []
        return None if not values or values[0] is None else dict(values[0])

    def _continuity(self) -> dict[str, Any]:
        result = self._collection.get(ids=[_CONTINUITY_ID], include=["metadatas"])
        values = result.get("metadatas") or []
        return {} if not values or values[0] is None else dict(values[0])

    def _seal_continuity(
        self,
        *,
        event_id: str,
        processing_sequence: int,
        disposition: str,
        previous_memory_hash: str,
        previous_transaction_head: str,
        memory_hash: str,
    ) -> None:
        payload = {
            "event_id": event_id,
            "processing_sequence": processing_sequence,
            "disposition": disposition,
            "previous_memory_hash": previous_memory_hash,
            "memory_hash": memory_hash,
            "previous_transaction_head": previous_transaction_head,
        }
        head_hash = hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()
        current = self._continuity()
        if int(current.get("sealed_through_sequence", -1)) > processing_sequence:
            raise RuntimeError("transaction continuity cannot move backward")
        metadata = {
            "kind": "continuity",
            "sealed_through_sequence": processing_sequence,
            "event_id": event_id,
            "disposition": disposition,
            "previous_memory_hash": previous_memory_hash,
            "memory_hash": memory_hash,
            "previous_head_hash": previous_transaction_head,
            "head_hash": head_hash,
            "updated_at": _now_iso(),
        }
        self._collection.upsert(
            ids=[_CONTINUITY_ID], documents=["continuity"], metadatas=[metadata]
        )

    def _upsert_canonical(self, role: str, row: dict[str, Any]) -> None:
        kwargs: dict[str, Any] = {"metadatas": [row["metadata"]]}
        for source, target in (
            ("document", "documents"),
            ("embedding", "embeddings"),
            ("uri", "uris"),
        ):
            if row[source] is not None:
                kwargs[target] = [row[source]]
        self._collections[role].upsert(ids=[row["id"]], **kwargs)

    def _delete_logical_history(self, role: str, logical_id: str) -> None:
        collection = self._collections[role]
        result = collection.get(include=["metadatas"])
        physical_ids = [
            str(physical_id)
            for physical_id, raw_metadata in zip(
                result.get("ids") or [], result.get("metadatas") or [], strict=False
            )
            if str(physical_id) == logical_id
            or str((raw_metadata or {}).get(_LOGICAL_ID, "")) == logical_id
        ]
        if physical_ids:
            collection.delete(ids=physical_ids)

    def _cleanup_commands(self, commands: list[dict[str, Any]]) -> None:
        version_ids_by_role: dict[str, list[str]] = {}
        for command in commands:
            payload = command["payload"]
            version_ids_by_role.setdefault(str(payload["collection"]), []).extend(
                str(item) for item in payload["version_ids"]
            )
        for role, version_ids in version_ids_by_role.items():
            if version_ids:
                self._collections[role].delete(ids=list(dict.fromkeys(version_ids)))
        command_ids = [str(command["command_id"]) for command in commands]
        if command_ids:
            self._collection.delete(ids=command_ids)

    def _retain_terminal_marker(
        self, kind: str, event_id: str, processing_sequence: int, memory_hash: str
    ) -> None:
        marker_id = _marker_id(kind, event_id)
        if self._collection.get(ids=[marker_id]).get("ids"):
            return
        self._collection.add(
            ids=[marker_id],
            documents=[kind],
            metadatas=[
                {
                    "kind": kind,
                    "event_id": event_id,
                    "processing_sequence": processing_sequence,
                    "memory_hash": memory_hash,
                    "applied_at": _now_iso(),
                }
            ],
        )

    def _prune_terminal_markers(self) -> None:
        result = self._collection.get(
            where={"kind": {"$in": ["materialized", "compensated"]}},
            include=["metadatas"],
        )
        markers = sorted(
            (
                (int(metadata.get("processing_sequence", -1)), str(marker_id))
                for marker_id, metadata in zip(
                    result.get("ids") or [],
                    result.get("metadatas") or [],
                    strict=False,
                )
                if metadata is not None
            ),
            reverse=True,
        )
        obsolete = [marker_id for _, marker_id in markers[_MARKER_RETENTION:]]
        if obsolete:
            self._collection.delete(ids=obsolete)

    def _inject_materialization_failure(self, phase: str, event_id: str) -> None:
        if self._materialization_failure_injector is not None:
            self._materialization_failure_injector(phase, event_id)


class TransactionalCollection:
    """Collection facade exposing only the committed logical record set."""

    def __init__(
        self, collection: Any, role: str, store: ChromaTransactionStore
    ) -> None:
        self._collection = collection
        self._role = role
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._collection, name)

    @property
    def name(self) -> str:
        return str(self._collection.name)

    def add(self, ids: str | list[str], **kwargs: Any) -> None:
        normalized = _ids(ids)
        if not self._store.is_staging():
            self._collection.add(ids=normalized, **kwargs)
            return
        existing = set(self.get(ids=normalized).get("ids") or [])
        if existing:
            raise ValueError(f"IDs already exist: {sorted(existing)}")
        self._stage_versions("add", normalized, kwargs)

    def update(self, ids: str | list[str], **kwargs: Any) -> None:
        normalized = _ids(ids)
        if not self._store.is_staging():
            physical = self._physical_ids(normalized)
            if physical:
                self._collection.update(ids=physical, **kwargs)
            return
        self._stage_versions("update", normalized, kwargs, existing_only=True)

    def upsert(self, ids: str | list[str], **kwargs: Any) -> None:
        normalized = _ids(ids)
        if not self._store.is_staging():
            physical = self._physical_ids(normalized)
            self._collection.upsert(ids=physical or normalized, **kwargs)
            return
        self._stage_versions("upsert", normalized, kwargs)

    def delete(
        self,
        ids: list[str] | None = None,
        where: Mapping[str, Any] | None = None,
        where_document: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> Any:
        if not self._store.is_staging():
            logical = (
                self.get(
                    ids=ids,
                    where=where,
                    where_document=where_document,
                    limit=limit,
                ).get("ids")
                or []
            )
            physical = self._physical_ids([str(item) for item in logical])
            return self._collection.delete(ids=physical)
        logical = (
            self.get(
                ids=ids, where=where, where_document=where_document, limit=limit
            ).get("ids")
            or []
        )
        self._store.stage(self._role, "delete", [str(item) for item in logical], [])
        return None

    def get(
        self,
        ids: str | list[str] | None = None,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        where_document: Mapping[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        requested = include or ["metadatas", "documents"]
        raw = self._collection.get(
            include=list({*requested, "metadatas", "documents", "embeddings"})
        )
        rows = self._visible_rows(raw)
        selected_ids = None if ids is None else set(_ids(ids))
        rows = [
            row
            for row in rows
            if (selected_ids is None or row["id"] in selected_ids)
            and _matches_where(row["metadata"], where)
            and _matches_document(row["document"], where_document)
        ]
        if offset:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        return _get_result(rows, requested)

    def query(
        self, n_results: int = 10, include: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        requested = include or ["metadatas", "documents", "distances"]
        where = kwargs.pop("where", None)
        where_document = kwargs.pop("where_document", None)
        raw = self._collection.query(
            n_results=max(1, self._collection.count()),
            include=list({*requested, "metadatas", "documents", "distances"}),
            **kwargs,
        )
        visible_physical = {
            row["physical_id"]: row
            for row in self._visible_rows(
                self._collection.get(include=["metadatas", "documents"])
            )
        }
        result: dict[str, Any] = {"ids": []}
        for field in requested:
            result[field] = []
        query_ids = raw.get("ids") or []
        for query_index, physical_ids in enumerate(query_ids):
            selected: list[tuple[int, dict[str, Any]]] = []
            for item_index, physical_id in enumerate(physical_ids):
                row = visible_physical.get(str(physical_id))
                if row is None or not _matches_where(row["metadata"], where):
                    continue
                if not _matches_document(row["document"], where_document):
                    continue
                selected.append((item_index, row))
                if len(selected) == n_results:
                    break
            result["ids"].append([row["id"] for _, row in selected])
            for field in requested:
                source = (raw.get(field) or [[] for _ in query_ids])[query_index]
                if field == "metadatas":
                    result[field].append([row["metadata"] for _, row in selected])
                elif field == "documents":
                    result[field].append([row["document"] for _, row in selected])
                else:
                    result[field].append([source[index] for index, _ in selected])
        return result

    def count(self) -> int:
        return len(self.get(include=[]).get("ids") or [])

    def _stage_versions(
        self,
        operation: str,
        ids: list[str],
        kwargs: dict[str, Any],
        *,
        existing_only: bool = False,
    ) -> None:
        existing = {
            row["id"]: row
            for row in self._visible_rows(
                self._collection.get(
                    include=["metadatas", "documents", "embeddings", "uris"]
                )
            )
        }
        event = self._store._event_provider()
        assert event is not None and event.processing_sequence is not None
        versions: list[dict[str, Any]] = []
        staged_ids: list[str] = []
        for index, logical_id in enumerate(ids):
            previous = existing.get(logical_id)
            if previous is None and existing_only:
                continue
            metadata_values = _value_at(kwargs.get("metadatas"), index)
            metadata = dict(previous["metadata"] if previous is not None else {})
            if metadata_values is not None:
                metadata.update(dict(metadata_values))
            command_hint = self._store._next_indices.get(
                event.event_id, self._store._existing_command_count(event.event_id)
            )
            physical_id = _version_id(event.event_id, command_hint, index)
            metadata.update(
                {
                    _LOGICAL_ID: logical_id,
                    _EVENT_ID: event.event_id,
                    _COMMAND_INDEX: command_hint,
                    _PROCESSING_SEQUENCE: event.processing_sequence,
                }
            )
            document = _value_at(kwargs.get("documents"), index)
            embedding = _value_at(kwargs.get("embeddings"), index)
            uri = _value_at(kwargs.get("uris"), index)
            if previous is not None:
                document = previous["document"] if document is None else document
                if embedding is None and kwargs.get("documents") is None:
                    embedding = previous["embedding"]
                uri = previous.get("uri") if uri is None else uri
            version: dict[str, Any] = {
                "physical_id": physical_id,
                "metadatas": [metadata],
                "documents": None if document is None else [document],
                "embeddings": None if embedding is None else [embedding],
                "uris": None if uri is None else [uri],
            }
            versions.append(version)
            staged_ids.append(logical_id)
        self._store.stage(self._role, operation, staged_ids, versions)

    def _visible_rows(self, result: Mapping[str, Any]) -> list[dict[str, Any]]:
        operations = self._store.command_operations(self._role)
        selected: dict[str, dict[str, Any]] = {}
        physical_rows: dict[str, dict[str, Any]] = {}
        ids = result.get("ids") or []
        for index, physical_id in enumerate(ids):
            metadata = dict(_value_at(result.get("metadatas"), index) or {})
            logical_id = str(metadata.get(_LOGICAL_ID, physical_id))
            row = {
                "id": logical_id,
                "physical_id": str(physical_id),
                "metadata": {
                    key: value
                    for key, value in metadata.items()
                    if key not in _RESERVED
                },
                "document": _value_at(result.get("documents"), index),
                "embedding": _value_at(result.get("embeddings"), index),
                "uri": _value_at(result.get("uris"), index),
            }
            physical_rows[str(physical_id)] = row
            if _LOGICAL_ID not in metadata:
                selected[logical_id] = row
        for operation in operations:
            if operation["operation"] == "delete":
                for logical_id in operation["ids"]:
                    selected.pop(str(logical_id), None)
                continue
            for physical_id in operation["version_ids"]:
                version_row = physical_rows.get(str(physical_id))
                if version_row is not None:
                    selected[version_row["id"]] = version_row
        return list(selected.values())

    def _physical_ids(self, logical_ids: list[str]) -> list[str]:
        rows = self._visible_rows(
            self._collection.get(include=["metadatas", "documents"])
        )
        by_id = {row["id"]: row["physical_id"] for row in rows}
        return [by_id[item] for item in logical_ids if item in by_id]


def _get_result(rows: list[dict[str, Any]], include: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"ids": [row["id"] for row in rows]}
    mapping = {
        "metadatas": "metadata",
        "documents": "document",
        "embeddings": "embedding",
        "uris": "uri",
    }
    for field in include:
        key = mapping.get(field)
        if key is not None:
            result[field] = [row[key] for row in rows]
    return result


def _matches_where(
    metadata: Mapping[str, Any], where: Mapping[str, Any] | None
) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_matches_where(metadata, item) for item in where["$and"])
    if "$or" in where:
        return any(_matches_where(metadata, item) for item in where["$or"])
    for key, expected in where.items():
        actual = metadata.get(key)
        if isinstance(expected, Mapping):
            for operator, value in expected.items():
                if operator == "$eq" and actual != value:
                    return False
                if operator == "$ne" and actual == value:
                    return False
                if operator == "$in" and actual not in value:
                    return False
                if operator == "$nin" and actual in value:
                    return False
                if operator == "$gt" and not (actual is not None and actual > value):
                    return False
                if operator == "$gte" and not (actual is not None and actual >= value):
                    return False
                if operator == "$lt" and not (actual is not None and actual < value):
                    return False
                if operator == "$lte" and not (actual is not None and actual <= value):
                    return False
        elif actual != expected:
            return False
    return True


def _matches_document(document: Any, where: Mapping[str, Any] | None) -> bool:
    if not where:
        return True
    text = "" if document is None else str(document)
    if "$and" in where:
        return all(_matches_document(text, item) for item in where["$and"])
    if "$or" in where:
        return any(_matches_document(text, item) for item in where["$or"])
    if "$contains" in where:
        return str(where["$contains"]) in text
    if "$not_contains" in where:
        return str(where["$not_contains"]) not in text
    return True


def _value_at(values: Any, index: int) -> Any:
    if values is None:
        return None
    if isinstance(values, (str, Mapping)):
        return values if index == 0 else None
    return values[index] if index < len(values) else None


def _ids(value: str | Sequence[str]) -> list[str]:
    return [value] if isinstance(value, str) else [str(item) for item in value]


def _command_id(event_id: str, index: int) -> str:
    return f"command-{_digest(event_id)}-{index:08d}"


def _version_id(event_id: str, command_index: int, item_index: int) -> str:
    return f"txv-{_digest(event_id)}-{command_index:08d}-{item_index:08d}"


def _marker_id(kind: str, event_id: str) -> str:
    return f"{kind}-{_digest(event_id)}"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _json_value(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _safe_reason(value: str) -> str:
    return (
        value if value.replace("_", "").isalnum() and len(value) <= 128 else "redacted"
    )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
