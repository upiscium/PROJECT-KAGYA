"""Event-boundary adapter runtime activation and rollback."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from kagya.learning.adapter_registry import AdapterEntry, AdapterRegistry, AdapterStatus
from kagya.models import ModelProvider


@dataclass(frozen=True)
class RuntimeAdapterState:
    adapter_id: str | None
    adapter_hash: str | None
    activation_sequence: int | None
    provider: ModelProvider


@dataclass(frozen=True)
class AdapterActivationRecord:
    action: str
    adapter_id: str | None
    adapter_hash: str | None
    previous_adapter_id: str | None
    previous_adapter_hash: str | None
    activation_sequence: int
    created_at: str
    schema_version: int = 1


class AdapterRuntimeManager:
    def __init__(
        self,
        registry: AdapterRegistry,
        *,
        provider_loader: Callable[[AdapterEntry | None], ModelProvider],
        runtime_switch: Callable[
            [ModelProvider, AdapterEntry | None, int | None], None
        ],
        runtime_snapshot: Callable[[], RuntimeAdapterState],
        history_path: Path,
    ) -> None:
        self.registry = registry
        self.provider_loader = provider_loader
        self.runtime_switch = runtime_switch
        self.runtime_snapshot = runtime_snapshot
        self.history_path = history_path
        self._staged: dict[str, tuple[AdapterEntry, ModelProvider]] = {}
        self._verified_adapter_ids: set[str] = set()
        active = self._active_entry()
        current = runtime_snapshot()
        if not _matches(active, current):
            raise RuntimeError("runtime provider and ACTIVE adapter registry disagree")

    def stage(self, adapter_id: str) -> AdapterEntry:
        entry = self.registry.lookup(adapter_id)
        if entry is None:
            raise ValueError(f"Unknown adapter: {adapter_id}")
        if entry.status != AdapterStatus.APPROVED:
            raise ValueError("Only approved adapters can be staged for activation")
        provider = self.provider_loader(entry)
        self._staged[adapter_id] = (entry, provider)
        self._verified_adapter_ids.discard(adapter_id)
        return entry

    def verify(self, adapter_id: str) -> AdapterEntry:
        staged = self._staged.get(adapter_id)
        if staged is None:
            raise ValueError("Adapter is not staged")
        staged[1].get_processor()
        staged[1].get_model()
        text = staged[1].generate(
            "Adapter activation verification. Reply with one short sentence."
        )
        if not text.strip():
            raise RuntimeError("staged adapter generated empty verification output")
        if getattr(staged[1], "last_fallback_used", False):
            raise RuntimeError("staged adapter verification used fallback model")
        self._verified_adapter_ids.add(adapter_id)
        return staged[0]

    def activate_at_event_boundary(self, adapter_id: str) -> AdapterActivationRecord:
        event = _activation_event()
        if (
            adapter_id not in self._staged
            or adapter_id not in self._verified_adapter_ids
        ):
            raise ValueError("Adapter is not staged and verified")
        entry, staged_provider = self._staged[adapter_id]
        previous = self.runtime_snapshot()
        previous_entry = (
            None
            if previous.adapter_id is None
            else self.registry.lookup(previous.adapter_id)
        )
        try:
            def authoritative_switch(fresh: AdapterEntry) -> None:
                self.runtime_switch(
                    staged_provider, fresh, event.processing_sequence
                )
                switched = self.runtime_snapshot()
                if (
                    switched.provider is not staged_provider
                    or switched.adapter_id != fresh.adapter_id
                    or switched.adapter_hash != fresh.adapter_hash
                    or switched.activation_sequence != event.processing_sequence
                ):
                    raise RuntimeError("runtime did not adopt the staged adapter")

            activated = self.registry.activate(
                adapter_id,
                activation_sequence=event.processing_sequence,
                loaded_adapter_manifest_hash=getattr(
                    staged_provider, "adapter_artifact_manifest_hash", None
                ),
                loaded_adapter_manifest=getattr(
                    staged_provider, "adapter_artifact_manifest", None
                ),
                runtime_switch=authoritative_switch,
            )
        except Exception:
            self.runtime_switch(
                previous.provider, previous_entry, previous.activation_sequence
            )
            raise
        record = AdapterActivationRecord(
            action="activate",
            adapter_id=activated.adapter_id,
            adapter_hash=activated.adapter_hash,
            previous_adapter_id=previous.adapter_id,
            previous_adapter_hash=previous.adapter_hash,
            activation_sequence=event.processing_sequence,
            created_at=_now(),
        )
        self._append(record)
        self._staged.pop(adapter_id, None)
        self._verified_adapter_ids.discard(adapter_id)
        return record

    def rollback(self) -> AdapterActivationRecord:
        event = _activation_event()
        current = self.runtime_snapshot()
        previous_id = self._rollback_target(current.adapter_id)
        previous_entry = (
            None if previous_id is None else self.registry.lookup(previous_id)
        )
        if previous_id is not None and previous_entry is None:
            raise ValueError(f"Unknown rollback adapter: {previous_id}")
        provider = self.provider_loader(previous_entry)
        provider.get_processor()
        provider.get_model()
        text = provider.generate("Adapter rollback verification. Reply briefly.")
        if not text.strip():
            raise RuntimeError("rollback provider generated empty verification output")
        if getattr(provider, "last_fallback_used", False):
            raise RuntimeError("rollback verification used fallback model")
        current_entry = (
            None
            if current.adapter_id is None
            else self.registry.lookup(current.adapter_id)
        )
        self.runtime_switch(provider, previous_entry, event.processing_sequence)
        try:
            switched = self.runtime_snapshot()
            if (
                switched.provider is not provider
                or switched.adapter_id != previous_id
                or switched.adapter_hash
                != (None if previous_entry is None else previous_entry.adapter_hash)
                or switched.activation_sequence != event.processing_sequence
            ):
                raise RuntimeError("runtime did not adopt the rollback provider")
            restored = self.registry.restore_active(
                previous_id, activation_sequence=event.processing_sequence
            )
        except Exception:
            self.runtime_switch(
                current.provider, current_entry, current.activation_sequence
            )
            raise
        record = AdapterActivationRecord(
            action="rollback",
            adapter_id=None if restored is None else restored.adapter_id,
            adapter_hash=None if restored is None else restored.adapter_hash,
            previous_adapter_id=current.adapter_id,
            previous_adapter_hash=current.adapter_hash,
            activation_sequence=event.processing_sequence,
            created_at=_now(),
        )
        self._append(record)
        return record

    def report_canary(self, *, success: bool) -> AdapterActivationRecord | None:
        current = self.runtime_snapshot()
        if current.adapter_id is None:
            raise ValueError("No active adapter canary")
        entry = self.registry.record_canary(current.adapter_id, success=success)
        if success:
            return None
        if (
            entry.canary_failures
            < self.registry.settings.adapter_registry.canary_failure_limit
        ):
            return None
        return self.rollback()

    def current(self) -> RuntimeAdapterState:
        state = self.runtime_snapshot()
        active = self._active_entry()
        if not _matches(active, state):
            raise RuntimeError("runtime provider and ACTIVE adapter registry disagree")
        return state

    def history(self, adapter_id: str | None = None) -> list[AdapterActivationRecord]:
        return [
            record
            for record in self._records()
            if adapter_id is None
            or record.adapter_id == adapter_id
            or record.previous_adapter_id == adapter_id
        ]

    def _rollback_target(self, current_adapter_id: str | None) -> str | None:
        records = self._records()
        for record in reversed(records):
            if record.adapter_id == current_adapter_id:
                return record.previous_adapter_id
        raise ValueError("No rollback target is recorded")

    def _active_entry(self) -> AdapterEntry | None:
        return next(
            (
                entry
                for entry in self.registry.list()
                if entry.status == AdapterStatus.ACTIVE
            ),
            None,
        )

    def _append(self, record: AdapterActivationRecord) -> None:
        records = self._records()
        records.append(record)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.history_path.with_name(
            f".{self.history_path.name}.{uuid4()}.tmp"
        )
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(
                {
                    "schema_version": 1,
                    "activations": [asdict(item) for item in records],
                },
                output,
                sort_keys=True,
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self.history_path)

    def _records(self) -> list[AdapterActivationRecord]:
        if not self.history_path.exists():
            return []
        raw: dict[str, Any] = json.loads(self.history_path.read_text("utf-8"))
        return [AdapterActivationRecord(**item) for item in raw.get("activations", [])]


def _activation_event():
    from kagya.runtime.agent_runtime import AgentEventType, current_agent_event

    event = current_agent_event()
    if (
        event is None
        or event.event_type != AgentEventType.ADAPTER_UPDATE
        or event.processing_sequence is None
    ):
        raise RuntimeError("adapter changes require an ADAPTER_UPDATE event boundary")
    return event


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _matches(entry: AdapterEntry | None, state: RuntimeAdapterState) -> bool:
    if entry is None:
        return state.adapter_id is None and state.adapter_hash is None
    return (
        entry.adapter_id == state.adapter_id
        and entry.adapter_hash == state.adapter_hash
        and entry.activation_sequence == state.activation_sequence
    )
