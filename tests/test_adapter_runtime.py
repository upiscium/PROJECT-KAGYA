from pathlib import Path
from threading import Event

import pytest

from kagya.config import load_settings
from kagya.learning import (
    AdapterRegistry,
    AdapterRuntimeManager,
    RuntimeAdapterState,
)
from kagya.models import DummyProvider
from kagya.runtime import AgentEventType, AgentRuntime


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_activation_and_rollback_switch_runtime_at_event_boundaries(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    entry = _approved(registry, tmp_path, "adapter-a", "hash-a")
    state = [RuntimeAdapterState(None, None, None, DummyProvider())]
    manager = _manager(registry, state, tmp_path)
    runtime = AgentRuntime(queue_capacity=4)
    runtime.start()

    manager.stage(entry.adapter_id)
    manager.verify(entry.adapter_id)
    activated = runtime.execute(
        AgentEventType.ADAPTER_UPDATE,
        source="test.activate",
        handler=lambda: manager.activate_at_event_boundary(entry.adapter_id),
    ).value
    restored = AdapterRuntimeManager(
        registry,
        provider_loader=lambda entry: DummyProvider(),
        runtime_switch=lambda provider, entry, sequence: state.__setitem__(
            0,
            RuntimeAdapterState(
                None if entry is None else entry.adapter_id,
                None if entry is None else entry.adapter_hash,
                sequence,
                provider,
            ),
        ),
        runtime_snapshot=lambda: state[0],
        history_path=tmp_path / "activations.json",
    )
    rolled_back = runtime.execute(
        AgentEventType.ADAPTER_UPDATE,
        source="test.rollback",
        handler=restored.rollback,
    ).value
    runtime.shutdown()

    assert activated.adapter_id == "adapter-a"
    assert activated.adapter_hash == "hash-a"
    assert activated.activation_sequence == 1
    assert rolled_back.adapter_id is None
    assert rolled_back.activation_sequence == 2
    assert state[0].adapter_id is None
    assert not [item for item in registry.list() if item.status.value == "active"]


def test_activation_load_or_registry_failure_keeps_current_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)
    entry = _approved(registry, tmp_path, "adapter-a", "hash-a")
    base_provider = DummyProvider()
    state = [RuntimeAdapterState(None, None, None, base_provider)]
    manager = _manager(registry, state, tmp_path)

    monkeypatch.setattr(
        registry,
        "activate",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("registry failed")),
    )
    manager.stage(entry.adapter_id)
    manager.verify(entry.adapter_id)
    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    with pytest.raises(OSError, match="registry failed"):
        runtime.execute(
            AgentEventType.ADAPTER_UPDATE,
            source="test.activate",
            handler=lambda: manager.activate_at_event_boundary(entry.adapter_id),
        )
    runtime.shutdown()

    assert state[0].adapter_id is None
    assert state[0].provider is base_provider
    assert registry.lookup(entry.adapter_id).status.value == "approved"


def test_activation_is_rejected_outside_event_boundary(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    entry = _approved(registry, tmp_path, "adapter-a", "hash-a")
    state = [RuntimeAdapterState(None, None, None, DummyProvider())]
    manager = _manager(registry, state, tmp_path)
    manager.stage(entry.adapter_id)
    manager.verify(entry.adapter_id)

    with pytest.raises(RuntimeError, match="event boundary"):
        manager.activate_at_event_boundary(entry.adapter_id)


def test_activation_waits_for_in_flight_event(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    entry = _approved(registry, tmp_path, "adapter-a", "hash-a")
    state = [RuntimeAdapterState(None, None, None, DummyProvider())]
    manager = _manager(registry, state, tmp_path)
    manager.stage(entry.adapter_id)
    manager.verify(entry.adapter_id)
    entered = Event()
    release = Event()
    runtime = AgentRuntime(queue_capacity=4)
    runtime.start()
    running = runtime.submit(
        AgentEventType.CHAT,
        source="test.chat",
        handler=lambda: (entered.set(), release.wait(1)),
    )
    assert entered.wait(1)
    activation = runtime.submit(
        AgentEventType.ADAPTER_UPDATE,
        source="test.activate",
        handler=lambda: manager.activate_at_event_boundary(entry.adapter_id),
    )

    assert state[0].adapter_id is None
    release.set()
    running.result()
    activation.result()
    runtime.shutdown()

    assert state[0].adapter_id == "adapter-a"


def test_concurrent_staging_keeps_each_adapter_provider(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    first = _approved(registry, tmp_path, "adapter-a", "hash-a")
    second = _approved(registry, tmp_path, "adapter-b", "hash-b")
    state = [RuntimeAdapterState(None, None, None, DummyProvider())]
    manager = _manager(registry, state, tmp_path)
    manager.stage(first.adapter_id)
    manager.verify(first.adapter_id)
    manager.stage(second.adapter_id)
    manager.verify(second.adapter_id)
    runtime = AgentRuntime(queue_capacity=4)
    runtime.start()

    first_result = runtime.submit(
        AgentEventType.ADAPTER_UPDATE,
        source="test.activate-a",
        handler=lambda: manager.activate_at_event_boundary(first.adapter_id),
    )
    second_result = runtime.submit(
        AgentEventType.ADAPTER_UPDATE,
        source="test.activate-b",
        handler=lambda: manager.activate_at_event_boundary(second.adapter_id),
    )
    first_result.result()
    second_result.result()
    runtime.shutdown()

    assert state[0].adapter_id == "adapter-b"
    assert registry.lookup("adapter-a").status.value == "archived"
    assert registry.lookup("adapter-b").status.value == "active"


def _manager(registry, state, tmp_path: Path) -> AdapterRuntimeManager:
    def switch(provider, entry, sequence):
        state[0] = RuntimeAdapterState(
            None if entry is None else entry.adapter_id,
            None if entry is None else entry.adapter_hash,
            sequence,
            provider,
        )

    return AdapterRuntimeManager(
        registry,
        provider_loader=lambda entry: DummyProvider(),
        runtime_switch=switch,
        runtime_snapshot=lambda: state[0],
        history_path=tmp_path / "activations.json",
    )


def _registry(tmp_path: Path) -> AdapterRegistry:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"path": tmp_path / "registry.json", "eval_sets": []}
            )
        }
    )
    return AdapterRegistry(settings)


def _approved(
    registry: AdapterRegistry, tmp_path: Path, adapter_id: str, adapter_hash: str
):
    registry.register_candidate(
        adapter_id=adapter_id,
        adapter_path=tmp_path / adapter_id,
        dataset_path=tmp_path / "dataset.jsonl",
        dataset_hash="dataset",
        adapter_hash=adapter_hash,
    )
    registry.apply_evaluation(adapter_id, score=0.9, result_path=tmp_path / "eval")
    return registry.approve(adapter_id)
