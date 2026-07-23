from pathlib import Path

import pytest

from kagya.config import load_settings
from kagya.memory import DeterministicEmbeddingFunction, DualMemorySystem
from kagya.models import DummyProvider
from kagya.runtime import KagyaMainLoop
from kagya.runtime.coordinators import (
    ChatOrchestrationCoordinator,
    ChatRollbackScope,
    ChatStage,
    ChatTransactionCallbacks,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_chat_transaction_contract_has_stable_stage_and_rollback_order() -> None:
    contract = ChatOrchestrationCoordinator[object].transaction_contract()

    assert contract.stages == (
        ChatStage.PREPARE_CONTEXT,
        ChatStage.APPRAISE_AND_GENERATE,
        ChatStage.PREPARE_EXTERNAL,
        ChatStage.INTEGRATE_DOMAINS,
        ChatStage.COMMIT_AUTHORITATIVE,
    )
    assert contract.rollback_scopes == (
        ChatRollbackScope.LOCAL_DOMAIN,
        ChatRollbackScope.EXTERNAL_SAGA,
    )


def test_chat_coordinator_propagates_failure_to_runtime_rollback_scope() -> None:
    expected = RuntimeError("domain integration failed")
    coordinator = ChatOrchestrationCoordinator[object](
        lambda: (_ for _ in ()).throw(expected)
    )

    with pytest.raises(RuntimeError) as raised:
        coordinator.chat()

    assert raised.value is expected


def test_chat_transaction_runs_stages_in_commit_order() -> None:
    events: list[str] = []
    callbacks = ChatTransactionCallbacks[str](
        prepare_context=lambda: events.append("prepare_context"),
        appraise_and_generate=lambda: events.append("appraise_and_generate"),
        prepare_external=lambda: events.append("prepare_external"),
        integrate_domains=lambda: events.append("integrate_domains"),
        commit_authoritative=lambda: events.append("commit_authoritative") or "ok",
        rollback_local_domain=lambda: events.append("rollback_local_domain"),
        rollback_external_saga=lambda: events.append("rollback_external_saga"),
    )

    assert ChatOrchestrationCoordinator.run_transaction(callbacks) == "ok"
    assert events == [stage.value for stage in ChatStage]


def test_chat_transaction_runs_each_rollback_scope_and_preserves_failure() -> None:
    events: list[str] = []
    expected = RuntimeError("integration failed")

    def fail_integration() -> None:
        events.append("integrate_domains")
        raise expected

    def fail_local_rollback() -> None:
        events.append("local_domain")
        raise RuntimeError("local rollback failed")

    callbacks = ChatTransactionCallbacks[None](
        prepare_context=lambda: events.append("prepare_context"),
        appraise_and_generate=lambda: events.append("appraise_and_generate"),
        prepare_external=lambda: events.append("prepare_external"),
        integrate_domains=fail_integration,
        commit_authoritative=lambda: events.append("commit_authoritative"),
        rollback_local_domain=fail_local_rollback,
        rollback_external_saga=lambda: events.append("external_saga"),
    )

    with pytest.raises(RuntimeError) as raised:
        ChatOrchestrationCoordinator.run_transaction(callbacks)

    assert raised.value is expected
    assert events == [
        "prepare_context",
        "appraise_and_generate",
        "prepare_external",
        "integrate_domains",
        "local_domain",
        "external_saga",
    ]


def test_private_main_loop_owns_only_wiring_and_observability_methods() -> None:
    from kagya.runtime.coordinated_main_loop import _MainLoopImplementation

    owned_methods = {
        name
        for name, value in _MainLoopImplementation.__dict__.items()
        if callable(value)
    }

    assert owned_methods == {"__init__", "_metric"}


def test_main_loop_wires_coordinators_to_authoritative_stores(tmp_path: Path) -> None:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={"persist_directory": tmp_path / "chroma"}
            )
        }
    )
    memory = DualMemorySystem(
        settings, embedding_function=DeterministicEmbeddingFunction()
    )

    loop = KagyaMainLoop(settings, DummyProvider(), memory)

    assert loop.motivation_coordinator.list_active_goal_ids() == ()
    assert loop.chat_coordinator.transaction_contract().stages[-1] is (
        ChatStage.COMMIT_AUTHORITATIVE
    )
