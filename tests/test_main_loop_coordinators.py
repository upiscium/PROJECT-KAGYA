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
