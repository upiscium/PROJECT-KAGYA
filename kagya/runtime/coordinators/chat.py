"""Explicit chat transaction stages and rollback ownership."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


class ChatStage(StrEnum):
    PREPARE_CONTEXT = "prepare_context"
    APPRAISE_AND_GENERATE = "appraise_and_generate"
    PREPARE_EXTERNAL = "prepare_external"
    INTEGRATE_DOMAINS = "integrate_domains"
    COMMIT_AUTHORITATIVE = "commit_authoritative"


class ChatRollbackScope(StrEnum):
    LOCAL_DOMAIN = "local_domain"
    EXTERNAL_SAGA = "external_saga"


@dataclass(frozen=True)
class ChatTransactionTrace:
    stages: tuple[ChatStage, ...]
    rollback_scopes: tuple[ChatRollbackScope, ...]


class ChatOrchestrationCoordinator(Generic[T]):
    """Runs the local transaction; AgentRuntime owns external saga completion."""

    STAGES = tuple(ChatStage)
    ROLLBACK_SCOPES = (
        ChatRollbackScope.LOCAL_DOMAIN,
        ChatRollbackScope.EXTERNAL_SAGA,
    )

    def __init__(self, execute_local_transaction: Callable[..., T]) -> None:
        self._execute_local_transaction = execute_local_transaction

    def chat(self, *args: object, **kwargs: object) -> T:
        return self._execute_local_transaction(*args, **kwargs)

    @classmethod
    def transaction_contract(cls) -> ChatTransactionTrace:
        return ChatTransactionTrace(cls.STAGES, cls.ROLLBACK_SCOPES)
