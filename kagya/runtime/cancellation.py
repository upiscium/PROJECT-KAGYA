"""Cooperative cancellation primitives for authoritative runtime events."""

from __future__ import annotations

from contextvars import ContextVar
from threading import Event
from typing import Callable


class OperationCanceled(RuntimeError):
    """Raised at a safe checkpoint before an event can commit."""

    def __init__(self, code: str = "client_request") -> None:
        super().__init__("Operation canceled")
        self.code = code


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()
        self._code = "client_request"

    def cancel(self, code: str = "client_request") -> bool:
        if self._event.is_set():
            return False
        self._code = code
        self._event.set()
        return True

    @property
    def is_canceled(self) -> bool:
        return self._event.is_set()

    @property
    def code(self) -> str:
        return self._code

    def raise_if_canceled(self) -> None:
        if self.is_canceled:
            raise OperationCanceled(self.code)


_current_cancellation: ContextVar[CancellationToken | None] = ContextVar(
    "kagya_event_cancellation", default=None
)
_current_finalization_boundary: ContextVar[Callable[[], None] | None] = ContextVar(
    "kagya_event_finalization_boundary", default=None
)


def current_cancellation_token() -> CancellationToken | None:
    return _current_cancellation.get()


def cancellation_checkpoint() -> None:
    token = current_cancellation_token()
    if token is not None:
        token.raise_if_canceled()


def enter_finalization_boundary() -> None:
    """Atomically leave the cancellable phase before authoritative writes."""

    boundary = _current_finalization_boundary.get()
    if boundary is None:
        cancellation_checkpoint()
        return
    boundary()
