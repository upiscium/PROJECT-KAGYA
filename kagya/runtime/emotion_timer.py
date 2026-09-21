"""Process-local producer for idle emotion recovery events."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
import math
from threading import Event, Lock, Thread, current_thread
from typing import Callable

from kagya.runtime.agent_runtime import (
    AgentEventOutcome,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeAdmissionBlocked,
    AgentRuntimeDurabilityError,
    AgentRuntimeQueueFull,
    AgentRuntimeStopped,
)


class EmotionTimer:
    """Submit one serialized emotion tick after each interval.

    The timer owns only wake-up scheduling.  The runtime owns admission,
    ordering, processing sequence, handler execution, and durability.
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        interval_seconds: float,
        handler: Callable[[], None],
    ) -> None:
        if not callable(getattr(runtime, "submit", None)):
            raise TypeError("runtime must provide submit")
        if not callable(handler):
            raise TypeError("handler must be callable")
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0.0
        ):
            raise ValueError("interval_seconds must be finite and greater than zero")
        self._runtime = runtime
        self._interval_seconds = float(interval_seconds)
        self._handler = handler
        self._stop_event = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._outstanding = False

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds

    def start(self) -> None:
        """Start the producer once; repeated starts do not create threads."""

        with self._lock:
            if self._thread is not None:
                return
            if self._stop_event.is_set():
                return
            self._thread = Thread(
                target=self._run,
                name="kagya-emotion-timer",
                daemon=True,
            )
            try:
                self._thread.start()
            except BaseException:
                self._thread = None
                raise

    def stop(self) -> None:
        """Stop and join the producer without cancelling accepted runtime work."""

        with self._lock:
            self._stop_event.set()
            thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join()
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            with self._lock:
                if self._stop_event.is_set():
                    return
                if self._outstanding:
                    continue
                self._outstanding = True
                try:
                    future = self._runtime.submit(
                        AgentEventType.EMOTION_TICK,
                        AgentEventSource.RUNTIME_EMOTION_TIMER,
                        self._handler,
                    )
                except (AgentRuntimeQueueFull, AgentRuntimeAdmissionBlocked):
                    self._outstanding = False
                    continue
                except (AgentRuntimeStopped, AgentRuntimeDurabilityError):
                    self._outstanding = False
                    return
                except Exception:
                    self._outstanding = False
                    return
            future.add_done_callback(self._event_completed)

    def _event_completed(self, future: Future[AgentEventOutcome[None]]) -> None:
        try:
            error = future.exception()
        except CancelledError:
            error = None
        self._clear_outstanding()
        if isinstance(error, AgentRuntimeDurabilityError):
            self._stop_event.set()

    def _clear_outstanding(self) -> None:
        with self._lock:
            self._outstanding = False
