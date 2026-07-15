"""Periodic producer for serialized emotion recovery events."""

from threading import Event, Thread
from typing import Callable

from kagya.runtime.agent_runtime import (
    AgentEventType,
    AgentRuntime,
    AgentRuntimeQueueFull,
    AgentRuntimeStopped,
)


class EmotionTimer:
    def __init__(
        self,
        runtime: AgentRuntime,
        handler: Callable[[float], object],
        *,
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.runtime = runtime
        self.handler = handler
        self.interval_seconds = interval_seconds
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = Thread(
            target=self._run,
            name="kagya-emotion-timer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.runtime.submit(
                    AgentEventType.EMOTION_TICK,
                    source="runtime.emotion_timer",
                    handler=lambda: self.handler(self.interval_seconds),
                    payload={"elapsed_seconds": self.interval_seconds},
                )
            except (AgentRuntimeQueueFull, AgentRuntimeStopped):
                continue
