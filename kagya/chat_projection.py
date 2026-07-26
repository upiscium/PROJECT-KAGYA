"""Asynchronous publication of deterministic public Chat terminal events."""

from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Protocol


class ProjectionRegistry(Protocol):
    def publish_terminal_projection(self, operation_id: str) -> None: ...

    def mark_projection_failed(self, operation_id: str) -> None: ...

    def pending_projection_ids(self) -> list[str]: ...

    def projection_ids_for_recovery(self) -> list[str]: ...


class ChatProjectionPublisher:
    """Publish replayable terminal bundles outside the AgentRuntime worker."""

    def __init__(
        self, registry: ProjectionRegistry, *, queue_capacity: int = 128
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("projection queue capacity must be positive")
        self.registry = registry
        self._queue: Queue[str] = Queue(maxsize=queue_capacity)
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(
            target=self._run, name="kagya-chat-projection", daemon=True
        )
        self._thread.start()
        for operation_id in self.registry.projection_ids_for_recovery():
            self.publish_terminal(operation_id)

    def publish_terminal(self, operation_id: str) -> None:
        try:
            self._queue.put_nowait(operation_id)
        except Full:
            return

    def shutdown(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError("chat projection publisher did not stop")

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                operation_id = self._queue.get(timeout=0.05)
            except Empty:
                self._refill()
                continue
            try:
                self.registry.publish_terminal_projection(operation_id)
            except Exception:
                try:
                    self.registry.mark_projection_failed(operation_id)
                except Exception:
                    pass
                self._stop.wait(0.05)
            finally:
                self._queue.task_done()
            self._refill()

    def _refill(self) -> None:
        for operation_id in self.registry.pending_projection_ids():
            if self._queue.full():
                return
            self.publish_terminal(operation_id)
