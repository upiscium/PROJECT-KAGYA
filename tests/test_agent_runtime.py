from collections.abc import Callable
from dataclasses import FrozenInstanceError, asdict
from threading import Barrier, Event, Lock, Thread, current_thread
import traceback

import pytest

from kagya.config.schema import Settings
from kagya.runtime import (
    AgentEventOutcome,
    AgentEventSource,
    AgentEventType,
    AgentRuntime as DurableAgentRuntime,
    AgentRuntimeDurabilityError,
    AgentRuntimeDurabilityPhase,
    AgentRuntimeExecutionError,
    AgentRuntimeQueueFull,
    AgentRuntimeStatus,
    AgentRuntimeStopped,
    AgentStateSaveError,
    AgentStateSaveStage,
)


class AgentRuntime(DurableAgentRuntime):
    """Explicitly volatile unit-test runtime unless callbacks are supplied."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, allow_volatile=True, **kwargs)


def test_authoritative_runtime_requires_complete_durability_lifecycle() -> None:
    runtime = DurableAgentRuntime(1)

    with pytest.raises(RuntimeError, match="durability lifecycle"):
        runtime.start()

    def callback(_event) -> None:
        pass

    runtime.configure_durability(
        initial_sequence=4,
        admission_checkpoint=callback,
        started_checkpoint=callback,
        preparation_checkpoint=lambda _event, value: value,
        internal_commit_checkpoint=callback,
        finalization_checkpoint=lambda _event, _evidence: None,
        terminal_completion_checkpoint=lambda _event, _evidence: None,
        failure_checkpoint=callback,
    )
    runtime.start()
    outcome = runtime.submit(
        AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: None
    ).result(timeout=2)
    runtime.shutdown()
    assert outcome.event.processing_sequence == 5


def test_fifo_order_and_consumer_sequences() -> None:
    runtime = AgentRuntime(4)
    runtime.start()
    futures = [
        runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda i=i: i)
        for i in range(4)
    ]

    outcomes = [future.result(timeout=2) for future in futures]
    runtime.shutdown()

    assert [outcome.value for outcome in outcomes] == [0, 1, 2, 3]
    assert [outcome.event.processing_sequence for outcome in outcomes] == [1, 2, 3, 4]
    assert runtime.status is AgentRuntimeStatus.STOPPED


def test_initial_sequence_is_used_for_first_event() -> None:
    runtime = AgentRuntime(1, initial_sequence=7)
    runtime.start()

    outcome = runtime.submit(
        AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: None
    ).result(timeout=2)
    runtime.shutdown()

    assert outcome.event.processing_sequence == 8


def test_checkpoint_runs_before_future_success_is_observable() -> None:
    observations: list[str] = []

    def checkpoint(event: object, _evidence: object) -> None:
        assert event is not None
        observations.append("checkpoint")

    runtime = AgentRuntime(1, terminal_completion_checkpoint=checkpoint)
    runtime.start()
    future = runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: observations.append("handler"),
    )
    future.add_done_callback(lambda _: observations.append("future"))

    assert future.result(timeout=2).value is None
    runtime.shutdown()

    assert observations == ["handler", "checkpoint", "future"]


def test_handler_failure_skips_checkpoint() -> None:
    checkpoint_called = False

    def checkpoint(_: object, _evidence: object) -> None:
        nonlocal checkpoint_called
        checkpoint_called = True

    runtime = AgentRuntime(1, terminal_completion_checkpoint=checkpoint)
    runtime.start()

    def fail() -> None:
        raise ValueError("handler failed")

    future = runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, fail)
    with pytest.raises(AgentRuntimeExecutionError) as error:
        future.result(timeout=2)
    runtime.shutdown()

    assert isinstance(error.value.__cause__, ValueError)
    assert not checkpoint_called


def test_checkpoint_failure_fail_stops_runtime() -> None:
    checkpoint_calls = 0

    def checkpoint(_: object) -> None:
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        if checkpoint_calls == 1:
            raise AgentStateSaveError(
                AgentStateSaveStage.TEMP_FSYNC,
                published=False,
            )

    handler_ran = Event()
    runtime = AgentRuntime(2, internal_commit_checkpoint=checkpoint)
    runtime.start()
    failed = runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        handler_ran.set,
    )
    succeeding = runtime.submit(
        AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: "ok"
    )

    with pytest.raises(AgentRuntimeDurabilityError) as error:
        failed.result(timeout=2)
    with pytest.raises(AgentRuntimeDurabilityError):
        succeeding.result(timeout=2)
    runtime.shutdown()

    assert handler_ran.is_set()
    assert error.value.phase is AgentRuntimeDurabilityPhase.INTERNAL_COMMIT
    assert error.value.failure_type == "AgentStateSaveError"
    assert error.value.published is False
    assert error.value.__cause__ is None
    assert checkpoint_calls == 1
    assert runtime.status is AgentRuntimeStatus.FAILED


def test_cancelled_future_still_runs_checkpoint() -> None:
    blocker_started = Event()
    release_blocker = Event()
    cancelled_handler_ran = Event()
    checkpoint_sequences: list[int] = []
    internal_commit_sequences: list[int] = []
    finalization_sequences: list[int] = []
    accepted_ids: list[str] = []
    started_sequences: list[int] = []

    def checkpoint(event) -> None:
        assert event.processing_sequence is not None
        checkpoint_sequences.append(event.processing_sequence)

    def internal_commit(event) -> None:
        assert event.processing_sequence is not None
        internal_commit_sequences.append(event.processing_sequence)

    def finalization(event, _evidence) -> None:
        assert event.processing_sequence is not None
        finalization_sequences.append(event.processing_sequence)

    def started_checkpoint(event) -> None:
        assert event.processing_sequence is not None
        started_sequences.append(event.processing_sequence)

    runtime = AgentRuntime(
        1,
        admission_checkpoint=lambda event: accepted_ids.append(event.event_id),
        started_checkpoint=started_checkpoint,
        internal_commit_checkpoint=internal_commit,
        finalization_checkpoint=finalization,
        terminal_completion_checkpoint=lambda event, _evidence: checkpoint(event),
    )
    runtime.start()
    blocker = runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: (blocker_started.set(), release_blocker.wait())[1],
    )
    assert blocker_started.wait(timeout=2)
    future = runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        cancelled_handler_ran.set,
    )
    assert future.cancel()
    release_blocker.set()
    assert blocker.result(timeout=2).value is True
    runtime.shutdown()

    assert future.cancelled()
    assert cancelled_handler_ran.is_set()
    assert len(accepted_ids) == 2
    assert started_sequences == [1, 2]
    assert internal_commit_sequences == [1, 2]
    assert finalization_sequences == [1, 2]
    assert checkpoint_sequences == [1, 2]


@pytest.mark.parametrize("initial_sequence", [-1, True, 1.5])
def test_invalid_initial_sequence_is_rejected(initial_sequence: object) -> None:
    with pytest.raises(ValueError):
        AgentRuntime(1, initial_sequence=initial_sequence)  # type: ignore[arg-type]


def test_concurrent_producers_share_one_consumer_and_preserve_local_order() -> None:
    producer_count = 4
    events_per_producer = 5
    event_count = producer_count * events_per_producer
    runtime = AgentRuntime(event_count)
    start_barrier = Barrier(producer_count + 1)
    results_lock = Lock()
    producer_outcomes: dict[int, list[AgentEventOutcome[tuple[int, int, str]]]] = {}
    producer_errors: list[BaseException] = []
    runtime.start()

    def produce(producer_id: int) -> None:
        try:
            start_barrier.wait(timeout=5)
            futures = [
                runtime.submit(
                    AgentEventType.CHAT,
                    AgentEventSource.API_CHAT,
                    lambda local_index=local_index: (
                        producer_id,
                        local_index,
                        current_thread().name,
                    ),
                )
                for local_index in range(events_per_producer)
            ]
            outcomes = [future.result(timeout=5) for future in futures]
            with results_lock:
                producer_outcomes[producer_id] = outcomes
        except BaseException as error:
            with results_lock:
                producer_errors.append(error)

    producers = [
        Thread(target=produce, args=(producer_id,), name=f"producer-{producer_id}")
        for producer_id in range(producer_count)
    ]
    for producer in producers:
        producer.start()
    start_barrier.wait(timeout=5)
    for producer in producers:
        producer.join(timeout=5)
    runtime.shutdown()

    assert not producer_errors
    assert all(not producer.is_alive() for producer in producers)
    assert set(producer_outcomes) == set(range(producer_count))

    all_outcomes = [
        outcome for outcomes in producer_outcomes.values() for outcome in outcomes
    ]
    sequences: list[int] = []
    for outcome in all_outcomes:
        sequence = outcome.event.processing_sequence
        assert sequence is not None
        sequences.append(sequence)

    assert sorted(sequences) == list(range(1, event_count + 1))
    consumer_threads = {outcome.value[2] for outcome in all_outcomes}
    assert consumer_threads == {"kagya-agent-runtime"}
    assert consumer_threads.isdisjoint({producer.name for producer in producers})

    for producer_id, outcomes in producer_outcomes.items():
        local_indexes = [outcome.value[1] for outcome in outcomes]
        local_sequences = [outcome.event.processing_sequence for outcome in outcomes]
        assert all(outcome.value[0] == producer_id for outcome in outcomes)
        assert local_indexes == list(range(events_per_producer))
        assert local_sequences == sorted(local_sequences)


def test_full_queue_is_rejected_without_waiting() -> None:
    runtime = AgentRuntime(1)
    started = Event()
    release = Event()
    rejected_handler_ran = Event()
    runtime.start()

    def block() -> None:
        started.set()
        release.wait()

    first = runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, block)
    assert started.wait(timeout=2)
    second = runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: 2)
    with pytest.raises(AgentRuntimeQueueFull) as error:
        runtime.submit(
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            rejected_handler_ran.set,
        )
    assert error.value.event.processing_sequence is None
    assert not rejected_handler_ran.is_set()
    release.set()
    assert first.result(timeout=2).value is None
    assert second.result(timeout=2).value == 2
    runtime.shutdown()
    assert not rejected_handler_ran.is_set()


def test_handler_failure_preserves_cause_and_consumer_isolated() -> None:
    runtime = AgentRuntime(2)
    runtime.start()

    def fail() -> None:
        raise ValueError("private handler details")

    failed = runtime.submit(
        AgentEventType.DEBUG_CHAT, AgentEventSource.API_CHAT_DEBUG, fail
    )
    succeeding = runtime.submit(
        AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: "ok"
    )
    with pytest.raises(AgentRuntimeExecutionError) as error:
        failed.result(timeout=2)
    assert isinstance(error.value.__cause__, ValueError)
    assert "private handler details" not in str(error.value)
    assert succeeding.result(timeout=2).value == "ok"
    runtime.shutdown()


def test_shutdown_drains_and_rejects_new_work() -> None:
    runtime = AgentRuntime(2)
    started = Event()
    release = Event()
    runtime.start()
    future = runtime.submit(
        AgentEventType.SLEEP,
        AgentEventSource.API_SLEEP_RUN,
        lambda: (started.set(), release.wait(), 4)[2],
    )
    assert started.wait(timeout=2)
    shutdown_thread = Thread(target=runtime.shutdown)
    shutdown_thread.start()
    while runtime.status is AgentRuntimeStatus.ACCEPTING:
        pass
    assert runtime.status is AgentRuntimeStatus.DRAINING
    with pytest.raises(AgentRuntimeStopped):
        runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: 5)
    release.set()
    shutdown_thread.join(timeout=2)
    assert not shutdown_thread.is_alive()
    assert future.result(timeout=2).value == 4
    with pytest.raises(AgentRuntimeStopped) as error:
        runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: 5)
    assert error.value.event.processing_sequence is None


def test_cancelled_future_does_not_cancel_handler() -> None:
    runtime = AgentRuntime(1)
    started = Event()
    release = Event()
    accepted_handler_ran = Event()
    runtime.start()
    blocker = runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: (started.set(), release.wait(), "released")[2],
    )
    assert started.wait(timeout=2)
    future = runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: accepted_handler_ran.set(),
    )
    assert future.cancel()
    release.set()
    assert blocker.result(timeout=2).value == "released"
    runtime.shutdown()
    assert future.cancelled()
    assert accepted_handler_ran.is_set()


def test_event_is_immutable_and_has_no_handler_payload() -> None:
    private_sentinel = "PRIVATE-SENTINEL-R02"
    runtime = AgentRuntime(1)
    runtime.start()
    future = runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: {"prompt": private_sentinel, "hidden_thought": private_sentinel},
    )
    event = future.result(timeout=2).event
    runtime.shutdown()
    assert set(asdict(event)) == {
        "event_id",
        "event_type",
        "source",
        "requested_at",
        "processing_sequence",
    }
    assert private_sentinel not in str(asdict(event))
    assert "prompt" not in str(asdict(event))
    assert "hidden_thought" not in str(asdict(event))
    with pytest.raises(FrozenInstanceError):
        event.source = "changed"  # type: ignore[misc]


def test_arbitrary_source_cannot_be_used_as_private_metadata() -> None:
    runtime = AgentRuntime(1)
    runtime.start()
    with pytest.raises(TypeError):
        runtime.submit(
            AgentEventType.CHAT,
            "PRIVATE-SENTINEL-R02",  # type: ignore[arg-type]
            lambda: None,
        )
    runtime.shutdown()


def test_arbitrary_event_type_cannot_be_used_as_private_metadata() -> None:
    runtime = AgentRuntime(1)
    runtime.start()
    with pytest.raises(TypeError):
        runtime.submit(
            "PRIVATE-SENTINEL-R02",  # type: ignore[arg-type]
            AgentEventSource.API_CHAT,
            lambda: None,
        )
    runtime.shutdown()


@pytest.mark.parametrize("capacity", [0, -1, True])
def test_invalid_capacity(capacity: int) -> None:
    with pytest.raises(ValueError):
        AgentRuntime(capacity)


def test_runtime_config_defaults_and_yaml_value() -> None:
    field = Settings.model_fields["runtime"]
    assert field.default_factory is not None
    assert field.default_factory().queue_capacity == 64


def test_processing_sequence_is_process_local_to_each_runtime() -> None:
    sequences = []
    for _ in range(2):
        runtime = AgentRuntime(1)
        runtime.start()
        outcome = runtime.submit(
            AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: None
        ).result(timeout=2)
        runtime.shutdown()
        sequences.append(outcome.event.processing_sequence)

    assert sequences == [1, 1]


def test_concurrent_durable_acceptance_order_equals_execution_order() -> None:
    accepted: list[str] = []
    executed: list[int] = []
    event_ids: dict[int, str] = {}
    errors: list[BaseException] = []
    lock = Lock()
    barrier = Barrier(9)

    def record_accepted(event) -> None:
        accepted.append(event.event_id)

    runtime = AgentRuntime(8, admission_checkpoint=record_accepted)
    runtime.start()

    def producer(number: int) -> None:
        try:
            barrier.wait(timeout=5)
            outcome = runtime.submit(
                AgentEventType.CHAT,
                AgentEventSource.API_CHAT,
                lambda: (executed.append(number), number)[1],
            ).result(timeout=5)
            with lock:
                event_ids[number] = outcome.event.event_id
        except BaseException as error:
            with lock:
                errors.append(error)

    threads = [Thread(target=producer, args=(number,)) for number in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)
    runtime.shutdown()

    assert not errors
    assert accepted == [event_ids[number] for number in executed]


def test_queue_full_and_stopped_create_no_durable_acceptance() -> None:
    accepted: list[str] = []
    started = Event()
    release = Event()
    runtime = AgentRuntime(
        1, admission_checkpoint=lambda event: accepted.append(event.event_id)
    )
    runtime.start()
    first = runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: (started.set(), release.wait())[1],
    )
    assert started.wait(timeout=2)
    second = runtime.submit(
        AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: None
    )
    with pytest.raises(AgentRuntimeQueueFull):
        runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: None)
    assert len(accepted) == 2
    release.set()
    first.result(timeout=2)
    second.result(timeout=2)
    runtime.shutdown()
    with pytest.raises(AgentRuntimeStopped):
        runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: None)
    assert len(accepted) == 2


def test_admission_failure_does_not_enqueue_and_fail_stops() -> None:
    handler_ran = Event()

    def fail_admission(_event) -> None:
        raise OSError("PRIVATE-SENTINEL-R05")

    runtime = AgentRuntime(1, admission_checkpoint=fail_admission)
    runtime.start()

    with pytest.raises(AgentRuntimeDurabilityError) as error:
        runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, handler_ran.set)
    runtime.shutdown()

    assert error.value.phase is AgentRuntimeDurabilityPhase.ADMISSION
    assert error.value.outcome_indeterminate is False
    assert not handler_ran.is_set()
    assert runtime.status is AgentRuntimeStatus.FAILED
    rendered = "".join(traceback.format_exception(error.value))
    assert "PRIVATE-SENTINEL-R05" not in rendered
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_later_admission_failure_does_not_abandon_started_event() -> None:
    admissions = 0
    handler_started = Event()
    release_handler = Event()
    completed: list[int] = []

    def admission(_event) -> None:
        nonlocal admissions
        admissions += 1
        if admissions == 2:
            raise OSError("journal unavailable")

    def completion(event, _evidence) -> None:
        assert event.processing_sequence is not None
        completed.append(event.processing_sequence)

    runtime = AgentRuntime(
        1,
        admission_checkpoint=admission,
        terminal_completion_checkpoint=completion,
    )
    runtime.start()
    active = runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: (handler_started.set(), release_handler.wait(), "done")[2],
    )
    assert handler_started.wait(timeout=2)

    with pytest.raises(AgentRuntimeDurabilityError):
        runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: None)
    release_handler.set()

    assert active.result(timeout=2).value == "done"
    runtime.shutdown()
    assert completed == [1]
    assert runtime.status is AgentRuntimeStatus.FAILED


def test_started_is_durable_before_handler_and_started_failure_runs_no_handler() -> (
    None
):
    order: list[str] = []
    runtime = AgentRuntime(
        1,
        started_checkpoint=lambda _event: order.append("started"),
    )
    runtime.start()
    runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: order.append("handler"),
    ).result(timeout=2)
    runtime.shutdown()
    assert order == ["started", "handler"]

    handler_ran = Event()

    def fail_started(_event) -> None:
        raise OSError("storage unavailable")

    failed_runtime = AgentRuntime(1, started_checkpoint=fail_started)
    failed_runtime.start()
    future = failed_runtime.submit(
        AgentEventType.CHAT, AgentEventSource.API_CHAT, handler_ran.set
    )
    with pytest.raises(AgentRuntimeDurabilityError) as error:
        future.result(timeout=2)
    failed_runtime.shutdown()
    assert error.value.phase is AgentRuntimeDurabilityPhase.STARTED
    assert not handler_ran.is_set()
    assert failed_runtime.status is AgentRuntimeStatus.FAILED


def test_full_success_protocol_precedes_future_success() -> None:
    order: list[str] = []
    runtime = AgentRuntime(
        1,
        admission_checkpoint=lambda _event: order.append("accepted"),
        started_checkpoint=lambda _event: order.append("started"),
        preparation_checkpoint=lambda _event, value: (
            order.append("transaction_preparation"),
            value,
        )[1],
        internal_commit_checkpoint=lambda _event: order.append("internal_commit"),
        finalization_checkpoint=lambda _event, _evidence: order.append("finalization"),
        terminal_completion_checkpoint=lambda _event, _evidence: order.append(
            "terminal_completion"
        ),
    )
    runtime.start()
    future = runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: order.append("handler"),
    )
    future.add_done_callback(lambda _future: order.append("future"))
    future.result(timeout=2)
    runtime.shutdown()
    assert order == [
        "accepted",
        "started",
        "handler",
        "transaction_preparation",
        "internal_commit",
        "finalization",
        "terminal_completion",
        "future",
    ]


def test_preparation_separates_handler_result_and_hands_internal_proof_forward() -> (
    None
):
    private_result = object()
    proof = object()
    observed_proofs: list[object] = []

    def prepare(_event: object, value: object) -> str:
        assert value is private_result
        return "public"

    runtime = AgentRuntime(
        1,
        preparation_checkpoint=prepare,
        internal_commit_checkpoint=lambda _event: proof,
        finalization_checkpoint=lambda _event, evidence: observed_proofs.append(
            evidence
        ),
        terminal_completion_checkpoint=lambda _event, evidence: observed_proofs.append(
            evidence
        ),
    )
    runtime.start()

    outcome = runtime.submit(
        AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: private_result
    ).result(timeout=2)
    runtime.shutdown()

    assert outcome.value == "public"
    assert observed_proofs == [proof, proof]


def test_success_phases_have_exact_future_observable_order() -> None:
    order: list[str] = []
    runtime = AgentRuntime(
        1,
        internal_commit_checkpoint=lambda _event: order.append("internal_commit"),
        finalization_checkpoint=lambda _event, _evidence: order.append("finalization"),
        terminal_completion_checkpoint=lambda _event, _evidence: order.append(
            "terminal_completion"
        ),
    )
    runtime.start()
    future = runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: order.append("handler"),
    )
    future.add_done_callback(lambda _future: order.append("future"))

    assert future.result(timeout=2).value is None
    runtime.shutdown()

    assert order == [
        "handler",
        "internal_commit",
        "finalization",
        "terminal_completion",
        "future",
    ]


def test_future_is_not_done_while_terminal_completion_is_blocked() -> None:
    terminal_started = Event()
    release_terminal = Event()

    def terminal_completion(_event: object, _evidence: object) -> None:
        terminal_started.set()
        assert release_terminal.wait(timeout=2)

    runtime = AgentRuntime(1, terminal_completion_checkpoint=terminal_completion)
    runtime.start()
    future = runtime.submit(
        AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: "complete"
    )

    assert terminal_started.wait(timeout=2)
    assert not future.done()
    release_terminal.set()
    assert future.result(timeout=2).value == "complete"
    runtime.shutdown()


@pytest.mark.parametrize(
    ("failed_phase", "callback_name"),
    [
        (
            AgentRuntimeDurabilityPhase.TRANSACTION_PREPARATION,
            "transaction_preparation",
        ),
        (AgentRuntimeDurabilityPhase.INTERNAL_COMMIT, "internal_commit"),
        (AgentRuntimeDurabilityPhase.FINALIZATION, "finalization"),
        (AgentRuntimeDurabilityPhase.TERMINAL_COMPLETION, "terminal_completion"),
    ],
)
def test_each_success_phase_failure_fail_stops_and_skips_later_phases(
    failed_phase: AgentRuntimeDurabilityPhase, callback_name: str
) -> None:
    calls: list[str] = []
    failure_checkpoint_called = False

    def phase_callback(name: str) -> Callable[..., None]:
        def callback(*_args: object) -> None:
            calls.append(name)
            if name == callback_name:
                raise OSError(f"{name} failed")

        return callback

    def failure_checkpoint(_event: object) -> None:
        nonlocal failure_checkpoint_called
        failure_checkpoint_called = True

    runtime = AgentRuntime(
        2,
        preparation_checkpoint=phase_callback("transaction_preparation"),
        internal_commit_checkpoint=phase_callback("internal_commit"),
        finalization_checkpoint=phase_callback("finalization"),
        terminal_completion_checkpoint=phase_callback("terminal_completion"),
        failure_checkpoint=failure_checkpoint,
    )
    runtime.start()
    handler_started = Event()
    release_handler = Event()
    failed = runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: (handler_started.set(), release_handler.wait())[1],
    )
    assert handler_started.wait(timeout=2)
    later = runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: None)
    release_handler.set()

    with pytest.raises(AgentRuntimeDurabilityError) as error:
        failed.result(timeout=2)
    with pytest.raises(AgentRuntimeDurabilityError):
        later.result(timeout=2)
    runtime.shutdown()

    assert error.value.phase is failed_phase
    phase_order = [
        "transaction_preparation",
        "internal_commit",
        "finalization",
        "terminal_completion",
    ]
    assert calls == phase_order[: phase_order.index(callback_name) + 1]
    assert not failure_checkpoint_called
    assert runtime.status is AgentRuntimeStatus.FAILED


def test_processing_sequences_are_unchanged_by_success_phases() -> None:
    def callback(_event, _evidence) -> None:
        pass

    runtime = AgentRuntime(
        2,
        internal_commit_checkpoint=lambda _event: None,
        finalization_checkpoint=callback,
        terminal_completion_checkpoint=callback,
    )
    runtime.start()
    futures = [
        runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: None)
        for _ in range(2)
    ]
    outcomes = [future.result(timeout=2) for future in futures]
    runtime.shutdown()

    assert [outcome.event.processing_sequence for outcome in outcomes] == [1, 2]


def test_handler_failure_checkpoint_consumes_sequence_and_runtime_continues() -> None:
    failed_sequences: list[int] = []

    def record_failure(event) -> None:
        assert event.processing_sequence is not None
        failed_sequences.append(event.processing_sequence)

    runtime = AgentRuntime(2, failure_checkpoint=record_failure)
    runtime.start()

    def fail() -> None:
        raise ValueError("domain failure")

    failed = runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, fail)
    succeeding = runtime.submit(
        AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: "ok"
    )
    with pytest.raises(AgentRuntimeExecutionError) as error:
        failed.result(timeout=2)
    outcome = succeeding.result(timeout=2)
    runtime.shutdown()

    assert isinstance(error.value.__cause__, ValueError)
    assert failed_sequences == [1]
    assert outcome.event.processing_sequence == 2
    assert runtime.status is AgentRuntimeStatus.STOPPED


def test_handler_failure_checkpoint_failure_stops_later_handlers() -> None:
    later_ran = Event()

    def fail_handler_checkpoint(_event) -> None:
        raise OSError("failed evidence unavailable")

    runtime = AgentRuntime(2, failure_checkpoint=fail_handler_checkpoint)
    runtime.start()

    def fail() -> None:
        raise ValueError("domain failure")

    failed = runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, fail)
    later = runtime.submit(
        AgentEventType.CHAT, AgentEventSource.API_CHAT, later_ran.set
    )
    with pytest.raises(AgentRuntimeDurabilityError) as error:
        failed.result(timeout=2)
    with pytest.raises(AgentRuntimeDurabilityError):
        later.result(timeout=2)
    runtime.shutdown()

    assert error.value.phase is AgentRuntimeDurabilityPhase.HANDLER_FAILURE
    assert error.value.outcome_indeterminate is True
    assert not later_ran.is_set()
    assert runtime.status is AgentRuntimeStatus.FAILED
