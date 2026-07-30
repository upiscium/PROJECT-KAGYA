from datetime import UTC, datetime
from pathlib import Path

from kagya.api.observability import OperationalTelemetry
from kagya.runtime import AgentEvent, AgentEventType


def _event(
    index: int,
    *,
    private_ids: bool = False,
    event_type: AgentEventType = AgentEventType.CHAT,
) -> AgentEvent:
    now = datetime.now(UTC)
    suffix = f"private prompt {index}" if private_ids else f"event-{index}"
    return AgentEvent(
        event_id=suffix,
        event_type=event_type,
        source="test",
        observed_at=now,
        requested_at=now,
        processing_sequence=index,
        correlation_id=f"secret correlation {index}" if private_ids else "trace-group",
        causation_id=f"hidden thought {index}" if private_ids else None,
    )


def test_metrics_survive_restart_and_labels_are_bounded(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    traces_path = tmp_path / "traces.json"
    telemetry = OperationalTelemetry(metrics_path, traces_path, max_series=32)
    for index in range(1000):
        telemetry.counter(
            "kagya_provider_fallback_total", provider=f"untrusted-secret-{index}"
        )

    assert telemetry.series_count == 1
    assert "untrusted-secret" not in metrics_path.read_text(encoding="utf-8")
    restarted = OperationalTelemetry(metrics_path, traces_path, max_series=32)
    exported = restarted.prometheus_text()
    assert 'provider="other"' in exported
    assert exported.endswith(" 1000\n")


def test_trace_retention_is_bounded_and_private_values_are_hashed(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics.json"
    traces_path = tmp_path / "traces.json"
    telemetry = OperationalTelemetry(
        metrics_path, traces_path, max_series=32, max_traces=10
    )
    for index in range(12):
        event = _event(index, private_ids=True)
        telemetry.event_accepted(event, 1)
        telemetry.event_started(event, 0)
        telemetry.event_finished(event, "success", 0)

    records = telemetry.recent_traces(100)
    serialized = traces_path.read_text(encoding="utf-8")
    assert len(records) == 10
    assert records[0].processing_sequence == 2
    assert "private prompt" not in serialized
    assert "hidden thought" not in serialized
    assert "secret correlation" not in serialized
    assert all(record.event_id.startswith("sha256:") for record in records)
    assert len(telemetry.otlp_json()["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 10


def test_corrupt_observability_history_does_not_block_restart(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    traces_path = tmp_path / "traces.json"
    metrics_path.write_text("not-json", encoding="utf-8")
    traces_path.write_text("[] trailing", encoding="utf-8")

    telemetry = OperationalTelemetry(metrics_path, traces_path)
    telemetry.gauge("kagya_agent_queue_depth", 0.0)

    assert "kagya_agent_queue_depth" in telemetry.prometheus_text()
    assert telemetry.recent_traces() == []


def test_context_read_has_bounded_observability_taxonomy(tmp_path: Path) -> None:
    telemetry = OperationalTelemetry(tmp_path / "metrics.json", tmp_path / "traces.json")
    event = _event(1, event_type=AgentEventType.CONTEXT_READ)

    telemetry.event_accepted(event, 1)
    telemetry.event_started(event, 0)
    telemetry.event_finished(event, "success", 0)

    exported = telemetry.prometheus_text()
    assert 'event_type="context_read"' in exported
    assert 'subsystem="runtime"' in exported
    assert 'event_type="other"' not in exported
