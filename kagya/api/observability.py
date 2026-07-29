"""Privacy-safe, dependency-free operational metrics and tracing."""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
from itertools import count
import json
import math
import os
from pathlib import Path
import re
from threading import RLock
import time
from typing import Any
from uuid import uuid4

from kagya.api.redaction import redact_private_fields


_ids = count(1)


@dataclass(frozen=True)
class RuntimeEvent:
    id: int
    timestamp: str
    category: str
    event_type: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RuntimeEventLog:
    """Small bounded event buffer for recent operator-visible runtime events."""

    def __init__(self, max_events: int = 100) -> None:
        self._events: deque[RuntimeEvent] = deque(maxlen=max_events)

    def record(
        self,
        *,
        category: str,
        event_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        event = RuntimeEvent(
            id=next(_ids),
            timestamp=datetime.now(UTC).isoformat(),
            category=category,
            event_type=event_type,
            message=message,
            metadata=redact_private_fields(metadata or {}),
        )
        self._events.append(event)
        return event

    def recent(self, limit: int = 50) -> list[RuntimeEvent]:
        return list(self._events)[-limit:]


@dataclass(frozen=True)
class TraceRecord:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    event_id: str
    correlation_id: str | None
    causation_id: str | None
    event_type: str
    subsystem: str
    status: str
    processing_sequence: int | None
    started_at: str
    ended_at: str
    duration_seconds: float


@dataclass
class _Series:
    metric_type: str
    labels: dict[str, str]
    value: float = 0.0
    count: int = 0
    total: float = 0.0


_METRICS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "kagya_agent_events_total": (
        "counter", ("subsystem", "event_type", "status"),
        "Completed subject events by bounded lifecycle category.",
    ),
    "kagya_agent_queue_depth": (
        "gauge", (), "Current accepted subject events awaiting completion.",
    ),
    "kagya_agent_queue_wait_seconds": (
        "summary", ("event_type",), "Time accepted events waited before execution.",
    ),
    "kagya_agent_event_duration_seconds": (
        "summary", ("subsystem", "event_type"), "Subject event execution duration.",
    ),
    "kagya_generation_duration_seconds": (
        "summary", ("provider", "fallback"), "Model generation duration.",
    ),
    "kagya_generation_tokens_per_second": (
        "summary", ("provider", "fallback"), "Approximate visible output token rate.",
    ),
    "kagya_provider_fallback_total": (
        "counter", ("provider",), "Model provider fallback activations.",
    ),
    "kagya_memory_retrieval_duration_seconds": (
        "summary", (), "Dual-memory retrieval duration.",
    ),
    "kagya_memory_relevance": (
        "summary", ("tier",), "Retrieved-memory semantic relevance.",
    ),
    "kagya_memory_quarantine_total": (
        "counter", ("reason",), "Generated memories quarantined by health category.",
    ),
    "kagya_working_memory_evictions_total": (
        "counter", ("reason",), "Working-memory evictions by bounded reason.",
    ),
    "kagya_working_memory_items": (
        "gauge", (), "Current working-memory item count.",
    ),
    "kagya_attention_focus_items": (
        "gauge", (), "Items selected into current attention focus.",
    ),
    "kagya_active_goals": (
        "gauge", (), "Current active goal count; not a throughput measure.",
    ),
    "kagya_unresolved_decisions": (
        "gauge", (), "Current decisions awaiting outcome; not a speed measure.",
    ),
    "kagya_autonomy_cycles_total": (
        "counter", ("result",), "Autonomy cycles by bounded result category.",
    ),
    "kagya_autonomy_wakeups_total": (
        "counter", ("outcome",), "Autonomy wake-ups processed or safely deferred.",
    ),
    "kagya_autonomy_pending_wakeups": (
        "gauge", (), "Current pending persistent and derived wake-ups.",
    ),
    "kagya_autonomy_cycle_duration_seconds": (
        "summary", (), "Autonomy cycle wall-clock duration.",
    ),
    "kagya_storage_operation_seconds": (
        "summary", ("component", "operation", "status"),
        "Snapshot, journal, and telemetry persistence duration.",
    ),
    "kagya_chat_job_compactions_total": (
        "counter", ("action",), "Chat job retention actions by bounded category.",
    ),
    "kagya_chat_job_registry_entries": (
        "gauge", ("kind",), "Current full chat job and tombstone entries.",
    ),
    "kagya_chat_job_registry_bytes": (
        "gauge", (), "Current durable chat job registry size in bytes.",
    ),
    "kagya_chat_job_cleanup_duration_seconds": (
        "summary", (), "Chat job registry cleanup duration.",
    ),
    "kagya_process_resident_memory_bytes": (
        "gauge", (), "Process resident memory where supported.",
    ),
    "kagya_accelerator_memory_bytes": (
        "gauge", ("kind",), "Allocated or reserved accelerator memory.",
    ),
}

_SAFE_VALUE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
_LABEL_VALUES: dict[tuple[str, str], set[str]] = {
    ("kagya_agent_events_total", "status"): {"success", "failure"},
    ("kagya_generation_duration_seconds", "fallback"): {"true", "false"},
    ("kagya_generation_tokens_per_second", "fallback"): {"true", "false"},
    ("kagya_memory_relevance", "tier"): {"episodic", "semantic"},
    ("kagya_memory_quarantine_total", "reason"): {"health_check"},
    ("kagya_working_memory_evictions_total", "reason"): {"decay", "capacity"},
    ("kagya_storage_operation_seconds", "component"): {
        "journal", "snapshot", "telemetry",
    },
    ("kagya_storage_operation_seconds", "operation"): {
        "append", "verify", "load", "save",
    },
    ("kagya_storage_operation_seconds", "status"): {"success", "failure"},
    ("kagya_accelerator_memory_bytes", "kind"): {"allocated", "reserved"},
    ("kagya_autonomy_cycles_total", "result"): {
        "processed", "no_action", "budget_exhausted", "stopped",
    },
    ("kagya_autonomy_wakeups_total", "outcome"): {"processed", "deferred"},
    ("kagya_chat_job_compactions_total", "action"): {
        "result_expired", "capacity", "tombstone_expired",
    },
    ("kagya_chat_job_registry_entries", "kind"): {"full", "tombstone"},
}


class OperationalTelemetry:
    """Persistent bounded-cardinality metrics and causal event spans."""

    def __init__(
        self,
        metrics_path: Path,
        traces_path: Path,
        *,
        max_series: int = 512,
        max_traces: int = 1000,
        enabled: bool = True,
    ) -> None:
        self.metrics_path = metrics_path
        self.traces_path = traces_path
        self.max_series = max_series
        self.max_traces = max_traces
        self.enabled = enabled
        self._lock = RLock()
        self._series: dict[str, _Series] = {}
        self._traces: deque[TraceRecord] = deque(maxlen=max_traces)
        self._started: dict[str, tuple[float, str]] = {}
        if enabled:
            self._load()

    def counter(self, name: str, amount: float = 1.0, **labels: str) -> None:
        if amount < 0 or not math.isfinite(amount):
            raise ValueError("counter amount must be finite and non-negative")
        with self._lock:
            series = self._get_series(name, labels, "counter")
            series.value += amount
            self._persist_metrics()

    def gauge(self, name: str, value: float, **labels: str) -> None:
        if not math.isfinite(value):
            return
        with self._lock:
            series = self._get_series(name, labels, "gauge")
            series.value = value
            self._persist_metrics()

    def observe(self, name: str, value: float, **labels: str) -> None:
        if value < 0 or not math.isfinite(value):
            return
        with self._lock:
            series = self._get_series(name, labels, "summary")
            series.count += 1
            series.total += value
            self._persist_metrics()

    def event_accepted(self, event: Any, queue_depth: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._started[event.event_id] = (time.perf_counter(), _iso_now())
        self.gauge("kagya_agent_queue_depth", float(queue_depth))

    def event_started(self, event: Any, queue_depth: int) -> None:
        if not self.enabled:
            return
        wait = max(0.0, (datetime.now(UTC) - event.observed_at).total_seconds())
        self.observe("kagya_agent_queue_wait_seconds", wait, event_type=event.event_type.value)
        self.gauge("kagya_agent_queue_depth", float(queue_depth))

    def event_finished(self, event: Any, status: str, queue_depth: int) -> None:
        if not self.enabled:
            return
        ended = _iso_now()
        with self._lock:
            started_clock, started_at = self._started.pop(
                event.event_id, (time.perf_counter(), event.observed_at.isoformat())
            )
        duration = max(0.0, time.perf_counter() - started_clock)
        subsystem = _subsystem(event.event_type.value)
        safe_status = status if status in {"success", "failure"} else "failure"
        self.counter(
            "kagya_agent_events_total",
            subsystem=subsystem,
            event_type=event.event_type.value,
            status=safe_status,
        )
        self.observe(
            "kagya_agent_event_duration_seconds",
            duration,
            subsystem=subsystem,
            event_type=event.event_type.value,
        )
        self.gauge("kagya_agent_queue_depth", float(queue_depth))
        trace = TraceRecord(
            trace_id=_trace_id(event.correlation_id or event.event_id),
            span_id=_span_id(event.event_id),
            parent_span_id=None if event.causation_id is None else _span_id(event.causation_id),
            event_id=_safe_identifier(event.event_id),
            correlation_id=_safe_optional_identifier(event.correlation_id),
            causation_id=_safe_optional_identifier(event.causation_id),
            event_type=event.event_type.value,
            subsystem=subsystem,
            status=safe_status,
            processing_sequence=event.processing_sequence,
            started_at=started_at,
            ended_at=ended,
            duration_seconds=duration,
        )
        with self._lock:
            self._traces.append(trace)
            self._persist_traces()

    def storage_observation(
        self, component: str, operation: str, status: str, duration: float
    ) -> None:
        self.observe(
            "kagya_storage_operation_seconds",
            duration,
            component=component,
            operation=operation,
            status=status,
        )

    def recent_traces(
        self, limit: int = 100, *, event_id: str | None = None
    ) -> list[TraceRecord]:
        with self._lock:
            records = list(self._traces)
        if event_id is not None:
            records = [record for record in records if record.event_id == event_id]
        return records[-max(0, limit):]

    def prometheus_text(self) -> str:
        lines: list[str] = []
        with self._lock:
            items = sorted(self._series.items())
        names = sorted({key.split("|", 1)[0] for key, _series in items})
        for name in names:
            metric_type, _label_names, help_text = _METRICS[name]
            export_type = "summary" if metric_type == "summary" else metric_type
            lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} {export_type}"))
            for key, series in items:
                if key.split("|", 1)[0] != name:
                    continue
                labels = _prometheus_labels(series.labels)
                if series.metric_type == "summary":
                    lines.append(f"{name}_count{labels} {series.count}")
                    lines.append(f"{name}_sum{labels} {series.total:.12g}")
                else:
                    lines.append(f"{name}{labels} {series.value:.12g}")
        return "\n".join(lines) + ("\n" if lines else "")

    def otlp_json(self) -> dict[str, Any]:
        with self._lock:
            series = list(self._series.items())
            traces = list(self._traces)
        metrics: list[dict[str, Any]] = []
        observed_at = str(time.time_ns())
        for key, item in sorted(series):
            name = key.split("|", 1)[0]
            point: dict[str, Any] = {
                "attributes": [
                    {"key": label, "value": {"stringValue": value}}
                    for label, value in sorted(item.labels.items())
                ],
                "timeUnixNano": observed_at,
            }
            if item.metric_type == "summary":
                point.update({"count": str(item.count), "sum": item.total})
                metrics.append({"name": name, "summary": {"dataPoints": [point]}})
            else:
                point["asDouble"] = item.value
                kind = "sum" if item.metric_type == "counter" else "gauge"
                aggregation: dict[str, Any] = {"dataPoints": [point]}
                if kind == "sum":
                    aggregation.update(
                        {
                            "aggregationTemporality": "AGGREGATION_TEMPORALITY_CUMULATIVE",
                            "isMonotonic": True,
                        }
                    )
                metrics.append({"name": name, kind: aggregation})
        spans = [
            {
                "traceId": record.trace_id,
                "spanId": record.span_id,
                **({"parentSpanId": record.parent_span_id} if record.parent_span_id else {}),
                "name": f"kagya.{record.subsystem}.{record.event_type}",
                "startTimeUnixNano": _unix_nanos(record.started_at),
                "endTimeUnixNano": _unix_nanos(record.ended_at),
                "status": {"code": "STATUS_CODE_OK" if record.status == "success" else "STATUS_CODE_ERROR"},
                "attributes": [
                    {"key": "kagya.event.id", "value": {"stringValue": record.event_id}},
                    {"key": "kagya.event.type", "value": {"stringValue": record.event_type}},
                    {"key": "kagya.subsystem", "value": {"stringValue": record.subsystem}},
                ],
            }
            for record in traces
        ]
        return {
            "resourceMetrics": [{"resource": {"attributes": []}, "scopeMetrics": [{"scope": {"name": "kagya"}, "metrics": metrics}]}],
            "resourceSpans": [{"resource": {"attributes": []}, "scopeSpans": [{"scope": {"name": "kagya"}, "spans": spans}]}],
        }

    @property
    def series_count(self) -> int:
        with self._lock:
            return len(self._series)

    def _get_series(
        self, name: str, labels: dict[str, str], expected_type: str
    ) -> _Series:
        if not self.enabled:
            return _Series(expected_type, labels)
        definition = _METRICS.get(name)
        if definition is None or definition[0] != expected_type:
            raise ValueError(f"unsupported operational metric: {name}")
        if set(labels) != set(definition[1]):
            raise ValueError(f"invalid labels for operational metric: {name}")
        clean = {
            key: _categorical_label(name, key, value) for key, value in labels.items()
        }
        key = name + "|" + json.dumps(clean, sort_keys=True, separators=(",", ":"))
        existing = self._series.get(key)
        if existing is not None:
            return existing
        if len(self._series) >= self.max_series:
            raise RuntimeError("operational metric series limit reached")
        created = _Series(metric_type=expected_type, labels=clean)
        self._series[key] = created
        return created

    def _load(self) -> None:
        if self.metrics_path.exists():
            payload = _read_json(self.metrics_path)
            for key, raw in payload.get("series", {}).items():
                if len(self._series) >= self.max_series:
                    break
                try:
                    name = key.split("|", 1)[0]
                    series = _Series(**raw)
                    definition = _METRICS.get(name)
                    if definition is None or definition[0] != series.metric_type:
                        continue
                    if (
                        not isinstance(series.value, (int, float))
                        or not math.isfinite(series.value)
                        or not isinstance(series.count, int)
                        or series.count < 0
                        or not isinstance(series.total, (int, float))
                        or not math.isfinite(series.total)
                    ):
                        continue
                    if set(series.labels) != set(definition[1]):
                        continue
                    series.labels = {
                        label: _categorical_label(name, label, value)
                        for label, value in series.labels.items()
                    }
                    normalized_key = name + "|" + json.dumps(
                        series.labels, sort_keys=True, separators=(",", ":")
                    )
                    self._series[normalized_key] = series
                except (TypeError, ValueError):
                    continue
        if self.traces_path.exists():
            payload = _read_json(self.traces_path)
            for raw in payload.get("traces", [])[-self.max_traces:]:
                try:
                    self._traces.append(TraceRecord(**raw))
                except (TypeError, ValueError):
                    continue

    def _persist_metrics(self) -> None:
        if not self.enabled:
            return
        payload = {
            "schema_version": 1,
            "updated_at": _iso_now(),
            "series": {key: series.__dict__ for key, series in self._series.items()},
        }
        _atomic_json(self.metrics_path, payload)

    def _persist_traces(self) -> None:
        payload = {
            "schema_version": 1,
            "updated_at": _iso_now(),
            "traces": [record.__dict__ for record in self._traces],
        }
        _atomic_json(self.traces_path, payload)


def _subsystem(event_type: str) -> str:
    prefix = event_type.split("_", 1)[0]
    return {
        "chat": "generation",
        "debug": "generation",
        "state": "snapshot",
        "goal": "goals",
        "decision": "decisions",
        "adapter": "adapter",
        "sleep": "sleep",
        "memory": "memory",
        "motivation": "goals",
        "autonomy": "runtime",
    }.get(prefix, "runtime")


def _safe_label_value(value: str) -> str:
    text = str(value)
    if _SAFE_VALUE.fullmatch(text) is None:
        raise ValueError("metric label is not a bounded safe categorical value")
    return text


def _categorical_label(metric: str, label: str, value: str) -> str:
    text = str(value)
    allowed = _LABEL_VALUES.get((metric, label))
    if allowed is not None:
        return text if text in allowed else "other"
    if label == "provider":
        return text if text in {"dummy", "transformers"} else "other"
    if label == "subsystem":
        return text if text in {
            "runtime", "generation", "snapshot", "goals", "decisions",
            "adapter", "sleep", "memory", "training",
        } else "other"
    if label == "event_type":
        return _safe_label_value(text) if text in _KNOWN_EVENT_TYPES else "other"
    return _safe_label_value(text)


_KNOWN_EVENT_TYPES = {
    "chat", "debug_chat", "sleep", "memory_read", "memory_update",
    "adapter_read", "adapter_update", "state_snapshot", "state_export",
    "state_restore", "state_reset", "context_update", "emotion_tick",
    "value_read", "value_update", "goal_read", "goal_update",
    "goal_reevaluate", "decision_read", "decision_update", "decision_generate",
    "self_model_read", "self_model_update", "experience_read",
    "experience_update", "belief_read", "belief_update", "motivation_read",
    "motivation_update", "motivation_reevaluate", "training_read",
    "training_update",
    "autonomy_schedule", "autonomy_wake",
}


def _safe_identifier(value: str) -> str:
    return value if re.fullmatch(r"[A-Za-z0-9._:@-]{1,128}", value) else "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _safe_optional_identifier(value: str | None) -> str | None:
    return None if value is None else _safe_identifier(value)


def _trace_id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def _span_id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _prometheus_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{key}="{value}"' for key, value in sorted(labels.items())) + "}"


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _unix_nanos(value: str) -> str:
    return str(int(datetime.fromisoformat(value).timestamp() * 1_000_000_000))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
