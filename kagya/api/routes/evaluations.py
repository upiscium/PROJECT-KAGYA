"""Adapter evaluation result routes."""

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from kagya.api.dependencies import (
    execute_agent_event,
    get_adapter_registry,
    get_agent_runtime,
    get_api_settings,
    require_admin,
)
from kagya.api.redaction import redact_private_fields
from kagya.api.schemas.evaluation import (
    AdapterEvaluationHistoryResponse,
    BehavioralEvaluationDetail,
    BehavioralEvaluationHistoryResponse,
    BehavioralEvaluationSummary,
    BehavioralFailureArtifact,
    BehavioralRerunRequest,
    BehavioralRerunResponse,
    BehavioralArtifactReconciliationResponse,
    EvaluationResultDetail,
    EvaluationResultListResponse,
    EvaluationResultSummary,
)
from kagya.config import Settings
from kagya.learning import (
    BehavioralArtifactStore,
    AdapterRegistry,
    run_deterministic_subject_evaluation,
    scenario_fixture_hash,
    subject_completion_scenarios,
)
from kagya.runtime import AgentEventType, AgentRuntime


router = APIRouter(
    prefix="/api/evaluations",
    tags=["evaluations"],
    dependencies=[Depends(require_admin)],
)


@router.get("/behavioral", response_model=BehavioralEvaluationHistoryResponse)
def list_behavioral_evaluations(
    settings: Settings = Depends(get_api_settings),
    registry: AdapterRegistry = Depends(get_adapter_registry),
) -> BehavioralEvaluationHistoryResponse:
    if not isinstance(registry, AdapterRegistry):
        registry = AdapterRegistry(settings)
    store = BehavioralArtifactStore(settings.adapter_registry.eval_result_dir)
    records = store.reconcile(registry)
    results = []
    for record in records:
        path = settings.adapter_registry.eval_result_dir / record.relative_path
        if record.status.value not in {"valid", "orphan_result"}:
            results.append(
                BehavioralEvaluationSummary(
                    evaluation_id=record.evaluation_id,
                    artifact_status=record.status.value,
                    quarantine_error="Artifact quarantined by integrity reconciliation",
                    created_at=record.updated_at.isoformat(),
                )
            )
            continue
        try:
            results.append(
                _behavioral_summary(path).model_copy(
                    update={"artifact_status": record.status.value}
                )
            )
        except (
            HTTPException,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ):
            results.append(
                BehavioralEvaluationSummary(
                    evaluation_id=record.evaluation_id,
                    artifact_status="corrupt",
                    quarantine_error="Artifact could not be safely decoded",
                    created_at=record.updated_at.isoformat(),
                )
            )
    return BehavioralEvaluationHistoryResponse(
        results=sorted(results, key=lambda item: item.created_at, reverse=True)
    )


@router.post(
    "/behavioral-reconciliation",
    response_model=BehavioralArtifactReconciliationResponse,
)
def reconcile_behavioral_artifacts(
    settings: Settings = Depends(get_api_settings),
    registry: AdapterRegistry = Depends(get_adapter_registry),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> BehavioralArtifactReconciliationResponse:
    records = execute_agent_event(
        runtime,
        AgentEventType.BEHAVIORAL_EVALUATE,
        source="api.evaluations.behavioral_reconciliation",
        handler=lambda: BehavioralArtifactStore(
            settings.adapter_registry.eval_result_dir
        ).reconcile(registry, quarantine_invalid=True),
    ).value
    return BehavioralArtifactReconciliationResponse(
        artifacts=[item.model_dump(mode="json") for item in records]
    )


@router.get("/behavioral/{evaluation_id}", response_model=BehavioralEvaluationDetail)
def get_behavioral_evaluation(
    evaluation_id: str,
    settings: Settings = Depends(get_api_settings),
) -> BehavioralEvaluationDetail:
    path = _behavioral_result_path(settings, evaluation_id)
    return BehavioralEvaluationDetail(
        evaluation_id=evaluation_id,
        payload=redact_private_fields(_read_json_object(path)),
    )


@router.get(
    "/behavioral/{evaluation_id}/failures/{scenario_id}",
    response_model=BehavioralFailureArtifact,
)
def get_behavioral_failure_artifact(
    evaluation_id: str,
    scenario_id: str,
    settings: Settings = Depends(get_api_settings),
) -> BehavioralFailureArtifact:
    if Path(scenario_id).name != scenario_id or not scenario_id.endswith(".json"):
        raise HTTPException(
            status_code=404, detail="Behavioral failure artifact not found"
        )
    _behavioral_result_path(settings, evaluation_id)
    path = (
        settings.adapter_registry.eval_result_dir
        / "behavioral"
        / "failures"
        / evaluation_id
        / scenario_id
    )
    if not path.is_file():
        raise HTTPException(
            status_code=404, detail="Behavioral failure artifact not found"
        )
    return BehavioralFailureArtifact(
        evaluation_id=evaluation_id,
        scenario_id=scenario_id.removesuffix(".json"),
        payload=redact_private_fields(_read_json_object(path)),
    )


@router.post(
    "/behavioral/{evaluation_id}/rerun", response_model=BehavioralRerunResponse
)
def rerun_behavioral_evaluation(
    evaluation_id: str,
    request: BehavioralRerunRequest,
    settings: Settings = Depends(get_api_settings),
) -> BehavioralRerunResponse:
    """Re-execute an immutable built-in deterministic fixture revision."""

    source = _read_json_object(_behavioral_result_path(settings, evaluation_id))
    reproducibility = _object(source.get("reproducibility"))
    if not reproducibility:
        raise HTTPException(status_code=409, detail="Evaluation has no rerun metadata")
    first = next(iter(reproducibility.values()))
    if not isinstance(first, dict) or first.get("runtime") != "deterministic_fixture":
        raise HTTPException(
            status_code=409, detail="Evaluation runtime is not reproducible"
        )
    subject_revision = str(first.get("subject_revision", ""))
    scenarios = subject_completion_scenarios(subject_revision=subject_revision)
    expected_hashes = {
        item.scenario_id: scenario_fixture_hash(item) for item in scenarios
    }
    source_hashes = {
        str(key): str(value)
        for key, value in _object(source.get("fixture_hashes")).items()
    }
    if source_hashes != expected_hashes:
        raise HTTPException(
            status_code=409,
            detail="Fixture revision or hashes are unavailable for rerun",
        )
    baseline = _object(source.get("baseline"))
    candidate = _object(source.get("candidate"))
    try:
        result = run_deterministic_subject_evaluation(
            settings.adapter_registry.eval_result_dir,
            request.rerun_id,
            baseline_id=str(baseline.get("subject_id", "baseline")),
            candidate_id=str(candidate.get("subject_id", "candidate")),
            subject_revision=subject_revision,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return BehavioralRerunResponse(
        source_evaluation_id=evaluation_id,
        evaluation_id=result.evaluation_id,
        fixture_hashes_match=result.fixture_hashes == source_hashes,
        activation_gate_passed=result.activation_gate_passed,
    )


@router.get("", response_model=EvaluationResultListResponse)
def list_evaluation_results(
    settings: Settings = Depends(get_api_settings),
) -> EvaluationResultListResponse:
    result_dir = settings.adapter_registry.eval_result_dir
    if not result_dir.exists():
        return EvaluationResultListResponse(results=[])

    results = [
        _summary_from_path(path)
        for path in sorted(result_dir.glob("*.json"))
        if path.is_file()
    ]
    return EvaluationResultListResponse(
        results=sorted(results, key=lambda result: result.updated_at, reverse=True)
    )


@router.get(
    "/adapters/{adapter_id}/history", response_model=AdapterEvaluationHistoryResponse
)
def get_adapter_evaluation_history(
    adapter_id: str,
    settings: Settings = Depends(get_api_settings),
) -> AdapterEvaluationHistoryResponse:
    result_dir = settings.adapter_registry.eval_result_dir
    if not result_dir.exists():
        return AdapterEvaluationHistoryResponse(adapter_id=adapter_id, results=[])
    results = [
        summary
        for summary in (_summary_from_path(path) for path in result_dir.glob("*.json"))
        if summary.adapter_id == adapter_id
    ]
    return AdapterEvaluationHistoryResponse(
        adapter_id=adapter_id,
        results=sorted(results, key=lambda result: result.updated_at, reverse=True),
    )


@router.get("/{filename}", response_model=EvaluationResultDetail)
def get_evaluation_result(
    filename: str,
    settings: Settings = Depends(get_api_settings),
) -> EvaluationResultDetail:
    if Path(filename).name != filename or not filename.endswith(".json"):
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    result_path = settings.adapter_registry.eval_result_dir / filename
    if not result_path.is_file():
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    return EvaluationResultDetail(
        filename=filename, payload=redact_private_fields(_read_json_object(result_path))
    )


def _summary_from_path(path: Path) -> EvaluationResultSummary:
    payload = _read_json_object(path)
    return EvaluationResultSummary(
        filename=path.name,
        adapter_id=str(payload.get("adapter_id", path.stem)),
        score=_optional_float(payload.get("score")),
        previous_score=_optional_float(payload.get("previous_score")),
        score_delta=_optional_float(payload.get("score_delta")),
        regression=bool(payload.get("regression", False)),
        decision=_optional_str(payload.get("decision")),
        status_before=_optional_str(payload.get("status_before")),
        status_after=_optional_str(payload.get("status_after")),
        case_count=_optional_int(payload.get("case_count")),
        updated_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
    )


def _behavioral_summary(path: Path) -> BehavioralEvaluationSummary:
    payload = _read_json_object(path)
    baseline = payload.get("baseline", {})
    candidate = payload.get("candidate", {})
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise HTTPException(
            status_code=400, detail=f"Invalid behavioral result: {path.name}"
        )
    return BehavioralEvaluationSummary(
        evaluation_id=str(payload.get("evaluation_id", path.stem)),
        baseline_id=str(baseline.get("subject_id", "")),
        candidate_id=str(candidate.get("subject_id", "")),
        baseline_score=float(baseline.get("aggregate_score", 0.0)),
        candidate_score=float(candidate.get("aggregate_score", 0.0)),
        baseline_dimensions=_dimension_scores(baseline),
        candidate_dimensions=_dimension_scores(candidate),
        dimension_deltas={
            str(key): float(value)
            for key, value in _object(payload.get("dimension_deltas")).items()
        },
        activation_gate_passed=bool(payload.get("activation_gate_passed", False)),
        regression_dimensions=[
            str(item) for item in payload.get("regression_dimensions", [])
        ],
        threshold_failure_dimensions=[
            str(item) for item in payload.get("threshold_failure_dimensions", [])
        ],
        hard_gate_failures=[
            str(item) for item in candidate.get("hard_gate_failures", [])
        ],
        tool_execution_dimensions_complete=bool(
            payload.get("tool_execution_dimensions_complete", False)
        ),
        created_at=str(
            payload.get(
                "created_at",
                datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
            )
        ),
    )


def _behavioral_result_path(settings: Settings, evaluation_id: str) -> Path:
    if (
        Path(evaluation_id).name != evaluation_id
        or not evaluation_id
        or evaluation_id.endswith(".json")
    ):
        raise HTTPException(status_code=404, detail="Behavioral evaluation not found")
    path = (
        settings.adapter_registry.eval_result_dir
        / "behavioral"
        / f"{evaluation_id}.json"
    )
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Behavioral evaluation not found")
    return path


def _dimension_scores(subject: dict[str, Any]) -> dict[str, float]:
    raw = subject.get("dimension_scores", [])
    if not isinstance(raw, list):
        return {}
    return {
        str(item["dimension"]): float(item["score"])
        for item in raw
        if isinstance(item, dict) and "dimension" in item and "score" in item
    }


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as result_file:
        data = json.load(result_file)
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=400,
            detail=f"Evaluation result is not a JSON object: {path.name}",
        )
    return data


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
