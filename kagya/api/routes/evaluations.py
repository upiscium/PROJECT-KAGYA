"""Adapter evaluation result routes."""

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from kagya.api.dependencies import get_api_settings, require_admin
from kagya.api.redaction import redact_private_fields
from kagya.api.schemas.evaluation import (
    AdapterEvaluationHistoryResponse,
    BehavioralEvaluationDetail,
    BehavioralEvaluationHistoryResponse,
    BehavioralEvaluationSummary,
    BehavioralFailureArtifact,
    EvaluationResultDetail,
    EvaluationResultListResponse,
    EvaluationResultSummary,
)
from kagya.config import Settings


router = APIRouter(
    prefix="/api/evaluations",
    tags=["evaluations"],
    dependencies=[Depends(require_admin)],
)


@router.get("/behavioral", response_model=BehavioralEvaluationHistoryResponse)
def list_behavioral_evaluations(
    settings: Settings = Depends(get_api_settings),
) -> BehavioralEvaluationHistoryResponse:
    result_dir = settings.adapter_registry.eval_result_dir / "behavioral"
    if not result_dir.exists():
        return BehavioralEvaluationHistoryResponse(results=[])
    results = [
        _behavioral_summary(path)
        for path in result_dir.glob("*.json")
        if path.is_file()
    ]
    return BehavioralEvaluationHistoryResponse(
        results=sorted(results, key=lambda item: item.created_at, reverse=True)
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
