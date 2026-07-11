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
    EvaluationResultDetail,
    EvaluationResultListResponse,
    EvaluationResultSummary,
)
from kagya.config import Settings


router = APIRouter(prefix="/api/evaluations", tags=["evaluations"], dependencies=[Depends(require_admin)])


@router.get("", response_model=EvaluationResultListResponse)
def list_evaluation_results(settings: Settings = Depends(get_api_settings)) -> EvaluationResultListResponse:
    result_dir = settings.adapter_registry.eval_result_dir
    if not result_dir.exists():
        return EvaluationResultListResponse(results=[])

    results = [_summary_from_path(path) for path in sorted(result_dir.glob("*.json")) if path.is_file()]
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
    return EvaluationResultDetail(filename=filename, payload=redact_private_fields(_read_json_object(result_path)))


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


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as result_file:
        data = json.load(result_file)
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail=f"Evaluation result is not a JSON object: {path.name}")
    return data


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
