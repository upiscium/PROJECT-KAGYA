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
    BehavioralArtifactBusyError,
    BehavioralArtifactStore,
    AdapterRegistry,
    run_deterministic_subject_evaluation,
    scenario_fixture_hash,
    subject_completion_scenarios,
    BehavioralRuntimeKind,
    deterministic_runtime_scenarios,
    run_deterministic_runtime_evaluation,
    run_real_model_runtime_evaluation,
)
from kagya.learning.behavioral_evaluation import PairedBehavioralEvaluationResult
from kagya.learning.runtime_behavioral_runner import _manifest
from kagya.runtime import AgentEventType, AgentRuntime


router = APIRouter(
    prefix="/api/evaluations",
    tags=["evaluations"],
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
                    evaluation_state=record.state.value,
                    failure_code=record.failure_code,
                    artifact_integrity=record.status.value,
                )
            )
            continue
        try:
            summary = _behavioral_summary(path)
            entry = registry.lookup(summary.candidate_id)
            eligibility = (
                "not_applicable"
                if entry is None
                else registry.activation_eligibility(entry.adapter_id).reason.value
            )
            results.append(
                summary.model_copy(
                    update={
                        "artifact_status": record.status.value,
                        "activation_eligibility": eligibility,
                        "evaluation_state": record.state.value,
                        "failure_code": record.failure_code,
                        "artifact_integrity": record.status.value,
                    }
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
                    evaluation_state=record.state.value,
                    failure_code=record.failure_code,
                    artifact_integrity="corrupt",
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
    registry: AdapterRegistry = Depends(get_adapter_registry),
) -> BehavioralRerunResponse:
    """Re-execute an immutable built-in deterministic fixture revision."""

    source = _read_json_object(_behavioral_result_path(settings, evaluation_id))
    runtime_kind = source.get("runtime_kind")
    if runtime_kind in {
        BehavioralRuntimeKind.DETERMINISTIC_RUNTIME.value,
        BehavioralRuntimeKind.REAL_MODEL_RUNTIME.value,
    }:
        return _rerun_runtime_evaluation(source, request, settings, registry)
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
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return BehavioralRerunResponse(
        source_evaluation_id=evaluation_id,
        evaluation_id=result.evaluation_id,
        fixture_hashes_match=result.fixture_hashes == source_hashes,
        activation_gate_passed=result.activation_gate_passed,
    )


def _rerun_runtime_evaluation(
    source: dict[str, Any],
    request: BehavioralRerunRequest,
    settings: Settings,
    registry: AdapterRegistry,
) -> BehavioralRerunResponse:
    try:
        original = PairedBehavioralEvaluationResult.model_validate(source)
    except ValueError as exc:
        raise HTTPException(
            status_code=409, detail="Runtime manifest is missing or invalid"
        ) from exc
    manifest = original.manifest
    assert manifest is not None
    entry = registry.lookup(manifest.candidate_adapter_id)
    if entry is None or entry.adapter_hash is None or entry.base_model_revision is None:
        raise HTTPException(
            status_code=409, detail="Candidate artifact provenance is unavailable"
        )
    runtime_kind = original.runtime_kind
    scenarios = deterministic_runtime_scenarios(
        subject_revision=manifest.subject_revision,
        runtime_kind=runtime_kind,
    )
    fixture_hashes = {
        item.scenario_id: scenario_fixture_hash(item) for item in scenarios
    }
    evaluator_source = (
        Path(__file__).resolve().parents[2]
        / "learning"
        / "real_model_runtime_behavioral.py"
        if runtime_kind == BehavioralRuntimeKind.REAL_MODEL_RUNTIME
        else None
    )
    current = None
    if runtime_kind != BehavioralRuntimeKind.REAL_MODEL_RUNTIME:
        try:
            current = _manifest(
                settings,
                candidate_id=entry.adapter_id,
                candidate_adapter_path=Path(entry.path),
                candidate_adapter_hash=entry.adapter_hash,
                base_model_revision=entry.base_model_revision,
                subject_revision=manifest.subject_revision,
                fixture_hashes=fixture_hashes,
                evaluator_source=evaluator_source,
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail="Candidate artifact is missing or differs: candidate_adapter_path_hash",
            ) from exc
    immutable_fields = (
        "source_commit_sha",
        "source_revision_status",
        "source_tree_hash",
        "build_id",
        "subject_revision",
        "runtime_schema_version",
        "evaluator_schema_version",
        "fixture_revision",
        "fixture_set_hash",
        "config_hash",
        "base_model_id",
        "base_model_revision",
        "base_model_revision_requested",
        "base_model_revision_resolved",
        "processor_revision_requested",
        "processor_revision_resolved",
        "base_model_artifact_hash",
        "model_artifact_manifest_hash",
        "model_artifact_manifest",
        "candidate_adapter_id",
        "candidate_adapter_hash",
        "candidate_adapter_path_hash",
        "adapter_artifact_manifest_hash",
        "adapter_artifact_manifest",
        "tool_registry_hash",
        "policy_revision",
        "state_schema_version",
        "evaluator_implementation_hash",
    )
    runner = (
        run_real_model_runtime_evaluation
        if runtime_kind == BehavioralRuntimeKind.REAL_MODEL_RUNTIME
        else run_deterministic_runtime_evaluation
    )
    store = BehavioralArtifactStore(settings.adapter_registry.eval_result_dir)
    reserved = False
    try:
        with store.adapter_lock(entry.adapter_id, blocking=False):
            store.begin(request.rerun_id, adapter_key=entry.adapter_id)
            reserved = True
            store.mark_running(request.rerun_id)
            result, status = runner(
                settings,
                request.rerun_id,
                baseline_id=original.baseline.subject_id,
                candidate_id=entry.adapter_id,
                candidate_adapter_path=Path(entry.path),
                candidate_adapter_hash=entry.adapter_hash,
                base_model_revision=entry.base_model_revision,
                subject_revision=manifest.subject_revision,
            )
            if status != "prepared" or result.manifest is None:
                raise ValueError("Rerun artifact was not prepared")
            if runtime_kind == BehavioralRuntimeKind.REAL_MODEL_RUNTIME:
                current = result.manifest
            assert current is not None
            mismatches = [
                name
                for name in immutable_fields
                if getattr(current, name) != getattr(manifest, name)
            ]
            if original.fixture_hashes != fixture_hashes:
                mismatches.append("fixture_hashes")
            if mismatches:
                raise ValueError(
                    f"Immutable rerun manifest differs: {', '.join(sorted(set(mismatches)))}; use a new evaluation ID"
                )
            store.finalize(request.rerun_id)
    except BehavioralArtifactBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        if reserved:
            store.fail(request.rerun_id, "evaluation_failed")
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return BehavioralRerunResponse(
        source_evaluation_id=original.evaluation_id,
        evaluation_id=result.evaluation_id,
        fixture_hashes_match=result.fixture_hashes == original.fixture_hashes,
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
    manifest = payload.get("manifest")
    manifest = manifest if isinstance(manifest, dict) else {}
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
        runtime_kind=str(payload.get("runtime_kind", "synthetic_evaluator_contract")),
        source_commit_sha=_optional_str(manifest.get("source_commit_sha")),
        adapter_hash=_optional_str(manifest.get("candidate_adapter_hash")),
        base_model_revision=_optional_str(manifest.get("base_model_revision")),
        fixture_set_hash=_optional_str(manifest.get("fixture_set_hash")),
        deterministic_runtime_gate_passed=bool(
            payload.get("deterministic_runtime_gate_passed", False)
        ),
        real_model_runtime_gate_passed=bool(
            payload.get("real_model_runtime_gate_passed", False)
        ),
        source_integrity=str(manifest.get("source_revision_status", "unknown")),
        model_integrity=(
            "verified"
            if manifest.get("model_artifact_manifest_hash")
            and manifest.get("base_model_revision_resolved")
            else "unknown"
        ),
        artifact_integrity="valid",
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
