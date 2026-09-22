"""Administrative Value governance and bounded read routes."""

from fastapi import APIRouter, Depends, HTTPException

from kagya.api.dependencies import (
    get_agent_runtime,
    get_api_settings,
    get_main_loop,
    require_admin,
)
from kagya.api.runtime_execution import execute
from kagya.api.schemas.value import (
    ValueEmptyRequest,
    ValueListResponse,
    ValueMutationResponse,
    ValueOriginReviewRequest,
    ValueResponse,
    ValueRevisionListResponse,
    ValueRevisionResponse,
    ValueRollbackRequest,
    ValueSeedListResponse,
    ValueSeedResponse,
)
from kagya.config import Settings
from kagya.identity import (
    ValueMutationResult,
    ValueNotFound,
    ValueOriginReviewDecision,
    ValueRevisionRecord,
    ValueState,
    ValueSystem,
    recompute_seed_contract_digest,
)
from kagya.identifiers import validate_identifier
from kagya.runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    KagyaMainLoop,
)


router = APIRouter(
    prefix="/api/values", tags=["values"], dependencies=[Depends(require_admin)]
)


def _value_id(value: str) -> str:
    try:
        return validate_identifier(value)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=[{"type": "value_error", "loc": ["path"], "msg": "Invalid request"}],
        ) from None


def _value(system_value: ValueState) -> ValueResponse:
    return ValueResponse(
        value_id=system_value.value_id,
        revision=system_value.revision,
        name=system_value.name,
        concept=system_value.concept,
        scope=system_value.scope.value,
        context_ids=system_value.context_ids,
        polarity=system_value.polarity,
        strength=system_value.strength,
        confidence=system_value.confidence,
        stability=system_value.stability,
        protectedness=system_value.protectedness,
        negotiability=system_value.negotiability,
        allowed_update_rate=system_value.allowed_update_rate,
        frozen=system_value.frozen,
        active=system_value.is_active(),
        admission=system_value.origin.admission.value,
        origin_actor=system_value.origin.actor.value,
        origin_input_kind=system_value.origin.input_kind.value,
        origin_id=system_value.origin.origin_id,
        seed_contract_digest=system_value.seed_contract_digest,
    )


def _mutation(result: ValueMutationResult) -> ValueMutationResponse:
    return ValueMutationResponse(
        value_id=result.value_id,
        revision=result.value.revision,
        status=result.status.value,
        frozen=result.value.frozen,
        admission=result.value.origin.admission.value,
    )


def _revision(record: ValueRevisionRecord) -> ValueRevisionResponse:
    if not isinstance(record, ValueRevisionRecord):
        raise RuntimeError("Value revision authority is malformed")
    return ValueRevisionResponse(
        value_id=record.value_id,
        from_revision=record.from_revision,
        to_revision=record.to_revision,
        before_digest=record.before_digest,
        after_digest=record.after_digest,
        operation=record.operation.value,
        origin_id=record.origin_id,
        event_id=record.event_id,
        event_sequence=record.event_sequence,
        recorded_at=record.recorded_at,
        target_revision=record.target_revision,
        record_digest=record.record_digest,
    )


def _system(main_loop: KagyaMainLoop) -> ValueSystem:
    return main_loop.value_system


@router.get("/config-seeds", response_model=ValueSeedListResponse)
def list_config_seeds(
    settings: Settings = Depends(get_api_settings),
    main_loop: KagyaMainLoop = Depends(get_main_loop),
) -> ValueSeedListResponse:
    values = _system(main_loop).value_map
    seeds = tuple(
        ValueSeedResponse(
            value_id=seed.value_id,
            name=seed.name,
            seed_contract_digest=recompute_seed_contract_digest(seed.to_declaration()),
            adopted=(
                seed.value_id in values
                and values[seed.value_id].seed_contract_digest
                == recompute_seed_contract_digest(seed.to_declaration())
            ),
        )
        for seed in sorted(settings.values.seeds, key=lambda item: item.value_id)
    )
    return ValueSeedListResponse(seeds=seeds)


@router.get("", response_model=ValueListResponse)
def list_values(
    main_loop: KagyaMainLoop = Depends(get_main_loop),
) -> ValueListResponse:
    return ValueListResponse(values=tuple(_value(value) for value in _system(main_loop).values))


@router.get("/{value_id}", response_model=ValueResponse)
def get_value(
    value_id: str,
    main_loop: KagyaMainLoop = Depends(get_main_loop),
) -> ValueResponse:
    checked = _value_id(value_id)
    try:
        return _value(_system(main_loop).get(checked))
    except ValueNotFound as exc:
        raise HTTPException(status_code=404, detail="Value not found") from exc


@router.get("/{value_id}/revisions", response_model=ValueRevisionListResponse)
def list_revisions(
    value_id: str,
    main_loop: KagyaMainLoop = Depends(get_main_loop),
) -> ValueRevisionListResponse:
    checked = _value_id(value_id)
    try:
        history = _system(main_loop).history(checked)
    except ValueNotFound as exc:
        raise HTTPException(status_code=404, detail="Value not found") from exc
    return ValueRevisionListResponse(
        revisions=tuple(_revision(record) for record in history.records)
    )


@router.post("/{value_id}/freeze", response_model=ValueMutationResponse)
def freeze_value(
    value_id: str,
    _request: ValueEmptyRequest | None = None,
    main_loop: KagyaMainLoop = Depends(get_main_loop),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ValueMutationResponse:
    checked = _value_id(value_id)
    result = execute(
        runtime,
        AgentEventType.VALUE_GOVERNANCE,
        AgentEventSource.API_VALUES_FREEZE,
        lambda: main_loop.freeze_value(runtime, checked),
    )
    return _mutation(result)


@router.post("/{value_id}/unfreeze", response_model=ValueMutationResponse)
def unfreeze_value(
    value_id: str,
    _request: ValueEmptyRequest | None = None,
    main_loop: KagyaMainLoop = Depends(get_main_loop),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ValueMutationResponse:
    checked = _value_id(value_id)
    result = execute(
        runtime,
        AgentEventType.VALUE_GOVERNANCE,
        AgentEventSource.API_VALUES_UNFREEZE,
        lambda: main_loop.unfreeze_value(runtime, checked),
    )
    return _mutation(result)


@router.post("/{value_id}/rollback", response_model=ValueMutationResponse)
def rollback_value(
    value_id: str,
    request: ValueRollbackRequest,
    main_loop: KagyaMainLoop = Depends(get_main_loop),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ValueMutationResponse:
    checked = _value_id(value_id)
    result = execute(
        runtime,
        AgentEventType.VALUE_GOVERNANCE,
        AgentEventSource.API_VALUES_ROLLBACK,
        lambda: main_loop.rollback_value(runtime, checked, request.target_revision),
    )
    return _mutation(result)


@router.post("/{value_id}/origin-review", response_model=ValueMutationResponse)
def review_value_origin(
    value_id: str,
    request: ValueOriginReviewRequest,
    main_loop: KagyaMainLoop = Depends(get_main_loop),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ValueMutationResponse:
    checked = _value_id(value_id)
    decision = ValueOriginReviewDecision(request.decision)
    result = execute(
        runtime,
        AgentEventType.VALUE_GOVERNANCE,
        AgentEventSource.API_VALUES_ORIGIN_REVIEW,
        lambda: main_loop.review_value_origin(runtime, checked, decision),
    )
    return _mutation(result)


@router.post("/config-seeds/{value_id}/adopt", response_model=ValueMutationResponse)
def adopt_configured_seed(
    value_id: str,
    _request: ValueEmptyRequest | None = None,
    settings: Settings = Depends(get_api_settings),
    main_loop: KagyaMainLoop = Depends(get_main_loop),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> ValueMutationResponse:
    checked = _value_id(value_id)
    if not any(seed.value_id == checked for seed in settings.values.seeds):
        raise HTTPException(status_code=404, detail="Configured Value seed not found")
    result = execute(
        runtime,
        AgentEventType.VALUE_GOVERNANCE,
        AgentEventSource.API_VALUES_SEED_ADOPT,
        lambda: main_loop.adopt_configured_value_seed(runtime, checked),
    )
    return _mutation(result)
