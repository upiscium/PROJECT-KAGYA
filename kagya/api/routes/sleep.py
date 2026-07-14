"""Sleep cycle routes."""

from fastapi import APIRouter, Depends, Request

from kagya.api.dependencies import (
    execute_agent_event,
    get_agent_runtime,
    get_runtime_event_log,
    get_sleep_cycle_manager,
    require_admin,
)
from kagya.api.observability import RuntimeEventLog
from kagya.api.schemas.sleep import SleepRunResponse
from kagya.runtime import AgentEventType, AgentRuntime


router = APIRouter(
    prefix="/api/sleep", tags=["sleep"], dependencies=[Depends(require_admin)]
)


@router.post("/run", response_model=SleepRunResponse)
def run_sleep(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    event_log: RuntimeEventLog = Depends(get_runtime_event_log),
) -> SleepRunResponse:
    result = execute_agent_event(
        runtime,
        AgentEventType.SLEEP,
        source="api.sleep",
        handler=lambda: get_sleep_cycle_manager(request).run(),
    ).value
    event_log.record(
        category="sleep",
        event_type="run_completed",
        message="Sleep cycle completed",
        metadata={
            "selected_episode_count": len(result.selected_episode_ids),
            "semantic_memory_count": len(result.semantic_memory_ids),
            "adapter_id": None
            if result.adapter_entry is None
            else result.adapter_entry.adapter_id,
            "adapter_status": None
            if result.adapter_entry is None
            else result.adapter_entry.status.value,
            "dry_run": None
            if result.training_result is None
            else result.training_result.dry_run,
        },
    )
    return SleepRunResponse(
        selected_episode_ids=result.selected_episode_ids,
        semantic_memory_ids=result.semantic_memory_ids,
        dream_dataset_path=result.dream_dataset_path,
        adapter_id=None
        if result.adapter_entry is None
        else result.adapter_entry.adapter_id,
        adapter_status=None
        if result.adapter_entry is None
        else result.adapter_entry.status.value,
        dry_run=None
        if result.training_result is None
        else result.training_result.dry_run,
    )
