"""Sleep cycle routes."""

from fastapi import APIRouter, Depends

from kagya.api.dependencies import (
    get_agent_runtime,
    get_sleep_cycle_manager,
    require_admin,
)
from kagya.api.runtime_execution import execute
from kagya.api.schemas.sleep import SleepRunResponse
from kagya.learning import SleepCycleManager
from kagya.runtime import AgentEventSource, AgentEventType, AgentRuntime


router = APIRouter(
    prefix="/api/sleep", tags=["sleep"], dependencies=[Depends(require_admin)]
)


@router.post("/run", response_model=SleepRunResponse)
def run_sleep(
    manager: SleepCycleManager = Depends(get_sleep_cycle_manager),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> SleepRunResponse:
    result = execute(
        runtime, AgentEventType.SLEEP, AgentEventSource.API_SLEEP_RUN, manager.run
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
