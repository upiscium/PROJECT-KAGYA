"""Sleep cycle routes."""

from fastapi import APIRouter, Depends

from kagya.api.dependencies import get_sleep_cycle_manager, require_admin
from kagya.api.schemas.sleep import SleepRunResponse
from kagya.learning import SleepCycleManager


router = APIRouter(prefix="/api/sleep", tags=["sleep"], dependencies=[Depends(require_admin)])


@router.post("/run", response_model=SleepRunResponse)
def run_sleep(manager: SleepCycleManager = Depends(get_sleep_cycle_manager)) -> SleepRunResponse:
    result = manager.run()
    return SleepRunResponse(
        selected_episode_ids=result.selected_episode_ids,
        semantic_memory_ids=result.semantic_memory_ids,
        dream_dataset_path=result.dream_dataset_path,
        adapter_id=None if result.adapter_entry is None else result.adapter_entry.adapter_id,
        adapter_status=None if result.adapter_entry is None else result.adapter_entry.status.value,
        dry_run=None if result.training_result is None else result.training_result.dry_run,
    )
