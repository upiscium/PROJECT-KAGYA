"""Sleep API schemas."""

from pydantic import BaseModel


class SleepRunResponse(BaseModel):
    selected_episode_ids: list[str]
    semantic_memory_ids: list[str]
    dream_dataset_path: str | None
    adapter_id: str | None
    adapter_status: str | None
    dry_run: bool | None
