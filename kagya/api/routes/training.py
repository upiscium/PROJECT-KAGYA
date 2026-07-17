"""Distributed training node observability routes."""

from fastapi import APIRouter, Depends

from kagya.api.dependencies import get_sleep_coordinator, require_admin
from kagya.training import SleepCoordinator


router = APIRouter(
    prefix="/api/training", tags=["training"], dependencies=[Depends(require_admin)]
)


@router.get("/nodes")
def training_nodes(
    coordinator: SleepCoordinator = Depends(get_sleep_coordinator),
) -> dict[str, list[dict]]:
    return {"nodes": coordinator.node_status()}
