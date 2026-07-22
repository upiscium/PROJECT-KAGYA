"""Distributed training node observability routes."""

from fastapi import APIRouter, Depends, HTTPException, Query

from kagya.api.dependencies import (
    get_dataset_governance,
    get_sleep_coordinator,
    require_admin,
)
from kagya.training import DatasetGovernanceStore, SleepCoordinator


router = APIRouter(
    prefix="/api/training", tags=["training"], dependencies=[Depends(require_admin)]
)


@router.get("/nodes")
def training_nodes(
    coordinator: SleepCoordinator = Depends(get_sleep_coordinator),
) -> dict[str, list[dict]]:
    return {"nodes": coordinator.node_status()}


@router.get("/datasets")
def dataset_revisions(
    store: DatasetGovernanceStore = Depends(get_dataset_governance),
) -> dict[str, list[dict]]:
    return {"datasets": store.list_revisions()}


@router.get("/datasets/diff")
def dataset_revision_diff(
    from_revision: str = Query(alias="from"),
    to_revision: str = Query(alias="to"),
    store: DatasetGovernanceStore = Depends(get_dataset_governance),
) -> dict:
    try:
        return store.diff(from_revision, to_revision)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{revision}")
def dataset_revision(
    revision: str,
    store: DatasetGovernanceStore = Depends(get_dataset_governance),
) -> dict:
    try:
        dataset = store.get_revision(revision)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "manifest": {**dataset.manifest, "manifest_hash": dataset.manifest_hash},
        "records": [record.to_json() for record in dataset.records],
    }
