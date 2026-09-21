"""Bounded administrative Value API schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class ValueEmptyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ValueRollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    target_revision: StrictInt = Field(ge=0)


class ValueOriginReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    decision: Literal["accept_provenance", "reject"]


class ValueResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value_id: str
    revision: int
    name: str
    concept: str | None
    scope: str
    context_ids: tuple[str, ...]
    polarity: int
    strength: float
    confidence: float
    stability: float
    protectedness: float
    negotiability: float
    allowed_update_rate: float
    frozen: bool
    active: bool
    admission: str
    origin_actor: str
    origin_input_kind: str
    origin_id: str
    seed_contract_digest: str | None


class ValueListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: tuple[ValueResponse, ...]


class ValueRevisionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value_id: str
    from_revision: int
    to_revision: int
    before_digest: str
    after_digest: str
    operation: str
    origin_id: str
    event_id: str
    event_sequence: int
    recorded_at: datetime
    target_revision: int | None
    record_digest: str


class ValueRevisionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revisions: tuple[ValueRevisionResponse, ...]


class ValueSeedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value_id: str
    name: str
    seed_contract_digest: str
    adopted: bool


class ValueSeedListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seeds: tuple[ValueSeedResponse, ...]


class ValueMutationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value_id: str
    revision: int
    status: str
    frozen: bool
    admission: str
