"""Versioned public operation progress shared by chat and actions."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OperationState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class OperationErrorCode(StrEnum):
    INTERNAL_ERROR = "internal_error"
    INTERRUPTED = "interrupted"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"


class OperationCancelCode(StrEnum):
    CLIENT_REQUEST = "client_request"
    TIMEOUT = "timeout"
    SHUTDOWN = "shutdown"


class OperationStatus(BaseModel):
    """Strict public-safe operation lifecycle projection."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    operation_id: str = Field(min_length=1, max_length=128)
    event_id: str = Field(min_length=1, max_length=128)
    status: OperationState
    status_sequence: int = Field(ge=1)
    queue_position: int | None = Field(default=None, ge=1)
    submitted_at: datetime
    started_at: datetime | None = None
    finalizing_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime
    error_code: OperationErrorCode | None = None
    cancel_code: OperationCancelCode | None = None
    cancel_requested: bool = False
    result_available: bool = False

    @model_validator(mode="after")
    def validate_public_lifecycle(self) -> "OperationStatus":
        for value in (
            self.submitted_at,
            self.started_at,
            self.finalizing_at,
            self.completed_at,
            self.updated_at,
        ):
            if value is not None and value.tzinfo is None:
                raise ValueError("operation timestamps must include a timezone")
        if self.status != OperationState.QUEUED and self.queue_position is not None:
            raise ValueError("only queued operations have a queue position")
        if self.result_available != (self.status == OperationState.COMPLETED):
            raise ValueError("result availability must match completed status")
        if self.error_code is not None and self.status != OperationState.FAILED:
            raise ValueError("error code requires failed status")
        if self.cancel_code is not None and self.status != OperationState.CANCELED:
            raise ValueError("cancel code requires canceled status")
        return self


def operation_now() -> datetime:
    return datetime.now(UTC)
