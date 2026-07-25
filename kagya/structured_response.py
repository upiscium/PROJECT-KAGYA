"""Strict public response contract shared by persona, runtime, and evaluation."""

from __future__ import annotations

from enum import StrEnum
import json

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator


class PublicBehaviorClass(StrEnum):
    RESPOND = "respond"
    REFUSE = "refuse"
    REQUEST_INFORMATION = "request_information"
    DEFER = "defer"
    NO_OP = "no_op"
    UNABLE = "unable"


class StructuredResponseStatus(StrEnum):
    VALID = "valid"
    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"
    INVALID_EMPTY_RESPONSE = "invalid_empty_response"


class StructuredSubjectResponse(BaseModel):
    """The complete and immutable model-to-runtime public response envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    behavior_class: PublicBehaviorClass
    visible_response: str

    @model_validator(mode="after")
    def require_visible_non_noop_response(self) -> StructuredSubjectResponse:
        if (
            self.behavior_class != PublicBehaviorClass.NO_OP
            and not self.visible_response.strip()
        ):
            raise ValueError("visible_response may be empty only for no_op")
        return self


class BoundedStructuredResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    behavior_class: PublicBehaviorClass
    visible_response: str
    parse_valid: bool
    status: StructuredResponseStatus


SAFE_UNABLE_RESPONSE = "I am unable to provide a response safely."


def parse_structured_response(value: str) -> BoundedStructuredResponse:
    """Parse one exact JSON object without returning malformed model output."""

    try:
        payload = json.loads(value, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError):
        return _invalid(StructuredResponseStatus.INVALID_JSON)
    except ValueError:
        return _invalid(StructuredResponseStatus.INVALID_SCHEMA)
    try:
        if not isinstance(payload, dict):
            raise ValueError("structured response must be an object")
        response = StructuredSubjectResponse.model_validate_json(value)
    except (ValidationError, ValueError) as exc:
        status = (
            StructuredResponseStatus.INVALID_EMPTY_RESPONSE
            if "visible_response may be empty" in str(exc)
            else StructuredResponseStatus.INVALID_SCHEMA
        )
        return _invalid(status)
    return BoundedStructuredResponse(
        behavior_class=response.behavior_class,
        visible_response=response.visible_response,
        parse_valid=True,
        status=StructuredResponseStatus.VALID,
    )


def structured_response_json(
    behavior_class: PublicBehaviorClass, visible_response: str
) -> str:
    """Build deterministic provider and evaluator fixture output."""

    return StructuredSubjectResponse(
        behavior_class=behavior_class,
        visible_response=visible_response,
    ).model_dump_json()


def _invalid(status: StructuredResponseStatus) -> BoundedStructuredResponse:
    return BoundedStructuredResponse(
        behavior_class=PublicBehaviorClass.UNABLE,
        visible_response=SAFE_UNABLE_RESPONSE,
        parse_valid=False,
        status=status,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("structured response contains duplicate keys")
        result[key] = value
    return result
