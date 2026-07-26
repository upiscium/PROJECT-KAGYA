"""Route helper for optional structured pressure observations."""

from typing import Any, TypeVar

from kagya.identity import SocialPressureMetadata


T = TypeVar("T")


def retain_pressure_observation(
    main_loop: Any,
    result: T,
    metadata: SocialPressureMetadata | None,
    *,
    context_id: str | None = None,
) -> T:
    if metadata is not None:
        main_loop.record_social_pressure(metadata, context_id=context_id)
    return result
