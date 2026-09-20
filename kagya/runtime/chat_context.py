"""Request-scoped chat Context selection policy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from kagya.identifiers import validate_identifier
from kagya.runtime.context import (
    ContextConflict,
    ContextFrame,
    ContextNotFound,
    ContextRegistry,
    ContextStateInvalid,
    ContextStatus,
    ContextType,
)


DEFAULT_CHAT_CONTEXT_ID = "conversation.default"
CHAT_SOURCE_CHANNEL = "chat"
_CHAT_SESSION_DOMAIN = b"PROJECT-KAGYA:R09:CHAT-SESSION:V1\0"


@dataclass(frozen=True, slots=True)
class ChatContextSelectors:
    """Validated, ephemeral Context selectors admitted with one chat request."""

    context_id: str | None = None
    client_session_id: str | None = None
    interlocutor_key: str | None = None

    def __post_init__(self) -> None:
        for value in (
            self.context_id,
            self.client_session_id,
            self.interlocutor_key,
        ):
            if value is not None:
                try:
                    validate_identifier(value)
                except Exception:
                    raise ContextStateInvalid("invalid chat selector") from None


def resolve_chat_context(
    registry: ContextRegistry,
    selectors: ChatContextSelectors | None = None,
) -> ContextFrame:
    """Resolve, validate, and select the one durable Context for a chat turn."""

    if not isinstance(registry, ContextRegistry):
        raise TypeError("registry must be ContextRegistry")
    selected = selectors or ChatContextSelectors()

    if selected.context_id is not None:
        frame = registry.get(selected.context_id)
        _require_active(frame)
        if (
            selected.client_session_id is not None
            and frame.source_session_id != selected.client_session_id
        ):
            raise ContextConflict("context session conflicts")
        frame = _bind_interlocutor(registry, frame, selected.interlocutor_key)
        registry.set_current(frame.context_id)
        return frame

    if selected.client_session_id is not None:
        return _resolve_session_context(registry, selected)

    return _resolve_default_context(registry, selected.interlocutor_key)


def _resolve_default_context(
    registry: ContextRegistry, interlocutor_key: str | None
) -> ContextFrame:
    try:
        frame = registry.get(DEFAULT_CHAT_CONTEXT_ID)
    except ContextNotFound:
        frame = registry.create(
            DEFAULT_CHAT_CONTEXT_ID,
            ContextType.CONVERSATION,
            CHAT_SOURCE_CHANNEL,
            source_session_id=None,
            participant_refs=(interlocutor_key,) if interlocutor_key else (),
        )
    else:
        _require_default_identity(frame)
        _require_active(frame)
        frame = _bind_interlocutor(registry, frame, interlocutor_key)
    registry.set_current(frame.context_id)
    return frame


def _resolve_session_context(
    registry: ContextRegistry, selectors: ChatContextSelectors
) -> ContextFrame:
    assert selectors.client_session_id is not None
    matches = registry.find_by_source_session(
        CHAT_SOURCE_CHANNEL, selectors.client_session_id
    )
    if len(matches) > 1:
        raise ContextConflict("client session maps to multiple contexts")
    if matches:
        frame = matches[0]
        _require_session_identity(frame, selectors.client_session_id)
        _require_active(frame)
        frame = _bind_interlocutor(registry, frame, selectors.interlocutor_key)
        registry.set_current(frame.context_id)
        return frame

    context_id = _session_context_id(selectors.client_session_id)
    try:
        frame = registry.get(context_id)
    except ContextNotFound:
        frame = registry.create(
            context_id,
            ContextType.CONVERSATION,
            CHAT_SOURCE_CHANNEL,
            source_session_id=selectors.client_session_id,
            participant_refs=(selectors.interlocutor_key,)
            if selectors.interlocutor_key
            else (),
        )
    else:
        _require_session_identity(frame, selectors.client_session_id)
        _require_active(frame)
        frame = _bind_interlocutor(registry, frame, selectors.interlocutor_key)
    registry.set_current(frame.context_id)
    return frame


def _session_context_id(client_session_id: str) -> str:
    digest = hashlib.sha256(
        _CHAT_SESSION_DOMAIN + client_session_id.encode("ascii")
    ).hexdigest()
    return f"conversation.session.{digest}"


def _require_active(frame: ContextFrame) -> None:
    if frame.status is not ContextStatus.ACTIVE:
        raise ContextStateInvalid("context must be active")


def _require_default_identity(frame: ContextFrame) -> None:
    if (
        frame.context_type is not ContextType.CONVERSATION
        or frame.source_channel != CHAT_SOURCE_CHANNEL
        or frame.source_session_id is not None
    ):
        raise ContextConflict("default context identity conflicts")


def _require_session_identity(frame: ContextFrame, session_id: str) -> None:
    if (
        frame.context_type is not ContextType.CONVERSATION
        or frame.source_channel != CHAT_SOURCE_CHANNEL
        or frame.source_session_id != session_id
    ):
        raise ContextConflict("session context identity conflicts")


def _bind_interlocutor(
    registry: ContextRegistry,
    frame: ContextFrame,
    interlocutor_key: str | None,
) -> ContextFrame:
    if interlocutor_key is None or interlocutor_key in frame.participant_refs:
        return frame
    if frame.participant_refs:
        raise ContextConflict("interlocutor conflicts")
    return registry.add_participant_ref(frame.context_id, interlocutor_key)
