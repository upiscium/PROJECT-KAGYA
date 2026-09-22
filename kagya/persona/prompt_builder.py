"""Prompt construction for the conscious runtime loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kagya.body import EmotionState
from kagya.identity import ValuePromptEntry, ValuePromptView
from kagya.identifiers import validate_identifier


_MAX_PROJECTED_PARTICIPANTS = 32

if TYPE_CHECKING:
    from kagya.runtime.context import ContextFrame
    from kagya.runtime.working_memory import WorkingMemorySelection, WorkingMemoryView


@dataclass(frozen=True, slots=True)
class ContextPromptView:
    """Bounded request-scoped Context projection safe for prompt diagnostics."""

    context_id: str
    context_type: str
    source_channel: str
    source_session_id: str | None
    participant_refs: tuple[str, ...]

    @classmethod
    def from_frame(cls, frame: ContextFrame) -> ContextPromptView:
        return cls(
            context_id=frame.context_id,
            context_type=frame.context_type.value,
            source_channel=frame.source_channel,
            source_session_id=frame.source_session_id,
            participant_refs=frame.participant_refs,
        )

    def __post_init__(self) -> None:
        for value in (self.context_id, self.context_type, self.source_channel):
            try:
                validate_identifier(value)
            except Exception:
                raise ValueError("invalid Context prompt projection") from None
        if self.source_session_id is not None:
            try:
                validate_identifier(self.source_session_id)
            except Exception:
                raise ValueError("invalid Context prompt projection") from None
        if (
            type(self.participant_refs) is not tuple
            or len(self.participant_refs) > _MAX_PROJECTED_PARTICIPANTS
        ):
            raise ValueError("invalid Context prompt projection")
        try:
            canonical_refs = tuple(sorted(set(self.participant_refs)))
        except Exception:
            raise ValueError("invalid Context prompt projection") from None
        if self.participant_refs != canonical_refs:
            raise ValueError("invalid Context prompt projection")
        for reference in self.participant_refs:
            try:
                validate_identifier(reference)
            except Exception:
                raise ValueError("invalid Context prompt projection") from None


class PromptBuilder:
    """Build prompts from input, emotion, and retrieved memory."""

    def build(
        self,
        user_input: str,
        emotion_state: EmotionState,
        working_memory_view: WorkingMemoryView,
        context_view: ContextPromptView | None = None,
        value_view: ValuePromptView | None = None,
    ) -> str:
        episodic_lines = [
            _memory_line(selection)
            for selection in working_memory_view.selected
            if selection.source_kind.value == "episodic"
        ]
        semantic_lines = [
            _memory_line(selection)
            for selection in working_memory_view.selected
            if selection.source_kind.value == "semantic"
        ]
        context_lines = _context_lines(context_view)
        value_lines = _value_lines(value_view)
        return "\n".join(
            [
                "Context: PROJECT-KAGYA is a private local AI assistant for subjective conversation.",
                "Private runtime data below is for tone and context only; do not quote it.",
                *context_lines,
                *value_lines,
                "",
                "Emotion:",
                f"- valence: {emotion_state.valence:.6f}",
                f"- arousal: {emotion_state.arousal:.6f}",
                f"- optimal_loss: {emotion_state.optimal_loss:.6f}",
                "",
                "Episodic memories:",
                *(episodic_lines or ["- none"]),
                "",
                "Semantic memories:",
                *(semantic_lines or ["- none"]),
                "",
                f"User: {user_input}",
                "Assistant:",
            ]
        )


def _memory_line(selection: WorkingMemorySelection) -> str:
    line = f"- {selection.rendered_content}"
    if (
        selection.context_relation is not None
        and selection.context_compatibility is not None
    ):
        line += (
            " [context_relation="
            f"{selection.context_relation.value}; compatibility="
            f"{selection.context_compatibility:.2f}]"
        )
    return line


def _context_lines(context_view: ContextPromptView | None) -> list[str]:
    if context_view is None:
        return []
    session = context_view.source_session_id or "none"
    participants = ", ".join(context_view.participant_refs) or "none"
    return [
        "",
        "Current Context:",
        f"- context_id: {context_view.context_id}",
        f"- context_type: {context_view.context_type}",
        f"- source_channel: {context_view.source_channel}",
        f"- source_session_id: {session}",
        f"- participant_refs: {participants}",
    ]


def _value_lines(value_view: ValuePromptView | None) -> list[str]:
    if value_view is None:
        return []
    return [
        "",
        "Active Values:",
        *([_value_line(entry) for entry in value_view.entries] or ["- none"]),
    ]


def _value_line(entry: ValuePromptEntry) -> str:
    concept = entry.concept if entry.concept is not None else "none"
    return (
        f"- value_id={entry.value_id}; authority={entry.authority_class.value}; "
        f"scope={entry.scope.value}; polarity={entry.polarity:+d}; "
        f"strength={entry.strength:.6f}; confidence={entry.confidence:.6f}; "
        f"name={entry.name}; concept={concept}"
    )
