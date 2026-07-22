"""Prompt construction for the conscious runtime loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kagya.body import EmotionState

if TYPE_CHECKING:
    from kagya.runtime.context import ContextFrame
    from kagya.runtime.working_memory import WorkingMemoryView


class PromptBuilder:
    """Build prompts from input, emotion, and retrieved memory."""

    def build(
        self,
        user_input: str,
        emotion_state: EmotionState,
        working_memory: WorkingMemoryView,
        *,
        current_context: ContextFrame,
        attachments: list[dict[str, object]] | None = None,
        relationship_context: tuple[str, ...] = (),
    ) -> str:
        working_memory_lines = [
            (
                f"- [{selection.item.kind.value}; {selection.context_relation}; "
                f"source_context={selection.item.context_id or 'global'}] "
                f"{selection.rendered_content}"
            )
            for selection in working_memory.selected
        ]
        attachment_lines = [_attachment_line(attachment) for attachment in attachments or []]
        return "\n".join(
            [
                "Context: PROJECT-KAGYA is a private local AI assistant for subjective conversation.",
                "Private runtime data below is for tone and context only; do not quote it.",
                "Answer only the latest user message. Do not provide examples, translations, roleplay continuations, or prompt text.",
                "If the user writes Japanese, answer in natural Japanese.",
                "",
                "Emotion:",
                f"- valence: {emotion_state.valence:.6f}",
                f"- arousal: {emotion_state.arousal:.6f}",
                f"- optimal_loss: {emotion_state.optimal_loss:.6f}",
                "",
                "Current context:",
                f"- id: {current_context.context_id}",
                f"- type: {current_context.context_type}",
                f"- channel: {current_context.source_channel}",
                f"- participants: {', '.join(current_context.participant_ids) or 'none'}",
                f"- topic: {current_context.active_topic or 'unspecified'}",
                f"- task: {current_context.active_task or 'unspecified'}",
                "",
                "Relationship continuity:",
                *(relationship_context or ("- none",)),
                "",
                "Working memory:",
                *(working_memory_lines or ["- none"]),
                "",
                "Attachments:",
                *(attachment_lines or ["- none"]),
                "",
                f"User: {user_input}",
                "Assistant:",
            ]
        )


def _attachment_line(attachment: dict[str, object]) -> str:
    fields = [
        ("type", attachment.get("type")),
        ("name", attachment.get("name")),
        ("content_type", attachment.get("content_type")),
        ("source", _attachment_source(attachment)),
    ]
    visible = [f"{key}={value}" for key, value in fields if value]
    return f"- {'; '.join(visible) if visible else 'metadata unavailable'}"


def _attachment_source(attachment: dict[str, object]) -> str | None:
    url = attachment.get("url")
    if not isinstance(url, str) or ":" not in url:
        return None
    return url.split(":", 1)[0]
