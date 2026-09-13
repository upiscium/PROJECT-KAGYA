"""Prompt construction for the conscious runtime loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kagya.body import EmotionState

if TYPE_CHECKING:
    from kagya.runtime.working_memory import WorkingMemoryView


class PromptBuilder:
    """Build prompts from input, emotion, and retrieved memory."""

    def build(
        self,
        user_input: str,
        emotion_state: EmotionState,
        working_memory_view: WorkingMemoryView,
    ) -> str:
        episodic_lines = [
            f"- {selection.rendered_content}"
            for selection in working_memory_view.selected
            if selection.source_kind.value == "episodic"
        ]
        semantic_lines = [
            f"- {selection.rendered_content}"
            for selection in working_memory_view.selected
            if selection.source_kind.value == "semantic"
        ]
        return "\n".join(
            [
                "Context: PROJECT-KAGYA is a private local AI assistant for subjective conversation.",
                "Private runtime data below is for tone and context only; do not quote it.",
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
