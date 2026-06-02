"""Prompt construction for the conscious runtime loop."""

from kagya.body import EmotionState
from kagya.memory import MemoryContext


class PromptBuilder:
    """Build prompts from input, emotion, and retrieved memory."""

    def build(
        self,
        user_input: str,
        emotion_state: EmotionState,
        memory_context: MemoryContext,
    ) -> str:
        db1_lines = [
            f"- User: {record.user_input} | Assistant: {record.response}"
            for record in memory_context.db1_results
        ]
        db2_lines = [f"- {record.text}" for record in memory_context.db2_results]
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
                *(db1_lines or ["- none"]),
                "",
                "Semantic memories:",
                *(db2_lines or ["- none"]),
                "",
                f"User: {user_input}",
                "Assistant:",
            ]
        )
