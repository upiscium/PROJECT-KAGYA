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
                "You are PROJECT-KAGYA, a subjective AI architecture.",
                "Use prediction error, emotion, and memory to answer.",
                "Internal thought must be wrapped in <think>...</think> and is not normally visible to the user.",
                "After the think block, provide the final visible answer.",
                "",
                "Current emotion state:",
                f"- valence: {emotion_state.valence:.6f}",
                f"- arousal: {emotion_state.arousal:.6f}",
                f"- optimal_loss: {emotion_state.optimal_loss:.6f}",
                "",
                "Related DB1 episodic memories:",
                *(db1_lines or ["- none"]),
                "",
                "Related DB2 semantic memories:",
                *(db2_lines or ["- none"]),
                "",
                f"User input: {user_input}",
            ]
        )
