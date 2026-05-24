"""Helpers for transforming episodic records into semantic prompts."""

from kagya.memory.memory_schema import EpisodicMemoryRecord


def build_consolidation_prompt(record: EpisodicMemoryRecord) -> str:
    """Build a small prompt for extracting stable semantic memory."""

    return (
        "Extract one concise semantic memory from this episode.\n"
        f"User: {record.user_input}\n"
        f"Assistant: {record.response}"
    )
