"""Postprocess model responses into visible and internal channels."""

from dataclasses import dataclass
import re


THINK_TAG_PATTERN = re.compile(r"<(\/?)think>", flags=re.IGNORECASE)
GEMMA_TURN_TOKEN_PATTERN = re.compile(r"<(?:start|end)_of_turn>\s*(?:user|model)?", flags=re.IGNORECASE)


@dataclass(frozen=True)
class ProcessedResponse:
    visible_response: str
    hidden_thought: str


class ResponsePostprocessor:
    """Extract internal thoughts while keeping normal output clean."""

    def process(self, response_text: str) -> ProcessedResponse:
        visible_response, hidden_parts = _partition_think_channel(response_text)
        visible_response = GEMMA_TURN_TOKEN_PATTERN.sub("", visible_response).strip()
        return ProcessedResponse(
            visible_response=visible_response,
            hidden_thought="\n".join(part for part in hidden_parts if part),
        )


def _partition_think_channel(response_text: str) -> tuple[str, list[str]]:
    visible_parts: list[str] = []
    hidden_parts: list[str] = []
    hidden_buffer: list[str] = []
    depth = 0
    cursor = 0
    for match in THINK_TAG_PATTERN.finditer(response_text):
        content = response_text[cursor : match.start()]
        if depth:
            hidden_buffer.append(content)
        else:
            visible_parts.append(content)
        is_closing = bool(match.group(1))
        if is_closing:
            if depth:
                depth -= 1
                if depth == 0:
                    hidden_parts.append("".join(hidden_buffer).strip())
                    hidden_buffer = []
        else:
            depth += 1
        cursor = match.end()

    remainder = response_text[cursor:]
    if depth:
        hidden_buffer.append(remainder)
        hidden_parts.append("".join(hidden_buffer).strip())
    else:
        visible_parts.append(remainder)
    return "".join(visible_parts), hidden_parts
