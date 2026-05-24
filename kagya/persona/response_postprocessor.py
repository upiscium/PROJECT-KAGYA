"""Postprocess model responses into visible and internal channels."""

from dataclasses import dataclass
import re


THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
THINK_TAG_PATTERN = re.compile(r"</?think>", flags=re.IGNORECASE)
HTML_LIKE_TAG_PATTERN = re.compile(r"</?[A-Za-z][^>]*>")


@dataclass(frozen=True)
class ProcessedResponse:
    visible_response: str
    hidden_thought: str


class ResponsePostprocessor:
    """Extract internal thoughts while keeping normal output clean."""

    def process(self, response_text: str) -> ProcessedResponse:
        hidden_parts = [match.group(1).strip() for match in THINK_BLOCK_PATTERN.finditer(response_text)]
        visible_response = THINK_BLOCK_PATTERN.sub("", response_text)
        visible_response = THINK_TAG_PATTERN.sub("", visible_response).strip()
        visible_response = HTML_LIKE_TAG_PATTERN.sub("", visible_response).strip()
        return ProcessedResponse(
            visible_response=visible_response,
            hidden_thought="\n".join(part for part in hidden_parts if part),
        )
