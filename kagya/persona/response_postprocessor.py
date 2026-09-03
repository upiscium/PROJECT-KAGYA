"""Postprocess model responses into visible and ephemeral-private channels."""

from dataclasses import dataclass
import re


THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
OPEN_THINK_TAG_PATTERN = re.compile(r"<think>", flags=re.IGNORECASE)
CLOSE_THINK_TAG_PATTERN = re.compile(r"</think>", flags=re.IGNORECASE)
THINK_TAG_PATTERN = re.compile(r"</?think>", flags=re.IGNORECASE)
HTML_LIKE_TAG_PATTERN = re.compile(r"</?[A-Za-z][^>]*>")
PROMPT_LABEL_ECHO_PATTERN = re.compile(
    r"\n(?:Question|User|Assistant|Answer|Context|Instruction):", flags=re.IGNORECASE
)
ASSISTANT_SELF_ECHO_PATTERN = re.compile(
    r"\nAssistant\b.*", flags=re.IGNORECASE | re.DOTALL
)
LEADING_ANSWER_LABEL_PATTERN = re.compile(r"^Answer:\s*", flags=re.IGNORECASE)
PROJECT_NAME_VARIANT_PATTERN = re.compile(
    r"PROJECT-KAGAY?A|Project-Kageye", flags=re.IGNORECASE
)
REPEATED_COMMA_WORD_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z-]*)(?:,\s*\1\b){2,}", flags=re.IGNORECASE
)


@dataclass(frozen=True)
class ProcessedResponse:
    visible_response: str
    hidden_thought: str


class ResponsePostprocessor:
    """Separate visible output from request-scoped private model output."""

    def process(self, response_text: str) -> ProcessedResponse:
        hidden_parts = [
            match.group(1).strip() for match in THINK_BLOCK_PATTERN.finditer(response_text)
        ]
        visible_response = THINK_BLOCK_PATTERN.sub("", response_text)

        # An unterminated opening tag makes the rest of the generation ambiguous.
        # Treat that tail as private rather than leaking it into visible output.
        opening = OPEN_THINK_TAG_PATTERN.search(visible_response)
        if opening is not None:
            hidden_tail = visible_response[opening.end() :].strip()
            if hidden_tail:
                hidden_parts.append(THINK_TAG_PATTERN.sub("", hidden_tail).strip())
            visible_response = visible_response[: opening.start()]

        # If only a closing tag survives, the prefix may be truncated private output.
        # Keep only the suffix after the boundary.
        closing = CLOSE_THINK_TAG_PATTERN.search(visible_response)
        if closing is not None:
            visible_response = visible_response[closing.end() :]

        visible_response = THINK_TAG_PATTERN.sub("", visible_response).strip()
        visible_response = HTML_LIKE_TAG_PATTERN.sub("", visible_response).strip()
        visible_response = PROMPT_LABEL_ECHO_PATTERN.split(
            visible_response, maxsplit=1
        )[0].strip()
        visible_response = ASSISTANT_SELF_ECHO_PATTERN.sub("", visible_response).strip()
        visible_response = LEADING_ANSWER_LABEL_PATTERN.sub("", visible_response).strip()
        visible_response = PROJECT_NAME_VARIANT_PATTERN.sub(
            "PROJECT-KAGYA", visible_response
        ).strip()
        visible_response = REPEATED_COMMA_WORD_PATTERN.sub(
            r"\1", visible_response
        ).strip(" ,")
        return ProcessedResponse(
            visible_response=visible_response,
            hidden_thought="\n".join(part for part in hidden_parts if part),
        )
