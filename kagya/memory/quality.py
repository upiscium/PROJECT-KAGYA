"""Deterministic quality checks for generated episodic memories."""

from collections import Counter
import math
import re

from kagya.memory.memory_schema import GenerationHealth


PROMPT_LEAKAGE = re.compile(
    r"(?:^|\n)(?:Context|Instruction|System|User|Assistant):", re.IGNORECASE
)


def assess_generation_health(
    response: str,
    *,
    loss: float,
    fallback_used: bool,
) -> GenerationHealth:
    reasons: list[str] = []
    stripped = response.strip()
    repetitive = _is_repetitive(stripped)
    prompt_leakage = bool(PROMPT_LEAKAGE.search(stripped))
    non_finite = not math.isfinite(loss)
    if not stripped:
        reasons.append("empty")
    if repetitive:
        reasons.append("repetitive")
    if prompt_leakage:
        reasons.append("prompt_leakage")
    if non_finite:
        reasons.append("non_finite_score")
    if fallback_used:
        reasons.append("fallback_used")
    unhealthy = any(
        reason in {"empty", "repetitive", "prompt_leakage", "non_finite_score"}
        for reason in reasons
    )
    return GenerationHealth(
        healthy=not unhealthy,
        reasons=reasons,
        repetitive=repetitive,
        prompt_leakage=prompt_leakage,
        non_finite_score=non_finite,
        fallback_used=fallback_used,
    )


def _is_repetitive(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 3 and max(Counter(lines).values(), default=0) >= 3:
        return True
    chunks = re.findall(r"[\w\u3040-\u30ff\u3400-\u9fff]+", text.lower())
    if len(chunks) < 12:
        return False
    trigrams = list(zip(chunks, chunks[1:], chunks[2:], strict=False))
    return max(Counter(trigrams).values(), default=0) >= 3
