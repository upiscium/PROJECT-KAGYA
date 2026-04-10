from __future__ import annotations

from project_kagya.thought_quality_assurance import (
    ThoughtExample,
    ThoughtQualityAssurer,
)


def test_validate_example_accepts_aligned_thought() -> None:
    assurer = ThoughtQualityAssurer()
    example = ThoughtExample(
        input="user wants tea",
        thought="consider the tea preference",
        output="I will mention tea",
        source_ids=["1"],
        confidence=0.9,
        status="confirmed",
    )

    report = assurer.validate_example(example)

    assert report.valid is True
    assert report.score >= assurer.min_score
    assert report.reasons == []


def test_validate_example_rejects_empty_thought() -> None:
    assurer = ThoughtQualityAssurer()
    example = ThoughtExample(
        input="user wants tea",
        thought="",
        output="I will mention tea",
        source_ids=["1"],
        confidence=0.9,
        status="confirmed",
    )

    report = assurer.validate_example(example)

    assert report.valid is False
    assert "thought is empty" in report.reasons


def test_filter_examples_keeps_only_valid_examples() -> None:
    assurer = ThoughtQualityAssurer()
    valid = ThoughtExample(
        input="user wants tea",
        thought="consider the tea preference",
        output="I will mention tea",
        source_ids=["1"],
        confidence=0.9,
        status="confirmed",
    )
    invalid = ThoughtExample(
        input="user wants tea",
        thought="",
        output="I will mention tea",
        source_ids=["2"],
        confidence=0.9,
        status="confirmed",
    )

    filtered = assurer.filter_examples([valid, invalid])

    assert filtered == [valid]


def test_score_thought_returns_zero_for_empty_text() -> None:
    assurer = ThoughtQualityAssurer()

    assert assurer.score_thought("") == 0.0
