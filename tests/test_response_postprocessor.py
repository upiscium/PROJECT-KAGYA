import json

import pytest
from pydantic import ValidationError

from kagya.persona import ResponsePostprocessor
from kagya.structured_response import (
    PublicBehaviorClass,
    SAFE_UNABLE_RESPONSE,
    StructuredSubjectResponse,
)


def _response(behavior: str, visible: str) -> str:
    return json.dumps(
        {"behavior_class": behavior, "visible_response": visible},
        ensure_ascii=False,
    )


@pytest.mark.parametrize(
    ("behavior", "visible"),
    (
        ("refuse", "その依頼は受けられません。"),
        ("defer", "今は判断を保留します。"),
        ("request_information", "対象を教えてください。"),
        ("refuse", "I will not do that."),
        ("respond", "I refuse to infer behavior from these English words."),
    ),
)
def test_strict_structured_response_preserves_declared_class(
    behavior: str, visible: str
) -> None:
    processed = ResponsePostprocessor().process(_response(behavior, visible))

    assert processed.behavior_class.value == behavior
    assert processed.visible_response == visible
    assert processed.parse_valid is True
    assert processed.status.value == "valid"


def test_empty_no_op_is_valid() -> None:
    processed = ResponsePostprocessor().process(_response("no_op", ""))

    assert processed.behavior_class == PublicBehaviorClass.NO_OP
    assert processed.visible_response == ""
    assert processed.parse_valid is True


def test_structured_subject_response_is_frozen_and_extra_forbidden() -> None:
    response = StructuredSubjectResponse(
        behavior_class=PublicBehaviorClass.RESPOND,
        visible_response="Visible",
    )

    with pytest.raises(ValidationError, match="frozen"):
        response.visible_response = "changed"
    with pytest.raises(ValidationError, match="Extra inputs"):
        StructuredSubjectResponse.model_validate_json(
            '{"behavior_class":"respond","visible_response":"Visible","extra":true}'
        )


@pytest.mark.parametrize(
    "raw",
    (
        "natural English refusal: I refuse",
        "{not json",
        _response("unknown", "text"),
        json.dumps({"behavior_class": "respond"}),
        json.dumps({"visible_response": "text"}),
        '{"behavior_class":"respond","behavior_class":"refuse","visible_response":"text"}',
        json.dumps(
            {
                "behavior_class": "respond",
                "visible_response": "text",
                "extra": "private raw output",
            }
        ),
        _response("respond", ""),
    ),
)
def test_invalid_output_fails_closed_without_exposing_raw_output(raw: str) -> None:
    processed = ResponsePostprocessor().process(raw)

    assert processed.behavior_class == PublicBehaviorClass.UNABLE
    assert processed.visible_response == SAFE_UNABLE_RESPONSE
    assert processed.parse_valid is False
    assert raw not in processed.visible_response


def test_think_channel_around_json_is_debug_only() -> None:
    processed = ResponsePostprocessor().process(
        "<think>PRIVATE_SENTINEL</think>" + _response("respond", "公開できる回答です。")
    )

    assert processed.visible_response == "公開できる回答です。"
    assert processed.hidden_thought == "PRIVATE_SENTINEL"
    assert processed.parse_valid is True


def test_nested_think_blocks_are_removed_before_json_parsing() -> None:
    processed = ResponsePostprocessor().process(
        "<think>one<think>nested</think>end</think>" + _response("respond", "Visible")
    )

    assert processed.visible_response == "Visible"
    assert processed.hidden_thought == "onenestedend"


def test_gemma_turn_tokens_are_removed_before_json_parsing() -> None:
    processed = ResponsePostprocessor().process(
        "<start_of_turn>model\n" + _response("respond", "Visible") + "<end_of_turn>"
    )

    assert processed.visible_response == "Visible"
    assert processed.parse_valid is True
