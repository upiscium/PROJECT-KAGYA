from kagya.api.redaction import REDACTED_VALUE, redact_private_fields


def test_redact_private_fields_handles_nested_payloads() -> None:
    payload = {
        "adapter_id": "adapter-a",
        "prompt": "raw prompt",
        "nested": {
            "hidden_thought": "private thought",
            "items": [
                {"retrieved_memory": {"db1_results": ["private"]}},
                {"safe": "visible"},
            ],
        },
        "safe": "visible",
    }

    redacted = redact_private_fields(payload)

    assert redacted == {
        "adapter_id": "adapter-a",
        "prompt": REDACTED_VALUE,
        "nested": {
            "hidden_thought": REDACTED_VALUE,
            "items": [
                {"retrieved_memory": REDACTED_VALUE},
                {"safe": "visible"},
            ],
        },
        "safe": "visible",
    }


def test_redact_private_fields_does_not_mutate_input() -> None:
    payload = {"prompt": "raw prompt", "safe": {"hidden_thought": "private"}}

    redact_private_fields(payload)

    assert payload == {"prompt": "raw prompt", "safe": {"hidden_thought": "private"}}
