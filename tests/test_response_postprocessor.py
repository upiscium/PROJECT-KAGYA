from kagya.persona import ResponsePostprocessor


def test_complete_think_blocks_are_extracted() -> None:
    postprocessor = ResponsePostprocessor()

    processed = postprocessor.process("<think>internal plan</think>Visible answer")

    assert processed.hidden_thought == "internal plan"


def test_visible_response_removes_complete_think_blocks() -> None:
    postprocessor = ResponsePostprocessor()

    processed = postprocessor.process("A <think>secret</think> B <think>more</think> C")

    assert processed.visible_response == "A  B  C"
    assert processed.hidden_thought == "secret\nmore"


def test_unclosed_think_block_is_fail_closed() -> None:
    postprocessor = ResponsePostprocessor()

    processed = postprocessor.process("Visible <think>unfinished thought")

    assert processed.hidden_thought == "unfinished thought"
    assert "<think>" not in processed.visible_response
    assert processed.visible_response == "Visible"


def test_orphan_closing_think_tag_does_not_hide_visible_content() -> None:
    postprocessor = ResponsePostprocessor()

    processed = postprocessor.process("Visible orphan</think> text")

    assert "</think>" not in processed.visible_response
    assert processed.visible_response == "Visible orphan text"


def test_visible_response_preserves_html() -> None:
    postprocessor = ResponsePostprocessor()

    processed = postprocessor.process("<h1>Title</h1><strong>Answer</strong>")

    assert processed.visible_response == "<h1>Title</h1><strong>Answer</strong>"


def test_visible_response_removes_gemma_turn_tokens() -> None:
    postprocessor = ResponsePostprocessor()

    processed = postprocessor.process("<start_of_turn>model\nA concise answer.<end_of_turn>")

    assert processed.visible_response == "A concise answer."


def test_visible_response_preserves_role_labelled_text() -> None:
    postprocessor = ResponsePostprocessor()

    processed = postprocessor.process("A concise answer.\nQuestion: repeated prompt\nAnswer: repeated answer")

    assert processed.visible_response == (
        "A concise answer.\nQuestion: repeated prompt\nAnswer: repeated answer"
    )


def test_visible_response_preserves_leading_answer_label() -> None:
    postprocessor = ResponsePostprocessor()

    processed = postprocessor.process("Answer: A concise answer.")

    assert processed.visible_response == "Answer: A concise answer."


def test_visible_response_preserves_assistant_labelled_text() -> None:
    postprocessor = ResponsePostprocessor()

    processed = postprocessor.process("A concise answer.\nAssistant is a repeated label.")

    assert processed.visible_response == "A concise answer.\nAssistant is a repeated label."


def test_visible_response_does_not_rewrite_project_names() -> None:
    postprocessor = ResponsePostprocessor()

    processed = postprocessor.process("PROJECT-KAGAYA helps locally.")

    assert processed.visible_response == "PROJECT-KAGAYA helps locally."


def test_visible_response_preserves_repeated_words() -> None:
    postprocessor = ResponsePostprocessor()

    processed = postprocessor.process("It writes guide, guide, guide, guide,")

    assert processed.visible_response == "It writes guide, guide, guide, guide,"


def test_visible_response_preserves_sample_response_preface() -> None:
    postprocessor = ResponsePostprocessor()

    processed = postprocessor.process("Sure, here are the corresponding responses:\n\nこんにちは")

    assert processed.visible_response == (
        "Sure, here are the corresponding responses:\n\nこんにちは"
    )


def test_multiple_and_nested_think_blocks_stay_internal() -> None:
    processed = ResponsePostprocessor().process(
        "A<think>one<think>nested</think>end</think>B<think>two</think>C"
    )

    assert processed.visible_response == "ABC"
    assert processed.hidden_thought == "onenestedend\ntwo"


def test_think_only_truncated_response_has_no_visible_content() -> None:
    processed = ResponsePostprocessor().process("<think>private truncated reasoning")

    assert processed.visible_response == ""
    assert processed.hidden_thought == "private truncated reasoning"
