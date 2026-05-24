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


def test_malformed_think_tags_do_not_crash_or_leave_tag_residue() -> None:
    postprocessor = ResponsePostprocessor()

    processed = postprocessor.process("Visible <think>unfinished thought")

    assert processed.hidden_thought == ""
    assert "<think>" not in processed.visible_response
    assert processed.visible_response == "Visible unfinished thought"


def test_visible_response_contains_no_closing_think_tag_residue() -> None:
    postprocessor = ResponsePostprocessor()

    processed = postprocessor.process("Visible orphan</think> text")

    assert "</think>" not in processed.visible_response
    assert processed.visible_response == "Visible orphan text"


def test_visible_response_removes_html_like_tag_residue() -> None:
    postprocessor = ResponsePostprocessor()

    processed = postprocessor.process("<h1>Title</h1><strong>Answer</strong>")

    assert processed.visible_response == "TitleAnswer"
