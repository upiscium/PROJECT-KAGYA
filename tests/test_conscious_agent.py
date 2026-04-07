from project_kagya.conscious_agent import ConsciousAgent


def test_conscious_agent_builds_prompt_with_required_fields() -> None:
    agent = ConsciousAgent()
    prompt = agent.build_prompt("hello", 0.25, 0.75, "[Episodic Memory]\n- hello")

    assert "Current Valence" in prompt.system
    assert "Current Arousal" in prompt.system
    assert "<think>...</think>" in prompt.system
    assert "Episodic Memory" in prompt.system
