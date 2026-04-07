from project_kagya.dual_memory_system import DualMemorySystem


def test_save_retrieve_and_consolidate_memory() -> None:
    memory = DualMemorySystem(top_k=2)
    memory.save_episodic("I like tea", "Tea is nice", 0.3, 0.4)
    memory.save_episodic("I like coffee", "Coffee is strong", -0.2, 0.8)
    memory.cortex.add(["semantic-1"], ["I like tea"], [{"type": "semantic"}])

    context = memory.retrieve_context("tea")
    assert "Episodic Memory" in context
    assert "Semantic Memory" in context

    moved = memory.consolidate_to_semantic(lambda prompt: {"facts": "User likes tea"})
    assert len(moved) == 2
    assert not memory.hippocampus.records
    assert any(
        record.text == "User likes tea" for record in memory.cortex.records.values()
    )
