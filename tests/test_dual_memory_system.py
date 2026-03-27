from __future__ import annotations

from dataclasses import dataclass

from project_kagya.dual_memory_system import DualMemorySystem


class FakeCollection:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, object]]] = []

    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, object]],
    ) -> None:
        for record_id, document, metadata in zip(
            ids, documents, metadatas, strict=False
        ):
            self.records.append((record_id, document, metadata))

    def query(self, *, query_texts: list[str], n_results: int) -> dict[str, object]:
        del query_texts
        documents = [[document for _, document, _ in self.records[:n_results]]]
        metadatas = [[metadata for _, _, metadata in self.records[:n_results]]]
        return {"documents": documents, "metadatas": metadatas}

    def get(self) -> dict[str, list[object]]:
        return {
            "ids": [record_id for record_id, _, _ in self.records],
            "documents": [document for _, document, _ in self.records],
            "metadatas": [metadata for _, _, metadata in self.records],
        }

    def delete(self, *, ids: list[str]) -> None:
        self.records = [record for record in self.records if record[0] not in ids]


@dataclass
class FakeClient:
    hippocampus: FakeCollection
    cortex: FakeCollection

    def get_or_create_collection(self, name: str, embedding_function=None):
        del embedding_function
        if name == "hippocampus":
            return self.hippocampus
        return self.cortex


def test_save_and_retrieve_context() -> None:
    client = FakeClient(FakeCollection(), FakeCollection())
    memory = DualMemorySystem(client=client)

    memory.save_episodic("hello", "hi", 0.4, 0.7)

    context = memory.retrieve_context("hello")

    assert "[DB1] User: hello" in context
    assert "valence" in context


def test_consolidate_moves_records_to_cortex() -> None:
    client = FakeClient(FakeCollection(), FakeCollection())
    memory = DualMemorySystem(client=client)
    memory.save_episodic("I like tea", "Noted", 0.8, 0.9)

    class FakePipeline:
        def __call__(self, prompt: str) -> str:
            assert "Extract durable facts" in prompt
            return "User likes tea."

    saved_ids = memory.consolidate_to_semantic(FakePipeline())

    assert saved_ids == ["semantic-0"]
    assert client.hippocampus.records == []
    assert client.cortex.records[0][1] == "User likes tea."
