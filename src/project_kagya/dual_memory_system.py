from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast


class CollectionProtocol(Protocol):
    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None: ...

    def query(
        self, *, query_texts: list[str], n_results: int
    ) -> dict[str, list[list[Any]]]: ...

    def get(self) -> dict[str, list[Any]]: ...

    def delete(self, *, ids: list[str]) -> None: ...


class ChromaClientProtocol(Protocol):
    def get_or_create_collection(
        self,
        name: str,
        embedding_function: Any | None = None,
    ) -> CollectionProtocol: ...


@dataclass(slots=True)
class MemoryHit:
    source: str
    text: str
    metadata: dict[str, Any]


class DualMemorySystem:
    def __init__(
        self,
        client: ChromaClientProtocol | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        self.client = client or self._create_client()
        self.embedding_function = embedding_function
        self.hippocampus = self.client.get_or_create_collection(
            name="hippocampus",
            embedding_function=embedding_function,
        )
        self.cortex = self.client.get_or_create_collection(
            name="cortex",
            embedding_function=embedding_function,
        )
        self._episodic_counter = 0
        self._semantic_counter = 0

    def save_episodic(
        self,
        user_input: str,
        response: str,
        valence: float,
        arousal: float,
    ) -> str:
        episodic_id = f"episodic-{self._episodic_counter}"
        self._episodic_counter += 1
        document = f"User: {user_input}\nAssistant: {response}"
        metadata = {
            "type": "episodic",
            "valence": valence,
            "arousal": arousal,
            "source": "hippocampus",
        }
        self.hippocampus.add(
            ids=[episodic_id],
            documents=[document],
            metadatas=[metadata],
        )
        return episodic_id

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        hits: list[MemoryHit] = []
        hits.extend(self._query_collection("DB1", self.hippocampus, query, top_k))
        hits.extend(self._query_collection("DB2", self.cortex, query, top_k))
        if not hits:
            return ""
        return "\n\n".join(
            f"[{hit.source}] {hit.text}\nmeta={hit.metadata}" for hit in hits
        )

    def consolidate_to_semantic(self, llm_pipeline: Any) -> list[str]:
        records = self.hippocampus.get()
        ids = list(records.get("ids", []))
        documents = list(records.get("documents", []))
        metadatas = list(records.get("metadatas", []))
        saved_ids: list[str] = []

        for record_id, document, metadata in zip(
            ids, documents, metadatas, strict=False
        ):
            prompt = self._build_consolidation_prompt(document, metadata)
            semantic_text = self._invoke_pipeline(llm_pipeline, prompt)
            semantic_id = f"semantic-{self._semantic_counter}"
            self._semantic_counter += 1
            self.cortex.add(
                ids=[semantic_id],
                documents=[semantic_text],
                metadatas=[{**metadata, "type": "semantic", "source_id": record_id}],
            )
            self.hippocampus.delete(ids=[record_id])
            saved_ids.append(semantic_id)

        return saved_ids

    def _query_collection(
        self,
        source: str,
        collection: CollectionProtocol,
        query: str,
        top_k: int,
    ) -> list[MemoryHit]:
        result = collection.query(query_texts=[query], n_results=top_k)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        hits: list[MemoryHit] = []
        for document, metadata in zip(documents, metadatas, strict=False):
            hits.append(
                MemoryHit(source=source, text=str(document), metadata=dict(metadata))
            )
        return hits

    def _build_consolidation_prompt(
        self, document: str, metadata: dict[str, Any]
    ) -> str:
        return (
            "Extract durable facts about the user from this episode.\n"
            f"Episode: {document}\n"
            f"Metadata: {metadata}\n"
            "Return a concise semantic memory."
        )

    def _invoke_pipeline(self, llm_pipeline: Any, prompt: str) -> str:
        if hasattr(llm_pipeline, "invoke"):
            result = llm_pipeline.invoke(prompt)
        elif callable(llm_pipeline):
            result = llm_pipeline(prompt)
        else:
            raise TypeError("llm_pipeline must be callable")
        return self._extract_text(result)

    def _extract_text(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict) and "text" in result:
            return str(result["text"])
        content = getattr(result, "content", None)
        if content is not None:
            return str(cast(Any, content))
        return str(result)

    def _create_client(self) -> ChromaClientProtocol:
        try:
            import chromadb  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - defensive fallback
            raise RuntimeError("chromadb is required for DualMemorySystem") from exc

        return chromadb.Client()
