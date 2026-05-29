# 04 Dual Memory System

## Goal

Implement initial ChromaDB-backed Dual Memory with DB1 episodic memory and DB2 semantic memory.

## Target Files

- `kagya/memory/__init__.py`
- `kagya/memory/memory_schema.py`
- `kagya/memory/dual_memory_system.py`
- `kagya/memory/memory_evaluator.py`
- `kagya/memory/consolidation.py`
- `tests/test_dual_memory_system.py`

## Data Model Requirements

- Define `MemoryRecordType` with `episodic_log`, `thought_log`, `extracted_fact`, `semantic_memory`, and `evaluation_log`.
- Define `EpisodicMemoryRecord` dataclass for DB1.
- Define `SemanticMemoryRecord` dataclass for DB2.
- Define `MemoryContext` containing DB1 and DB2 retrieval results.
- Store logical memory layer through `record_type`, not separate physical databases.

## DualMemorySystem Requirements

- Use ChromaDB persistent storage from configuration.
- Use DB1 collection name `hippocampus` from configuration.
- Use DB2 collection name `cortex` from configuration.
- Implement `save_episodic(...) -> str`.
- Implement `retrieve_context(query: str) -> MemoryContext`.
- Implement `consolidate_to_semantic(model_provider: ModelProvider) -> list[str]`.
- Archive DB1 records with `archived=true` after consolidation when needed.
- Do not physically delete DB1 records in the initial implementation.

## Test Requirements

- Saving an episodic record returns an episode ID.
- Saved episodic records can be retrieved from DB1.
- Semantic records can be retrieved from DB2.
- Consolidation archives DB1 records instead of deleting them.
- Retrieval respects configured DB1 and DB2 top-k values.
- Tests use isolated temporary ChromaDB directories.

## Completion Criteria

- Memory tests pass without requiring a real embedding model when using deterministic test embeddings or Chroma defaults suitable for tests.
- DB1 physical deletion is absent from implementation.
