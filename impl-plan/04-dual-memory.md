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
- `thought_log` is a historical logical label, not authority to retain hidden/private model reasoning. Any record under that label must still satisfy the R02 public/persistable-data boundary.
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
- Store only visible/structured episodic data in DB1; hidden/private reasoning is forbidden in both documents and metadata, including nested or extra metadata.
- Preserve archive semantics and avoid physical deletion in the ordinary episodic lifecycle.
- As a one-way privacy migration exception, allow sanitization or physical recreation of pre-R02 DB1 storage when required to remove legacy private data. Migration failure must not silently retain such data as a valid record.

## Test Requirements

- Saving an episodic record returns an episode ID.
- Saved episodic records can be retrieved from DB1.
- Semantic records can be retrieved from DB2.
- Consolidation archives DB1 records instead of deleting them.
- New DB1 records reject hidden/private reasoning in documents and metadata.
- Reopening pre-R02 DB1 records deterministically sanitizes hidden/private data, including nested and extra metadata.
- Retrieval respects configured DB1 and DB2 top-k values.
- Tests use isolated temporary ChromaDB directories.

## Completion Criteria

- Memory tests pass without requiring a real embedding model when using deterministic test embeddings or Chroma defaults suitable for tests.
- Ordinary DB1 lifecycle operations use archive semantics; physical recreation is limited to the one-way R02 privacy migration.
- R03 and later runtime/persistence implementations cannot weaken this zero-private-persistence boundary.
