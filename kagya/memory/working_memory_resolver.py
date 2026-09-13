"""Bridge committed dual-memory reads into the runtime resolution contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kagya.runtime.working_memory import (
    WorkingMemoryItem,
    WorkingMemoryResolution,
    WorkingMemoryResolutionStatus,
    WorkingMemorySourceKind,
)

if TYPE_CHECKING:
    from kagya.memory.dual_memory_system import DualMemorySystem


class MemoryWorkingMemoryResolver:
    """Resolve Working Memory references using only committed domain reads."""

    def __init__(self, memory: DualMemorySystem) -> None:
        self._memory = memory

    def __call__(self, item: WorkingMemoryItem) -> WorkingMemoryResolution:
        return self.resolve(item)

    def resolve(self, item: WorkingMemoryItem) -> WorkingMemoryResolution:
        if not isinstance(item, WorkingMemoryItem):
            raise TypeError("item must be WorkingMemoryItem")

        # Importing the domain exceptions here keeps the bridge out of the
        # memory package's public initializer and avoids an import cycle.
        from kagya.memory.dual_memory_system import (
            EpisodicMemoryFormatError,
            EpisodicMemoryReadError,
            SemanticMemoryFormatError,
            SemanticMemoryReadError,
        )

        if item.source_kind is WorkingMemorySourceKind.EPISODIC:
            try:
                committed = self._memory.get_committed_episodic(item.source_id)
            except EpisodicMemoryReadError:
                return WorkingMemoryResolution(
                    WorkingMemoryResolutionStatus.UNAVAILABLE
                )
            except EpisodicMemoryFormatError:
                return WorkingMemoryResolution(WorkingMemoryResolutionStatus.MALFORMED)
            if committed is None:
                return WorkingMemoryResolution(WorkingMemoryResolutionStatus.MISSING)
            if committed.record.archived:
                return WorkingMemoryResolution(WorkingMemoryResolutionStatus.ARCHIVED)
            return WorkingMemoryResolution(
                WorkingMemoryResolutionStatus.RESOLVED, committed.document
            )

        if item.source_kind is WorkingMemorySourceKind.SEMANTIC:
            try:
                semantic = self._memory.get_committed_semantic(item.source_id)
            except SemanticMemoryReadError:
                return WorkingMemoryResolution(
                    WorkingMemoryResolutionStatus.UNAVAILABLE
                )
            except SemanticMemoryFormatError:
                return WorkingMemoryResolution(WorkingMemoryResolutionStatus.MALFORMED)
            if semantic is None:
                return WorkingMemoryResolution(WorkingMemoryResolutionStatus.MISSING)
            return WorkingMemoryResolution(
                WorkingMemoryResolutionStatus.RESOLVED, semantic.document
            )

        raise TypeError("Working Memory source kind is unsupported")
