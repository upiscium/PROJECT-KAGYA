"""Build-injected and development source provenance resolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import os
from pathlib import Path
import re
import subprocess


class SourceRevisionStatus(StrEnum):
    VERIFIED = "verified"
    UNKNOWN = "unknown"
    DIRTY = "dirty"


@dataclass(frozen=True)
class SourceBuildInfo:
    commit_sha: str | None
    tree_hash: str | None
    build_id: str | None
    status: SourceRevisionStatus


_SHA40 = re.compile(r"[0-9a-f]{40}")


def resolve_source_build_info(root: Path | None = None) -> SourceBuildInfo:
    """Resolve source identity without retaining or exposing dirty file names."""

    injected_commit = os.environ.get("KAGYA_SOURCE_COMMIT_SHA")
    injected_tree = os.environ.get("KAGYA_SOURCE_TREE_HASH")
    build_id = os.environ.get("KAGYA_BUILD_ID")
    if injected_commit is not None or injected_tree is not None or build_id is not None:
        verified = bool(
            injected_commit
            and _SHA40.fullmatch(injected_commit)
            and injected_tree
            and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", injected_tree)
        )
        return SourceBuildInfo(
            commit_sha=injected_commit
            if injected_commit and _SHA40.fullmatch(injected_commit)
            else None,
            tree_hash=injected_tree
            if injected_tree
            and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", injected_tree)
            else None,
            build_id=build_id[:128] if build_id else None,
            status=SourceRevisionStatus.VERIFIED
            if verified
            else SourceRevisionStatus.UNKNOWN,
        )
    repository = (root or Path(__file__).resolve().parents[1]).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True, timeout=5
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repository, text=True, timeout=5
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=repository,
                text=True,
                timeout=5,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        return SourceBuildInfo(None, None, build_id, SourceRevisionStatus.UNKNOWN)
    if not _SHA40.fullmatch(commit) or not _SHA40.fullmatch(tree):
        return SourceBuildInfo(None, None, build_id, SourceRevisionStatus.UNKNOWN)
    return SourceBuildInfo(
        commit,
        tree,
        build_id,
        SourceRevisionStatus.DIRTY if dirty else SourceRevisionStatus.VERIFIED,
    )
