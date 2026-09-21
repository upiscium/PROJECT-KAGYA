"""Explicit audit references for the whole R11 F1-F18 regression contract.

The referenced tests are intentionally kept in their responsibility-oriented
modules. This matrix gives focused and whole-R11 reviewers one stable index of
the evidence without adding a second authority or duplicating those tests.
"""

import ast
from pathlib import Path


R11_F_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "F1",
        "tests/test_fastapi_backend.py::test_chat_and_appraisal_leave_value_system_unchanged_in_u3",
    ),
    (
        "F2",
        "tests/test_fastapi_backend.py::test_values_api_origin_review_and_seed_adoption_are_narrow",
    ),
    (
        "F3",
        "tests/test_value_system.py::test_prompt_view_selects_active_applicable_values_without_mutation; "
        "tests/test_value_system.py::test_prompt_view_preserves_system_authority_conflicts_and_bounds_concept",
    ),
    (
        "F4",
        "tests/test_agent_state.py::test_v5_seed_drift_fails_closed_without_rewriting_snapshot",
    ),
    (
        "F5",
        "tests/test_value_system.py::test_origin_review_preserves_lineage_and_never_endorses; "
        "tests/test_value_system.py::test_origin_review_rejects_self_and_system_lineage; "
        "tests/test_value_system.py::test_prompt_view_selects_active_applicable_values_without_mutation",
    ),
    (
        "F6",
        "tests/test_value_system.py::test_protectedness_raises_reversal_threshold",
    ),
    (
        "F7",
        "tests/test_value_system.py::test_support_and_opposition_policy_is_bounded_and_deterministic; "
        "tests/test_value_system.py::test_duplicate_targets_and_duplicate_evidence_are_fail_closed",
    ),
    (
        "F8",
        "tests/test_value_system.py::test_prompt_view_selects_active_applicable_values_without_mutation",
    ),
    (
        "F9",
        "tests/test_fastapi_backend.py::test_value_governance_failure_restores_the_last_committed_view; "
        "tests/test_agent_state.py::test_v5_restore_failure_rolls_back_value_authority_with_other_five",
    ),
    (
        "F10",
        "tests/test_fastapi_backend.py::test_value_governance_commit_is_published_before_finalization_failure; "
        "tests/test_fastapi_backend.py::test_finalization_failure_preserves_internal_commit_without_restore; "
        "tests/test_state_recovery.py::test_committed_before_crash_v5_reconstructs_value_without_replay",
    ),
    (
        "F11",
        "tests/test_fastapi_backend.py::test_retained_v4_lazy_upgrade_preserves_bytes_then_publishes_v5",
    ),
    (
        "F12",
        "tests/test_agent_state.py::test_v5_noncanonical_value_order_fails_closed_without_rewrite; "
        "tests/test_agent_state.py::test_v5_origin_witness_rejects_provenance_tampering_without_rewrite; "
        "tests/test_agent_state.py::test_current_schema_rejects_unknown_root_and_nested_fields; "
        "tests/test_agent_state.py::test_existing_corrupt_snapshot_never_defaults_or_changes; "
        "tests/test_agent_state.py::test_future_version_is_distinct_and_never_defaults; "
        "tests/test_state_wal.py::test_future_record_version_is_rejected; "
        "tests/test_bootstrap_config.py::test_value_seed_bounds_and_finiteness; "
        "tests/test_bootstrap_config.py::test_value_seed_ids_must_be_unique; "
        "tests/test_value_system.py::test_scalar_bounds_are_strict; "
        "tests/test_value_system.py::test_revision_requires_a_nonnegative_exact_integer; "
        "tests/test_value_system.py::test_refs_conflicts_and_non_authoritative_types; "
        "tests/test_value_system.py::test_restore_rejects_current_history_and_ledger_inconsistency; "
        "tests/test_value_system.py::test_restore_rejects_surplus_ledger_ref_with_exact_digest_witness; "
        "tests/test_value_system.py::test_revision_history_rejects_broken_state_continuity",
    ),
    (
        "F13",
        "tests/test_fastapi_backend.py::test_values_api_reads_are_pure_and_governance_is_runtime_bound; "
        "tests/test_value_system.py::test_prompt_view_selects_active_applicable_values_without_mutation",
    ),
    (
        "F14",
        "tests/test_prompt_builder.py::test_build_renders_bounded_active_value_projection_deterministically; "
        "tests/test_fastapi_backend.py::test_values_api_reads_are_pure_and_governance_is_runtime_bound; "
        "tests/test_agent_state.py::test_v5_canonical_snapshot_contains_value_authority_but_no_prompt_or_independent_store_data; "
        "tests/test_state_wal.py::test_private_sentinel_and_bounded_errors; "
        "tests/test_event_journal.py::test_schema_is_strict_and_has_no_metadata_escape_hatch; "
        "tests/test_event_journal.py::test_u1_f11_invalid_transaction_fields_are_bounded_and_private_free",
    ),
    (
        "F15",
        "R11 production-scope audit: no later-domain authority is introduced",
    ),
    (
        "F16",
        "tests/test_fastapi_backend.py::test_api_chat_works_with_dummy_provider_without_debug_leak",
    ),
    (
        "F17",
        "tests/test_value_system.py::test_prompt_view_preserves_system_authority_conflicts_and_bounds_concept",
    ),
    (
        "F18",
        "tests/test_value_system.py::test_revision_state_and_record_digests_are_canonical",
    ),
)


def test_r11_f_matrix_is_complete_and_explicit() -> None:
    assert tuple(item[0] for item in R11_F_MATRIX) == tuple(
        f"F{index}" for index in range(1, 19)
    )
    assert all(reference for _item, reference in R11_F_MATRIX)


def test_r11_f_matrix_references_existing_test_evidence() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    for requirement, references in R11_F_MATRIX:
        for reference in references.split("; "):
            if "::" not in reference:
                continue
            test_path, test_name = reference.split("::", 1)
            path = repository_root / test_path
            assert path.is_file(), f"{requirement} references missing file {test_path}"
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            names = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assert test_name in names, (
                f"{requirement} references missing test "
                f"{test_path}::{test_name}"
            )
