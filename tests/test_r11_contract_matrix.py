"""Explicit audit references for the whole R11 F1-F18 regression contract.

The referenced tests are intentionally kept in their responsibility-oriented
modules. This matrix gives focused and whole-R11 reviewers one stable index of
the evidence without adding a second authority or duplicating those tests.
"""

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
        "tests/test_value_system.py::test_prompt_view_preserves_system_authority_conflicts_and_bounds_concept",
    ),
    (
        "F4",
        "tests/test_agent_state.py::test_v5_seed_drift_fails_closed_without_rewriting_snapshot",
    ),
    (
        "F5",
        "tests/test_value_system.py::test_origin_review_preserves_lineage_and_never_endorses",
    ),
    (
        "F6",
        "tests/test_value_system.py::test_protectedness_raises_reversal_threshold",
    ),
    (
        "F7",
        "tests/test_value_system.py::test_replayed_evidence_makes_the_whole_event_idempotent",
    ),
    (
        "F8",
        "tests/test_value_system.py::test_prompt_view_selects_active_applicable_values_without_mutation",
    ),
    (
        "F9",
        "tests/test_fastapi_backend.py::test_value_governance_failure_restores_the_last_committed_view",
    ),
    (
        "F10",
        "tests/test_fastapi_backend.py::test_value_governance_commit_is_published_before_finalization_failure",
    ),
    (
        "F11",
        "tests/test_fastapi_backend.py::test_retained_v4_lazy_upgrade_preserves_bytes_then_publishes_v5",
    ),
    (
        "F12",
        "tests/test_agent_state.py::test_v5_noncanonical_value_order_fails_closed_without_rewrite",
    ),
    (
        "F13",
        "tests/test_fastapi_backend.py::test_values_api_reads_are_pure_and_governance_is_runtime_bound",
    ),
    (
        "F14",
        "tests/test_prompt_builder.py::test_build_renders_bounded_active_value_projection_deterministically; "
        "tests/test_fastapi_backend.py::test_values_api_reads_are_pure_and_governance_is_runtime_bound",
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
