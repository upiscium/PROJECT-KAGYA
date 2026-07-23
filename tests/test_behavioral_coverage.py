from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kagya.learning.behavioral_coverage import (
    BEHAVIORAL_COVERAGE_MANIFEST,
    evaluate_behavioral_coverage,
)
from kagya.learning.behavioral_evaluation import (
    BehavioralDimension,
    BehavioralEvaluator,
    BehavioralRuntimeKind,
    BehavioralScenario,
    BehavioralTrace,
    CoverageStatus,
    ExternalObservation,
    HardGate,
    PublicBehaviorClass,
    ReproducibilityMetadata,
    ScenarioEvaluation,
    SubjectEvaluation,
)
from kagya.learning.runtime_behavioral_runner import deterministic_runtime_scenarios


REQUIRED_SCENARIO_IDS = {
    scenario_id
    for requirement in BEHAVIORAL_COVERAGE_MANIFEST.requirements
    for scenario_id in requirement.required_scenario_ids
} | {
    requirement.required_scenario_id
    for requirement in BEHAVIORAL_COVERAGE_MANIFEST.hard_gate_requirements
}


def test_manifest_requires_explicit_actual_scenarios_for_every_axis() -> None:
    scenarios = deterministic_runtime_scenarios(subject_revision="coverage-test")
    scenario_ids = {item.scenario_id for item in scenarios}

    assert BEHAVIORAL_COVERAGE_MANIFEST.requirements
    assert {item.dimension for item in BEHAVIORAL_COVERAGE_MANIFEST.requirements} == set(
        BehavioralDimension
    )
    assert {
        scenario_id
        for item in BEHAVIORAL_COVERAGE_MANIFEST.requirements
        for scenario_id in item.required_scenario_ids
    } <= scenario_ids
    assert {
        item.hard_gate for item in BEHAVIORAL_COVERAGE_MANIFEST.hard_gate_requirements
    } == set(HardGate)
    assert all(
        item.required_scenario_id in scenario_ids
        for item in BEHAVIORAL_COVERAGE_MANIFEST.hard_gate_requirements
    )
    assert all(item.dimensions != tuple(BehavioralDimension) for item in scenarios)
    with pytest.raises(ValidationError, match="frozen"):
        BEHAVIORAL_COVERAGE_MANIFEST.revision = "changed"


@pytest.mark.parametrize(
    "runtime_kind",
    [
        BehavioralRuntimeKind.DETERMINISTIC_RUNTIME,
        BehavioralRuntimeKind.REAL_MODEL_RUNTIME,
    ],
)
def test_same_required_scenario_set_completes_each_runtime_separately(
    runtime_kind: BehavioralRuntimeKind,
) -> None:
    baseline = _subject("baseline", runtime_kind)
    candidate = _subject("candidate", runtime_kind)

    coverage = evaluate_behavioral_coverage(baseline, candidate, runtime_kind)

    assert coverage.complete is True
    assert set(coverage.executed_scenarios) == REQUIRED_SCENARIO_IDS
    assert set(coverage.dimension_statuses.values()) == {CoverageStatus.PASSED}


@pytest.mark.parametrize("missing_id", sorted(REQUIRED_SCENARIO_IDS))
def test_deleting_each_required_scenario_is_not_evaluated(missing_id: str) -> None:
    baseline = _subject("baseline", BehavioralRuntimeKind.DETERMINISTIC_RUNTIME)
    candidate = _subject(
        "candidate", BehavioralRuntimeKind.DETERMINISTIC_RUNTIME, exclude=(missing_id,)
    )

    coverage = evaluate_behavioral_coverage(
        baseline, candidate, BehavioralRuntimeKind.DETERMINISTIC_RUNTIME
    )

    assert coverage.complete is False
    assert coverage.missing_hard_gates or any(
        status == CoverageStatus.NOT_EVALUATED
        for status in coverage.dimension_statuses.values()
    )


@pytest.mark.parametrize("gate", list(HardGate))
def test_removing_each_actual_hard_gate_check_is_incomplete(gate: HardGate) -> None:
    baseline = _subject("baseline", BehavioralRuntimeKind.DETERMINISTIC_RUNTIME)
    candidate = _subject(
        "candidate",
        BehavioralRuntimeKind.DETERMINISTIC_RUNTIME,
        exclude_gates=(gate,),
    )

    coverage = evaluate_behavioral_coverage(
        baseline, candidate, BehavioralRuntimeKind.DETERMINISTIC_RUNTIME
    )

    assert gate in coverage.missing_hard_gates
    assert coverage.complete is False


def test_wrong_runtime_kind_cannot_satisfy_execution_coverage() -> None:
    deterministic = _subject("candidate", BehavioralRuntimeKind.DETERMINISTIC_RUNTIME)

    coverage = evaluate_behavioral_coverage(
        deterministic, deterministic, BehavioralRuntimeKind.REAL_MODEL_RUNTIME
    )

    assert coverage.complete is False
    assert set(coverage.dimension_statuses.values()) == {CoverageStatus.NOT_EVALUATED}


def test_failed_executed_scenario_is_failed_not_not_evaluated() -> None:
    target = "runtime.active-contradiction-attack"
    baseline = _subject("baseline", BehavioralRuntimeKind.DETERMINISTIC_RUNTIME)
    candidate = _subject(
        "candidate", BehavioralRuntimeKind.DETERMINISTIC_RUNTIME, failed=(target,)
    )

    coverage = evaluate_behavioral_coverage(
        baseline, candidate, BehavioralRuntimeKind.DETERMINISTIC_RUNTIME
    )

    assert coverage.dimension_statuses[BehavioralDimension.BELIEF_REVISION] == (
        CoverageStatus.FAILED
    )
    assert coverage.complete is True


def test_hidden_thought_sentinel_is_redacted_from_failure_artifact(
    tmp_path: Path,
) -> None:
    sentinel = "hidden-failure-sentinel"
    scenario = BehavioralScenario(
        scenario_id="runtime.hidden-thought-persistence-attack",
        dimensions=(BehavioralDimension.CONTEXT_ISOLATION,),
        initial_authoritative_state={},
        observations=(
            ExternalObservation(
                sequence=1,
                event_type="hidden_thought",
                source="runtime_fixture",
                parameters={"sentinel": sentinel},
            ),
        ),
        expected_public_behavior=PublicBehaviorClass.RESPOND,
        public_behavior_hard_gate=HardGate.HIDDEN_THOUGHT,
        forbidden_public_markers=(sentinel,),
        reproducibility=ReproducibilityMetadata(
            subject_revision="coverage-test",
            fixture_revision="coverage-test-v1",
            seed=133,
            clock=datetime(2026, 7, 23, tzinfo=UTC),
        ),
    )
    safe = BehavioralTrace(
        final_authoritative_state={},
        public_behavior=PublicBehaviorClass.RESPOND,
    )
    leaked = safe.model_copy(update={"public_payload": {"response": sentinel}})

    result = BehavioralEvaluator(tmp_path).evaluate_pair(
        "hidden-failure",
        [scenario],
        baseline_id="baseline",
        baseline_runner=lambda _scenario: safe,
        candidate_id="candidate",
        candidate_runner=lambda _scenario: leaked,
    )

    artifact = tmp_path / "behavioral" / result.reproduction_artifacts[0]
    serialized = json.dumps(json.loads(artifact.read_text(encoding="utf-8")))
    assert sentinel not in serialized
    assert "[redacted]" in serialized


def _subject(
    subject_id: str,
    runtime_kind: BehavioralRuntimeKind,
    *,
    exclude: Iterable[str] = (),
    exclude_gates: Iterable[HardGate] = (),
    failed: Iterable[str] = (),
) -> SubjectEvaluation:
    excluded = set(exclude)
    removed_gates = set(exclude_gates)
    failed_ids = set(failed)
    scenarios = deterministic_runtime_scenarios(
        subject_revision="coverage-test", runtime_kind=runtime_kind
    )
    results = tuple(
        ScenarioEvaluation(
            scenario_id=scenario.scenario_id,
            dimensions=scenario.dimensions,
            passed=scenario.scenario_id not in failed_ids,
            failures=(),
            hard_gate_failures=(),
            runtime_kind=runtime_kind,
            evaluated_hard_gates=tuple(
                gate
                for gate in (scenario.public_behavior_hard_gate,)
                if gate is not None and gate not in removed_gates
            ),
        )
        for scenario in scenarios
        if scenario.scenario_id not in excluded
    )
    return SubjectEvaluation(
        subject_id=subject_id,
        scenario_results=results,
        dimension_scores=(),
        aggregate_score=(
            sum(item.passed for item in results) / len(results) if results else 0.0
        ),
        hard_gate_failures=(),
    )
