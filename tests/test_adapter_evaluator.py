import json
from pathlib import Path

import pytest

from kagya.config import Settings, load_settings
from kagya.learning import AdapterEvaluationDecision, AdapterEvaluator, AdapterRegistry, AdapterStatus
from kagya.models import DummyProvider
from kagya.structured_response import PublicBehaviorClass, structured_response_json


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class MatchingProvider(DummyProvider):
    def generate(self, prompt: str) -> str:
        return structured_response_json(
            PublicBehaviorClass.RESPOND, f"answer for {prompt}"
        )


class ResponseProvider(DummyProvider):
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return structured_response_json(
            PublicBehaviorClass.RESPOND, self.responses[prompt]
        )


class RawResponseProvider(ResponseProvider):
    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses[prompt]


class FallbackResponseProvider(ResponseProvider):
    last_fallback_used = True


def test_evaluator_promotes_candidate_to_trial_active_with_high_score(tmp_path: Path) -> None:
    settings, registry = _scored_registry(tmp_path, expected="answer")
    evaluator = AdapterEvaluator(settings, registry)

    result = evaluator.evaluate(
        "adapter-a",
        ResponseProvider({"case": "answer"}),
        baseline_provider=ResponseProvider({"case": "answer"}),
    )

    assert result.decision == AdapterEvaluationDecision.TRIAL_ACTIVE
    assert registry.lookup("adapter-a").status == AdapterStatus.TRIAL_ACTIVE
    assert Path(result.result_path).exists()


def test_evaluator_rejects_candidate_with_low_score(tmp_path: Path) -> None:
    settings, registry = _scored_registry(tmp_path, expected="answer")
    evaluator = AdapterEvaluator(settings, registry)

    result = evaluator.evaluate(
        "adapter-a",
        ResponseProvider({"case": "wrong"}),
        baseline_provider=ResponseProvider({"case": "wrong"}),
    )

    assert result.decision == AdapterEvaluationDecision.REJECTED
    assert registry.lookup("adapter-a").status == AdapterStatus.REJECTED


def test_evaluator_keeps_candidate_with_mid_score(tmp_path: Path) -> None:
    settings, registry = _scored_registry(tmp_path, expected="one two")
    evaluator = AdapterEvaluator(settings, registry)

    result = evaluator.evaluate(
        "adapter-a",
        ResponseProvider({"case": "one"}),
        baseline_provider=ResponseProvider({"case": "one"}),
    )

    assert result.decision == AdapterEvaluationDecision.CANDIDATE
    assert registry.lookup("adapter-a").status == AdapterStatus.CANDIDATE


def test_plain_expected_text_is_an_output_contract_failure_with_zero_score(
    tmp_path: Path,
) -> None:
    settings, registry = _scored_registry(tmp_path, expected="answer")

    result = AdapterEvaluator(settings, registry).evaluate(
        "adapter-a",
        RawResponseProvider({"case": "answer"}),
        baseline_provider=RawResponseProvider({"case": "answer"}),
    )

    assert result.baseline_score == 0.0
    assert result.candidate_score == 0.0
    assert result.output_contract_passed is False
    assert result.activation_gate_passed is False
    data = json.loads(Path(result.result_path).read_text(encoding="utf-8"))
    assert data["baseline_parse_status_counts"] == {"invalid_json": 1}
    assert data["candidate_parse_status_counts"] == {"invalid_json": 1}
    assert "answer" not in data


@pytest.mark.parametrize("expected", ("A useful answer", "有用な回答です。"))
def test_valid_structured_output_scores_visible_text(
    tmp_path: Path, expected: str
) -> None:
    settings, registry = _scored_registry(tmp_path, expected=expected)

    result = AdapterEvaluator(settings, registry).evaluate(
        "adapter-a",
        ResponseProvider({"case": expected}),
        baseline_provider=ResponseProvider({"case": expected}),
    )

    assert result.candidate_score == 1.0
    assert result.output_contract_passed is True


@pytest.mark.parametrize(
    ("invalid", "status"),
    (
        ("not json", "invalid_json"),
        (
            '{"behavior_class":"respond","visible_response":"answer","extra":true}',
            "invalid_schema",
        ),
        (
            '{"behavior_class":"unknown","visible_response":"answer"}',
            "invalid_schema",
        ),
        (
            '{"behavior_class":"respond","visible_response":"\\u003cthink\\u003ePRIVATE\\u003c/think\\u003e"}',
            "invalid_private_content",
        ),
    ),
)
def test_invalid_candidate_cannot_pass_against_valid_baseline(
    tmp_path: Path, invalid: str, status: str
) -> None:
    settings, registry = _scored_registry(tmp_path, expected="answer")

    result = AdapterEvaluator(settings, registry).evaluate(
        "adapter-a",
        RawResponseProvider({"case": invalid}),
        baseline_provider=ResponseProvider({"case": "answer"}),
    )

    assert result.baseline_score == 1.0
    assert result.candidate_score == 0.0
    assert result.regression is True
    assert result.output_contract_passed is False
    assert result.activation_gate_passed is False
    assert result.candidate_parse_status_counts == {status: 1}
    entry = registry.lookup("adapter-a")
    assert entry is not None
    assert entry.quality_gate_passed is False
    assert entry.holdout_gate_passed is False
    assert entry.drift_gate_passed is False


def test_evaluator_writes_history_and_score_comparison(tmp_path: Path) -> None:
    settings, registry = _scored_registry(tmp_path, expected="one two")
    evaluator = AdapterEvaluator(settings, registry)

    first = evaluator.evaluate(
        "adapter-a",
        ResponseProvider({"case": "one"}),
        baseline_provider=ResponseProvider({"case": "one"}),
    )
    second = evaluator.evaluate(
        "adapter-a",
        ResponseProvider({"case": "wrong"}),
        baseline_provider=ResponseProvider({"case": "one"}),
    )

    assert first.result_path != second.result_path
    result_files = sorted((tmp_path / "eval_results").glob("*.json"))
    assert len(result_files) == 2
    second_data = json.loads(Path(second.result_path).read_text(encoding="utf-8"))
    assert second.previous_score == pytest.approx(2 / 3)
    assert second.score_delta == pytest.approx(-2 / 3)
    assert second.regression is True
    assert second_data["previous_score"] == pytest.approx(2 / 3)
    assert second_data["regression"] is True
    assert second_data["status_before"] == "candidate"
    assert second_data["status_after"] == "candidate"


def test_evaluator_loads_eval_sets_and_writes_result_json(tmp_path: Path) -> None:
    eval_set_path = tmp_path / "eval_set.json"
    eval_set_path.write_text(
        json.dumps({"cases": [{"prompt": "alpha", "expected": "answer for alpha"}]}),
        encoding="utf-8",
    )
    settings = _settings_for_tmp_registry(tmp_path, eval_sets=[eval_set_path])
    registry = _registry_with_candidate(tmp_path, settings=settings)
    evaluator = AdapterEvaluator(settings, registry)

    result = evaluator.evaluate("adapter-a", MatchingProvider())

    result_data = json.loads(Path(result.result_path).read_text(encoding="utf-8"))
    assert result.score == 1.0
    assert result.eval_set_count == 1
    assert result.case_count == 1
    assert result_data["decision"] == "trial_active"


def test_evaluator_scores_baseline_and_candidate_on_the_same_cases(tmp_path: Path) -> None:
    eval_set_path = tmp_path / "paired_eval.json"
    eval_set_path.write_text(
        json.dumps(
            {
                "cases": [
                    {"prompt": "first secret prompt", "expected": "red blue"},
                    {"prompt": "second secret prompt", "expected": "green"},
                ]
            }
        ),
        encoding="utf-8",
    )
    settings = _settings_for_tmp_registry(tmp_path, eval_sets=[eval_set_path])
    registry = _registry_with_candidate(tmp_path, settings=settings)
    evaluator = AdapterEvaluator(settings, registry)
    baseline = ResponseProvider(
        {
            "first secret prompt": "red",
            "second secret prompt": "evergreen",
        }
    )
    candidate = ResponseProvider(
        {
            "first secret prompt": "RED, blue!",
            "second secret prompt": "green",
        }
    )

    result = evaluator.evaluate(
        "adapter-a", candidate, baseline_provider=baseline
    )

    assert baseline.prompts == candidate.prompts == [
        "first secret prompt",
        "second secret prompt",
    ]
    assert result.baseline_score == pytest.approx(1 / 3)
    assert result.candidate_score == 1.0
    assert result.score == result.candidate_score
    assert result.score_delta == pytest.approx(2 / 3)
    assert result.regression is False
    result_data = json.loads(Path(result.result_path).read_text(encoding="utf-8"))
    assert result_data["baseline_score"] == pytest.approx(1 / 3)
    assert result_data["candidate_score"] == 1.0
    assert "prompt" not in result_data
    assert "output" not in result_data
    assert "first secret prompt" not in Path(result.result_path).read_text(encoding="utf-8")


def test_paired_regression_cannot_promote_candidate(tmp_path: Path) -> None:
    eval_set_path = tmp_path / "regression_eval.json"
    eval_set_path.write_text(
        json.dumps({"cases": [{"prompt": "case", "expected": "expected"}]}),
        encoding="utf-8",
    )
    settings = _settings_for_tmp_registry(tmp_path, eval_sets=[eval_set_path])
    registry = _registry_with_candidate(tmp_path, settings=settings)
    baseline = ResponseProvider({"case": "expected"})
    candidate = ResponseProvider({"case": "unrelated"})

    result = AdapterEvaluator(settings, registry).evaluate(
        "adapter-a", candidate, baseline_provider=baseline
    )

    assert result.regression is True
    assert result.decision == AdapterEvaluationDecision.CANDIDATE
    assert registry.lookup("adapter-a").status == AdapterStatus.CANDIDATE


def test_candidate_fallback_invalidates_paired_evaluation(tmp_path: Path) -> None:
    eval_set_path = tmp_path / "fallback_eval.json"
    eval_set_path.write_text(
        json.dumps({"cases": [{"prompt": "case", "expected": "expected"}]}),
        encoding="utf-8",
    )
    settings = _settings_for_tmp_registry(tmp_path, eval_sets=[eval_set_path])
    registry = _registry_with_candidate(tmp_path, settings=settings)

    with pytest.raises(ValueError, match="Candidate provider used fallback"):
        AdapterEvaluator(settings, registry).evaluate(
            "adapter-a",
            FallbackResponseProvider({"case": "expected"}),
            baseline_provider=ResponseProvider({"case": "expected"}),
        )

    assert registry.lookup("adapter-a").status == AdapterStatus.CANDIDATE


def test_evaluator_does_not_treat_substrings_as_exact_matches(tmp_path: Path) -> None:
    eval_set_path = tmp_path / "substring_eval.json"
    eval_set_path.write_text(
        json.dumps({"cases": [{"prompt": "case", "expected": "green"}]}),
        encoding="utf-8",
    )
    settings = _settings_for_tmp_registry(tmp_path, eval_sets=[eval_set_path])
    registry = _registry_with_candidate(tmp_path, settings=settings)
    evaluator = AdapterEvaluator(settings, registry)

    result = evaluator.evaluate(
        "adapter-a",
        ResponseProvider({"case": "green"}),
        baseline_provider=ResponseProvider({"case": "evergreen"}),
    )

    assert result.baseline_score == 0.0
    assert result.candidate_score == 1.0


def test_evaluator_fails_when_configured_eval_set_is_missing(tmp_path: Path) -> None:
    settings = _settings_for_tmp_registry(tmp_path, eval_sets=[tmp_path / "missing.json"])
    registry = _registry_with_candidate(tmp_path, settings=settings)
    evaluator = AdapterEvaluator(settings, registry)

    with pytest.raises(ValueError, match="Configured eval set does not exist"):
        evaluator.evaluate("adapter-a", DummyProvider())

    assert registry.lookup("adapter-a").status == AdapterStatus.CANDIDATE


def test_evaluator_has_no_deterministic_score_override(tmp_path: Path) -> None:
    settings = _settings_for_tmp_registry(tmp_path, eval_sets=[tmp_path / "missing.json"])
    registry = _registry_with_candidate(tmp_path, settings=settings)
    evaluator = AdapterEvaluator(settings, registry)

    with pytest.raises(TypeError, match="deterministic_score"):
        evaluator.evaluate("adapter-a", DummyProvider(), deterministic_score=0.9)  # type: ignore[call-arg]


def test_evaluator_fails_when_eval_sets_have_no_cases(tmp_path: Path) -> None:
    eval_set_path = tmp_path / "empty_eval_set.json"
    eval_set_path.write_text(json.dumps({"cases": []}), encoding="utf-8")
    settings = _settings_for_tmp_registry(tmp_path, eval_sets=[eval_set_path])
    registry = _registry_with_candidate(tmp_path, settings=settings)
    evaluator = AdapterEvaluator(settings, registry)

    with pytest.raises(ValueError, match="No evaluation cases loaded"):
        evaluator.evaluate("adapter-a", DummyProvider())

    assert registry.lookup("adapter-a").status == AdapterStatus.CANDIDATE


def _registry_with_candidate(tmp_path: Path, settings: Settings | None = None) -> AdapterRegistry:
    registry = AdapterRegistry(settings or _settings_for_tmp_registry(tmp_path))
    registry.register_candidate(
        adapter_id="adapter-a",
        adapter_path=tmp_path / "adapter-a",
        dataset_path=tmp_path / "dataset.jsonl",
        dataset_hash="hash-a",
    )
    return registry


def _settings_for_tmp_registry(
    tmp_path: Path,
    *,
    eval_sets: list[Path] | None = None,
) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "path": tmp_path / "adapter_registry.json",
                    "eval_result_dir": tmp_path / "eval_results",
                    "eval_sets": eval_sets or [],
                    "trial_threshold": 0.8,
                    "reject_threshold": 0.4,
                }
            )
        }
    )


def _scored_registry(
    tmp_path: Path, *, expected: str
) -> tuple[Settings, AdapterRegistry]:
    eval_set_path = tmp_path / "eval_set.json"
    eval_set_path.write_text(
        json.dumps({"cases": [{"prompt": "case", "expected": expected}]}),
        encoding="utf-8",
    )
    settings = _settings_for_tmp_registry(tmp_path, eval_sets=[eval_set_path])
    return settings, _registry_with_candidate(tmp_path, settings=settings)
