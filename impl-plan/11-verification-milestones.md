# 11 Verification Milestones

## Goal

Define the verification gates that must pass before PROJECT-KAGYA v1.0 is considered complete.

## Unit Test Gates

- `test_surprisal_calculator.py`: target-only loss, empty target error, DummyProvider loss.
- `test_emotion_engine.py`: arousal clamp, valence clamp, optimal loss update, no NaN.
- `test_response_postprocessor.py`: think extraction, visible response cleanup, malformed think handling.
- `test_dual_memory_system.py`: save episodic, retrieve DB1/DB2, archive instead of delete.
- `test_adapter_registry.py`: registration, trial promotion, approval, activation, old active archived.
- `test_adapter_evaluator.py`: score calculation and threshold behavior.
- `test_main_loop.py`: full DummyProvider runtime flow, DB1 save, hidden thought separation.

## Integration Test Gates

- `DummyProvider` completes `user_input -> response`.
- Normal API does not include `hidden_thought`.
- Debug API includes `hidden_thought`, loss, emotion, memory, prompt, and generation params.
- Sleep cycle extracts high-emotion episodes.
- Dream dataset JSONL is generated.
- QLoRA dry-run registers a candidate adapter.
- Adapter evaluator gates `trial_active` and `rejected` outcomes.
- Manual approval is required before `active`.
- Frontend `/chat` and `/debug` communicate with FastAPI.

## Command Gates

- `uv run pytest`
- `uv run python -m kagya.api.server`
- Frontend typecheck command once frontend is introduced.
- Frontend lint command once frontend is introduced.

## Regression Checks

- Search the codebase for forbidden providers and APIs before release.
- Verify no normal response schema includes `hidden_thought`.
- Verify no normal UI renders `hidden_thought`, raw prompt, or raw retrieved memory.
- Verify DB1 archive behavior does not physically delete initial episodic records.
- Verify adapter activation can occur only through approved registry transition.

## Completion Criteria

- All command gates pass or documented external constraints explain why they cannot run locally.
- No forbidden implementation path is present.
