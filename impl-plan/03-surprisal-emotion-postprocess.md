# 03 Surprisal Emotion And Response Postprocessing

## Goal

Implement the lightweight cognition/body primitives that are independent of memory and model loading.

## Target Files

- `kagya/cognition/__init__.py`
- `kagya/cognition/surprisal_calculator.py`
- `kagya/body/__init__.py`
- `kagya/body/emotion_engine.py`
- `kagya/persona/__init__.py`
- `kagya/persona/response_postprocessor.py`
- `tests/test_surprisal_calculator.py`
- `tests/test_emotion_engine.py`
- `tests/test_response_postprocessor.py`

## Surprisal Requirements

- Implement `SurprisalCalculator` as a thin wrapper over `ModelProvider.calculate_loss`.
- Calculate loss only for the new user target text, not the full context.
- Raise `ValueError` through the provider path for empty target text.

## Emotion Requirements

- Implement `EmotionState` as a dataclass with `valence`, `arousal`, and `optimal_loss`.
- Implement `EmotionEngineAllostasis.update(loss)` using:

```text
A_new = clamp(A_current * 0.8 + loss * 0.2, 0.0, 1.0)
W = 1.0 - 0.3 * (loss - L_opt)^2
V_new = clamp(V_current * 0.4 + W * 0.6, -1.0, 1.0)
L_opt_new = (1.0 - alpha) * L_opt_current + alpha * loss
```

- Avoid NaN even for extreme loss values.

## Response Postprocessor Requirements

- Extract `<think>...</think>` into `hidden_thought`.
- Remove all complete think blocks from `visible_response`.
- Never raise on malformed think tags.
- Ensure normal visible output contains no `<think>` tag residue.

## Test Requirements

- Surprisal delegates to provider and returns fixed loss with `DummyProvider`.
- Empty target raises `ValueError`.
- Long context does not affect the intended target-only masking behavior.
- Arousal is always clamped to `[0.0, 1.0]`.
- Valence is always clamped to `[-1.0, 1.0]`.
- `optimal_loss` updates according to `adaptation_rate`.
- Extreme loss values do not produce NaN.
- Complete think blocks are extracted.
- Visible response removes think blocks.
- Malformed think tags do not crash processing.

## Completion Criteria

- These tests pass without loading any real model.
