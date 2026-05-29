# 05 Prompt Agent And Runtime Main Loop

## Goal

Connect prediction error, emotion, memory retrieval, prompt construction, response generation, postprocessing, and episode storage into one runtime loop.

## Target Files

- `kagya/persona/prompt_builder.py`
- `kagya/persona/conscious_agent.py`
- `kagya/runtime/__init__.py`
- `kagya/runtime/session_state.py`
- `kagya/runtime/main_loop.py`
- `tests/test_main_loop.py`

## PromptBuilder Requirements

- Build prompts from user input, current emotion state, and retrieved memory context.
- Include valence, arousal, and optimal loss.
- Include related DB1 episodes.
- Include related DB2 semantic memories.
- Instruct the model to produce `<think>...</think>` followed by the final answer.
- Make clear that internal thought is not normally visible to the user.

## ConsciousAgent Requirements

- Wrap a `ModelProvider` and call `generate` with configured generation parameters.
- Keep generation provider-agnostic.
- Do not expose hidden thought filtering here; leave that to `ResponsePostProcessor`.

## Main Loop Requirements

- Implement `KagyaMainLoop.chat(user_input: str, debug: bool = False) -> ChatResult`.
- Process in this order: input, context, surprisal, emotion update, memory retrieval, prompt build, generation, postprocess, DB1 save, result return.
- Store `hidden_thought` in DB1.
- Return `hidden_thought` only when debug behavior explicitly allows it at API layer.
- Include `episode_id`, response, hidden thought, loss, valence, arousal, optimal loss, model ID, and adapter ID in `ChatResult`.

## Test Requirements

- `DummyProvider` drives `user_input -> response` end-to-end.
- DB1 receives a saved episode.
- Visible response does not contain `<think>`.
- Hidden thought is available in `ChatResult` for debug API use.
- Normal API schema later excludes hidden thought.
- Emotion state changes after loss calculation.

## Completion Criteria

- Main loop integration test passes with no real model load.
