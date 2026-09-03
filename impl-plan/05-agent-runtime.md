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
- If the model is instructed to produce `<think>...</think>` followed by the final answer, treat the private segment only as ephemeral raw generation input to postprocessing.
- Make clear that internal thought is neither user-visible nor durable/training authority.

## ConsciousAgent Requirements

- Wrap a `ModelProvider` and call `generate` with configured generation parameters.
- Keep generation provider-agnostic.
- Do not expose hidden thought filtering here; leave that to `ResponsePostProcessor`.

## Main Loop Requirements

- Implement `KagyaMainLoop.chat(user_input: str) -> ChatResult` for ordinary chat; it never returns private/debug data.
- Implement the separate ephemeral diagnostic boundary `KagyaMainLoop.chat_debug(user_input: str) -> tuple[ChatResult, DebugChatTrace]`.
- Process in this order: input, context, surprisal, emotion update, memory retrieval, prompt build, generation, postprocess, DB1 save, result return.
- Store no hidden/private model reasoning in DB1 documents or metadata.
- Keep ordinary `ChatResult` limited to visible response and explicitly public structured data such as episode ID, loss/emotion values, model ID, and adapter ID; it does not own a hidden-thought field.
- When explicitly requested and authorized, expose private diagnostics through a separate request-scoped debug boundary that cannot be persisted or returned through the ordinary result contract.

## Test Requirements

- `DummyProvider` drives `user_input -> response` end-to-end.
- DB1 receives a saved episode.
- Visible response does not contain `<think>`.
- Ordinary `ChatResult` and normal API schemas have no private/debug field.
- Explicit debug inspection can observe private data only for the current request, and saving the same turn leaves no private sentinel in DB1.
- Emotion state changes after loss calculation.

## Completion Criteria

- Main loop integration test passes with no real model load.
- R03 and later AgentRuntime or persistence work must preserve this R02 boundary and must not make private reasoning durable or authoritative.
