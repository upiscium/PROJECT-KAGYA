# 05 Prompt Agent And Runtime Main Loop

## Goal

Connect prediction error, emotion, memory retrieval, prompt construction, response generation, postprocessing, and episode storage into one runtime loop.

## Target Files

- `kagya/persona/prompt_builder.py`
- `kagya/persona/conscious_agent.py`
- `kagya/runtime/__init__.py`
- `kagya/runtime/agent_runtime.py`
- `kagya/runtime/session_state.py`
- `kagya/runtime/main_loop.py`
- `tests/test_agent_runtime.py`
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

- Treat `KagyaMainLoop` as the chat/cognition orchestration compatibility facade, not as the authority that orders concurrent subject mutations.
- Implement `KagyaMainLoop.chat(user_input: str) -> ChatResult` for ordinary chat; it never returns private/debug data.
- Implement the separate ephemeral diagnostic boundary `KagyaMainLoop.chat_debug(user_input: str) -> tuple[ChatResult, DebugChatTrace]`.
- Process in this order: input, context, surprisal, emotion update, memory retrieval, prompt build, generation, postprocess, DB1 save, result return.
- Store no hidden/private model reasoning in DB1 documents or metadata.
- Keep ordinary `ChatResult` limited to visible response and explicitly public structured data such as episode ID, loss/emotion values, model ID, and adapter ID; it does not own a hidden-thought field.
- When explicitly requested and authorized, expose private diagnostics through a separate request-scoped debug boundary that cannot be persisted or returned through the ordinary result contract.

## AgentRuntime Requirements

- Treat `AgentRuntime` as the single process-local authority for acceptance ordering and execution of authoritative subject mutations.
- Admit events non-blockingly to one bounded queue and execute accepted handlers in FIFO order on exactly one consumer thread.
- Assign a strictly increasing processing sequence on that consumer. The sequence starts at one for each runtime lifetime and is not restored after restart.
- Keep event metadata immutable and bounded to event identity, event type, constant source, request/acceptance time, and processing sequence. Request bodies, prompts, hidden/private reasoning, retrieved private memory, credentials, attachments, and arbitrary payloads must remain only in ephemeral in-memory handler closures and must not enter event metadata.
- Distinguish `submit -> accepted -> ordered -> executed`. Acceptance does not mean that execution, persistence, durability, or an external effect has completed.
- Reject submission without mutation when the queue is full or the runtime is not accepting.
- Once accepted, execute an event even if its caller stops waiting or cancels its result future.
- On shutdown, stop accepting first, drain accepted events, and then stop the consumer.
- Isolate handler failures so later accepted events still execute. R03 provides no transactional rollback: a handler that mutates state and then fails may leave a partial mutation.
- `AgentRuntime` is not persistence authority and provides no queue persistence, crash durability, restart continuity, replay, Snapshot, EventJournal, or StateWAL. A process crash may lose the queue and sequence.

## Test Requirements

- `DummyProvider` drives `user_input -> response` end-to-end.
- DB1 receives a saved episode.
- Visible response does not contain `<think>`.
- Ordinary `ChatResult` and normal API schemas have no private/debug field.
- Explicit debug inspection can observe private data only for the current request, and saving the same turn leaves no private sentinel in DB1.
- Emotion state changes after loss calculation.
- Concurrent mutation producers cannot bypass the single-consumer runtime, rejected queue-full work does not execute, shutdown drains accepted work, caller cancellation does not cancel accepted work, and one handler failure does not terminate the consumer.
- Serialized event metadata contains no private sentinel or arbitrary operation payload.

## Completion Criteria

- Main loop integration test passes with no real model load.
- R03 and later AgentRuntime or persistence work must preserve this R02 boundary and must not make private reasoning durable or authoritative.
- R03 validation proves process-local ordering only; later Snapshot, Journal, and WAL layers must add durability without redefining acceptance as persistence.
