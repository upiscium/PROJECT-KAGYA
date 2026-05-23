# 00 Governing Principles

## Scope

PROJECT-KAGYA is a subjective AI architecture with prediction error, emotion, memory, and sleep-time learning. It is not a generic chatbot.

This file defines non-negotiable constraints that apply to every implementation phase.

## Hard Constraints

- Do not implement or call Ollama.
- Do not implement OpenAI, Gemini, Claude, or any other external LLM API provider.
- Use Hugging Face Transformers as the single model execution foundation.
- Use `AutoProcessor` and `AutoModelForImageTextToText` for model loading.
- Use configured model IDs only. Do not hard-code model IDs in implementation code.
- Keep `google/gemma-4-E4B` as primary model and `google/gemma-4-E2B` as fallback in configuration.
- Use the same Transformers-based stack for generation, loss calculation, QLoRA, and adapter evaluation.
- Treat `<think>` as internal data only. It may appear in logs, learning data, and debug UI/API, but never in normal UI/API responses.
- Do not physically delete DB1 episodic records in the initial implementation. Use `archived` flags.
- Implement Dual Memory as ChromaDB DB1 `hippocampus` and DB2 `cortex` in the initial version.
- Treat four-layer memory as logical classification through `record_type`, not physical DB splitting.
- Do not activate adapters immediately after training.
- Promote adapters only through `candidate -> trial_active -> approved -> active`.
- Do not load adapters outside the adapter registry.
- Do not execute generated tool code without sandboxing and human approval.

## Architecture Defaults

- Backend: FastAPI.
- Frontend: Next.js, TypeScript, shadcn/ui, TanStack Query.
- Initial flow must pass with `DummyProvider` before relying on real model loading.
- Initial multimodal support is schema-only with text-only execution.
- Initial tool system is skeleton-only. Tool execution and tool generation are later phases.

## Global Completion Criteria

- `uv run pytest` passes.
- `DummyProvider` supports end-to-end `/api/chat`.
- `TransformersProvider` can load configured Gemma 4 model IDs.
- The runtime flow executes `user_input -> loss -> emotion -> memory retrieval -> generation -> postprocess -> DB1 save`.
- Normal chat responses never leak `<think>` or `hidden_thought`.
- Debug chat exposes `hidden_thought`, loss, emotion, prompt, and retrieved memory.
- Sleep cycle can extract high-emotion DB1 episodes and create a dream dataset.
- QLoRA trainer supports dry-run or minimal-run.
- Trained adapters are registered as `candidate`.
- Evaluator thresholds promote to `trial_active` or reject as specified.
- No adapter becomes `active` without manual approval.
- Frontend `/chat` and `/debug` can communicate with FastAPI.
