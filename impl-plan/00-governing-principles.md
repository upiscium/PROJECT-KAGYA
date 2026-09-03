# 00 Governing Principles

## Scope

PROJECT-KAGYA is a subjective AI architecture with prediction error, emotion, memory, and sleep-time learning. It is not a generic chatbot.

This file defines non-negotiable constraints that apply to every implementation phase. Issue #245 and the R02 privacy/public-output boundary supersede earlier plan text that treated hidden/private model reasoning as persistent or trainable data.

## Hard Constraints

- Do not implement or call Ollama.
- Do not implement OpenAI, Gemini, Claude, or any other external LLM API provider.
- Use Hugging Face Transformers as the single model execution foundation.
- Use `AutoProcessor` and `AutoModelForImageTextToText` for model loading.
- Use configured model IDs only. Do not hard-code model IDs in implementation code.
- Keep `google/gemma-4-E4B` as primary model and `google/gemma-4-E2B` as fallback in configuration.
- Use the same Transformers-based stack for generation, loss calculation, QLoRA, and adapter evaluation.
- Treat hidden/private model reasoning, including `<think>` content, as ephemeral and non-authoritative. It must not enter logs, durable state, training material, or ordinary UI/API results.
- Limit private diagnostic data to an explicitly requested, request-scoped debug boundary; it must not flow back into persistence, datasets, caches, or ordinary `ChatResult` values.
- Preserve archive semantics instead of physical deletion during the ordinary DB1 episodic lifecycle. A one-way privacy migration may sanitize or recreate DB1 storage to remove private data retained before R02.
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
- Ordinary `ChatResult` and normal chat responses contain only visible response data and explicitly public structured data; they never expose `<think>`, `hidden_thought`, or another private-reasoning field.
- Debug chat may expose private diagnostics only through the explicit ephemeral debug boundary; those diagnostics are not owned by ordinary `ChatResult` and are not durable.
- Sleep cycle can extract high-emotion DB1 episodes and create a dream dataset.
- DB1 episodic documents and metadata contain no hidden/private model reasoning.
- Dream and QLoRA datasets use visible `input` and `output` only and never use private reasoning as a training target.
- QLoRA trainer supports dry-run or minimal-run.
- Trained adapters are registered as `candidate`.
- Evaluator thresholds promote to `trial_active` or reject as specified.
- No adapter becomes `active` without manual approval.
- Frontend `/chat` and `/debug` can communicate with FastAPI.

R03 and every later runtime or persistence layer must preserve this boundary. Hidden/private reasoning cannot become evidence for reconstruction, memory, identity, decisions, evaluation, or training authority.
