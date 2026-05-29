# 10 Tool And Multimodal Skeletons

## Goal

Prepare schema and extension points for later tool execution and multimodal support without implementing unsafe execution or image processing in v1.0.

## Target Files

- `kagya/tools/__init__.py`
- `kagya/tools/tool_schema.py`
- `kagya/tools/tool_registry.py`
- `kagya/tools/tool_executor.py`
- `kagya/tools/tool_sandbox.py`
- `kagya/tools/tool_generator.py`
- API chat schemas that include `attachments`.

## Tool Skeleton Requirements

- Define tool schema types.
- Define registry interfaces.
- Define executor interfaces.
- Define sandbox interfaces.
- Define generator interfaces.
- Do not execute generated code.
- Do not register generated tools without human approval.
- Do not run shell commands from generated tools.

## Multimodal Skeleton Requirements

- Keep request schema with `attachments: []`.
- Treat Gemma 4 as image-text-to-text capable at model class level.
- Keep v1.0 runtime text-only.
- Do not implement file upload or image processing yet.

## Test Requirements

- Chat request accepts empty attachments.
- Non-empty attachments are either ignored safely or rejected with a clear v1.0 unsupported response.
- Tool executor skeleton does not execute anything.
- Tool registry does not auto-register generated tools.

## Completion Criteria

- Extension points exist without introducing unsafe execution paths.
