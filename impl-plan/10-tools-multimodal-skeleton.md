# 10 Tool And Multimodal Skeletons

## Goal

Prepare schema and extension points for later tool execution and multimodal support without implementing unsafe code execution or image processing in v1.0.

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
- Only approved static `text_template` tools may execute in the first safe milestone.
- The next safe milestone adds approved static `metadata_lookup` tools that read one key from human-approved tool metadata.
- Do not execute generated code.
- Do not register generated tools without human approval.
- Do not run shell commands from generated tools.
- Record an audit event for every allowed or blocked execution request.

## Multimodal Skeleton Requirements

- Keep request schema with `attachments: []`.
- Treat Gemma 4 as image-text-to-text capable at model class level.
- Keep v1.0 runtime text-only.
- Do not implement file upload or image processing yet.

## Test Requirements

- Chat request accepts empty attachments.
- Non-empty attachments are either ignored safely or rejected with a clear v1.0 unsupported response.
- Tool executor runs approved static `text_template` and `metadata_lookup` tools only.
- Tool executor blocks shell tools, generated tools, unknown tools, and unapproved tools.
- Tool registry does not auto-register generated tools.

## Completion Criteria

- Extension points exist without introducing unsafe execution paths.
- The executable milestones are limited to deterministic string formatting from supplied arguments and static metadata lookup from approved tool definitions.
