# Issue 12 Implementation Plan

## Goal

Add multimodal input support to the HTTPS API so it can accept the same `text + attachments[]` request shape as the CLI.

## Scope

- Support text plus file attachments in the HTTP server layer.
- Parse `text + attachments[]` requests with multiple attachments in order.
- Reuse the same input semantics as the CLI.
- Keep the request model independent from client-specific behavior.

## Planned Work

1. Define the shared multimodal request shape.
   - Represent input as `text` plus an ordered `attachments` list.
   - Keep the payload lightweight so CLI and HTTP clients can serialize it the same way.

2. Add HTTP request parsing and validation.
   - Accept synchronous multimodal requests in the API layer.
   - Validate that text or attachments are present.
   - Validate attachment paths and media types before execution.

3. Connect the request model to chat preprocessing.
   - Keep request parsing, attachment loading, and model-specific preparation separate.
   - Preserve text-only fallback behavior for non-multimodal requests.

4. Wire multimodal input into API execution.
   - Prefer a processor-based path when the runtime supports multimodal preprocessing.
   - Surface a clear error if attachments are provided but the runtime cannot handle them.

5. Update tests and docs.
   - Add regression tests for request parsing, attachment ordering, and fallback behavior.
   - Update API docs or examples if the request shape changes.

## Acceptance Criteria

- The HTTPS API accepts `text + attachments[]` requests.
- A request can include multiple attachments in order.
- Missing or unsupported attachments raise clear errors.
- The request shape stays compatible with the CLI semantics.

## Non-Goals

- Streaming responses.
- Training data changes for multimodal inputs.
