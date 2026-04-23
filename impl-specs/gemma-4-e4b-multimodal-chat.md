# Gemma 4 E4B Multimodal Chat Design

## Purpose

Define the first-step design for adding multimodal input support to `project_kagya.chat` while keeping the structure suitable for a future HTTPS API.

## Scope

- Support text plus file attachments in the chat CLI.
- Prepare the input model so it can later be exposed through an HTTPS server without changing core request semantics.
- Support multiple attachments per turn from the start.
- Keep the initial implementation limited to `chat.py`.

## Entry Point

- Script: `project-kagya-chat`
- Module: `project_kagya.chat:main`

## Design Goals

- Treat a chat turn as a structured multimodal request, not just a plain string.
- Preserve the existing CLI UX for text-only chat.
- Make attachment handling deterministic and easy to map to an HTTP request body later.
- Avoid hard-coding CLI-only concepts into the core request model.

## Request Model

### Turn Shape

Each user turn should be represented as:

- `text`: the message text.
- `attachments`: an ordered list of attachment objects.

### Attachment Shape

Each attachment should carry:

- `path`: local filesystem path.
- `media_type`: one of `image`, `audio`, or `video`.
- `name`: optional display name derived from the file name.

### Validation Rules

- Empty text is allowed only when attachments are present.
- A turn with neither text nor attachments should be rejected.
- Attachment paths must exist before request execution.
- The same file may be attached more than once only if explicitly listed multiple times.

## CLI Behavior

### Proposed Interaction

- Keep `you> ` as the main text prompt.
- Add an attachment command such as `:attach /path/to/file`.
- Support multiple `:attach` commands before submission.
- Allow `:clear-attachments` to reset the pending attachment list.
- Allow `:list-attachments` to show the current pending attachments.

### Turn Submission

- Hitting enter on a non-command line submits the current text plus all pending attachments.
- After a successful request, pending attachments are cleared.
- History should store the normalized multimodal turn metadata, not just a rendered string.

## Future HTTPS API Shape

### Recommended Request Semantics

The eventual API should accept a single chat turn payload shaped like:

```json
{
  "text": "Describe these files",
  "attachments": [
    { "path": "/abs/path/image.png", "media_type": "image" },
    { "path": "/abs/path/audio.wav", "media_type": "audio" }
  ]
}
```

### Why This Shape

- It matches the CLI attachment flow.
- It can be serialized directly from a web form or JSON request.
- It keeps multimodal expansion localized to request parsing and preprocessing.

## Processing Pipeline

### Current Phase

`chat.py` should build a structured turn, then convert it into model inputs through a preprocessing layer.

### Required Abstraction

Introduce an intermediate representation that separates:

- user-facing turn collection
- attachment validation and loading
- model-specific prompt construction
- tokenization / processor invocation

### Model Integration

- Continue to support text-only fallback.
- Prefer a processor-based path when the selected Gemma 4 E4B runtime exposes multimodal preprocessing.
- Keep prompt rendering logic isolated from file loading logic.

## Attachment Handling

### Media Detection

- Infer `media_type` from file extension when not specified explicitly.
- Reject unsupported extensions with a clear error.

### Loading Rules

- Load attachments only when the user submits the turn.
- Do not keep binary file contents in chat history.
- Keep history references lightweight so repeated turns do not duplicate large blobs.

### Ordering

- Preserve attachment order as entered by the user.
- Pass attachments to the model in the same order.

## Error Handling

- Missing attachment files should fail fast with `FileNotFoundError`.
- Unsupported attachment types should raise `ValueError`.
- If the runtime cannot handle multimodal input, the CLI should fail with a clear `RuntimeError` rather than silently dropping attachments.

## Non-Goals

- Implementing the HTTPS server itself.
- Training-time multimodal dataset support.
- Automatic transcription, captioning, or media conversion.
- Streaming responses.

## Output

This design should let `chat.py` evolve from text-only interaction into a multimodal request front-end while keeping the eventual server API shape stable.
