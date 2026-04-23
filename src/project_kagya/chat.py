"""Interactive CLI chat for Gemma models loaded with Unsloth."""

from __future__ import annotations

import argparse
import shlex
import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]
PromptStyle = Literal["auto", "chat", "plain"]
MediaType = Literal["image", "audio", "video"]
ChatContent = str | list[dict[str, str]]


@dataclass(frozen=True)
class ChatTurn:
    role: Role
    content: ChatContent


@dataclass(frozen=True)
class Attachment:
    path: Path
    media_type: MediaType
    name: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with a Gemma model in the CLI.")
    parser.add_argument("--model-name", default="google/gemma-4-E4B-it")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--backend", choices=["auto", "transformers"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument(
        "--prompt-style", choices=["auto", "chat", "plain"], default="auto"
    )
    return parser


def _infer_media_type(path: Path) -> MediaType:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
        return "image"
    if suffix in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".oga"}:
        return "audio"
    if suffix in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
        return "video"
    raise ValueError(f"Unsupported attachment type: {path}")


def _build_attachment(path_str: str) -> Attachment:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Attachment not found: {path_str}")
    media_type = _infer_media_type(path)
    return Attachment(path=path.resolve(), media_type=media_type, name=path.name)


def parse_attachment_command(command: str) -> list[Attachment]:
    parts = shlex.split(command)
    if not parts or parts[0] != ":attach":
        raise ValueError("Attachment command must start with :attach.")
    if len(parts) == 1:
        raise ValueError("Provide at least one attachment path.")
    return [_build_attachment(path_str) for path_str in parts[1:]]


def build_attachment_content(attachments: Sequence[Attachment]) -> list[dict[str, str]]:
    content: list[dict[str, str]] = []
    for attachment in attachments:
        if attachment.media_type == "image":
            content.append({"type": "image", "url": str(attachment.path)})
        elif attachment.media_type == "audio":
            content.append({"type": "audio", "audio": str(attachment.path)})
        else:
            content.append({"type": "video", "video": str(attachment.path)})
    return content


def build_user_content(text: str, attachments: Sequence[Attachment]) -> ChatContent:
    if not attachments:
        return text
    content = build_attachment_content(attachments)
    if text:
        content.append({"type": "text", "text": text})
    if not content:
        raise ValueError("A turn must include text or attachments.")
    return content


def build_messages(
    system_prompt: str,
    history: Sequence[ChatTurn],
    user_message: str,
    attachments: Sequence[Attachment] = (),
) -> list[dict[str, Any]]:
    multimodal = bool(attachments)
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
                if multimodal
                else system_prompt,
            }
        )
    for turn in history:
        content = turn.content
        if multimodal and isinstance(content, str):
            content = [{"type": "text", "text": content}]
        messages.append({"role": turn.role, "content": content})
    messages.append(
        {
            "role": "user",
            "content": build_user_content(user_message, attachments),
        }
    )
    return messages


def _content_to_text(content: ChatContent) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        text = item.get("text")
        if text:
            parts.append(text)
    return " ".join(parts)


def render_prompt(
    tokenizer: Any, messages: Sequence[dict[str, Any]], prompt_style: PromptStyle
) -> str:
    turn_start = getattr(tokenizer, "sot_token", None)
    turn_end = getattr(tokenizer, "eot_token", None)
    if prompt_style != "plain" and turn_start and turn_end:
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = _content_to_text(message.get("content", ""))
            parts.append(f"{turn_start}{role}\n{content}{turn_end}\n")
        parts.append(f"{turn_start}assistant\n")
        return "".join(parts)

    if prompt_style != "plain" and hasattr(tokenizer, "apply_chat_template"):
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except ValueError:
                pass

    lines: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = _content_to_text(message.get("content", ""))
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def trim_generated_text(prompt: str, generated_text: str) -> str:
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :].lstrip()
    reply = generated_text.strip()
    for prefix in ("Assistant:", "assistant:"):
        if reply.startswith(prefix):
            reply = reply[len(prefix) :].lstrip()
    for marker in ("<|turn>", "<turn|>", "<|channel>", "<channel|>"):
        if marker in reply:
            reply = reply.split(marker, 1)[-1].lstrip()
    if reply.startswith("<unused"):
        reply = ""
    return reply


def _prepare_inputs(
    processor: Any, messages: Sequence[dict[str, Any]], prompt_style: PromptStyle
) -> tuple[Any, int]:
    if prompt_style != "plain" and hasattr(processor, "apply_chat_template"):
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            input_ids = inputs["input_ids"]
            return inputs, input_ids.shape[-1]
        except ValueError:
            pass

    prompt = render_prompt(processor, messages, prompt_style)
    inputs = processor(text=prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    return inputs, input_ids.shape[-1]


def _has_cuda_device() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def _load_model_and_processor(args: argparse.Namespace):
    if not _has_cuda_device():
        raise RuntimeError("Gemma 4 multimodal chat requires a CUDA-capable GPU.")

    if not args.model_name.endswith("-it"):
        raise RuntimeError(
            "Use an instruction-tuned Gemma 4 model for chat, for example google/gemma-4-E4B-it."
        )

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto",
        load_in_4bit=args.load_in_4bit,
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    return model, processor, "transformers"


def _decode_outputs(processor: Any, outputs: Any, input_length: int) -> str:
    generated = outputs[:, input_length:]
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(generated, skip_special_tokens=False)[0].strip()
    return processor.decode(generated[0], skip_special_tokens=False).strip()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        model, processor, backend = _load_model_and_processor(args)
    except Exception as error:
        print("Failed to start chat.", file=sys.stderr)
        traceback.print_exception(error, file=sys.stderr)
        raise SystemExit(1) from error

    history: list[ChatTurn] = []
    pending_attachments: list[Attachment] = []
    print(f"backend: {backend}")
    print("Type /exit to quit, /reset to clear history.")
    print("Use :attach PATH to add files, :clear-attachments to reset them.")

    while True:
        try:
            user_message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_message:
            continue
        if user_message in {"/exit", "/quit"}:
            break
        if user_message == "/reset":
            history.clear()
            pending_attachments.clear()
            print("history cleared")
            continue
        if user_message == ":clear-attachments":
            pending_attachments.clear()
            print("attachments cleared")
            continue
        if user_message == ":list-attachments":
            if not pending_attachments:
                print("no attachments")
            else:
                for attachment in pending_attachments:
                    print(f"- {attachment.media_type}: {attachment.path}")
            continue
        if user_message.startswith(":attach"):
            try:
                attachments = parse_attachment_command(user_message)
            except Exception as error:
                print(f"attachment error: {error}", file=sys.stderr)
                continue
            pending_attachments.extend(attachments)
            for attachment in attachments:
                print(f"attached {attachment.media_type}: {attachment.path}")
            continue

        if not user_message and not pending_attachments:
            continue

        messages = build_messages(
            args.system_prompt,
            history,
            user_message,
            pending_attachments,
        )
        inputs, input_length = _prepare_inputs(processor, messages, args.prompt_style)
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
        )

        reply = _decode_outputs(processor, outputs, input_length)
        print(f"assistant> {reply}")
        history.append(
            ChatTurn(
                role="user",
                content=build_user_content(user_message, pending_attachments),
            )
        )
        history.append(ChatTurn(role="assistant", content=reply))
        pending_attachments.clear()


if __name__ == "__main__":
    main()
