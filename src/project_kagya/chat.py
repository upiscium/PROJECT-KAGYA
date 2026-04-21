"""Interactive CLI chat for Gemma models loaded with Unsloth."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]
PromptStyle = Literal["auto", "chat", "plain"]


@dataclass(frozen=True)
class ChatTurn:
    role: Role
    content: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with a Gemma model in the CLI.")
    parser.add_argument("--model-name", default="google/gemma-4-E4B")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument(
        "--backend", choices=["auto", "unsloth", "transformers"], default="auto"
    )
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


def build_messages(
    system_prompt: str, history: Sequence[ChatTurn], user_message: str
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend({"role": turn.role, "content": turn.content} for turn in history)
    messages.append({"role": "user", "content": user_message})
    return messages


def render_prompt(
    tokenizer: Any, messages: Sequence[dict[str, str]], prompt_style: PromptStyle
) -> str:
    if prompt_style != "plain" and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    lines: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
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
    return generated_text.strip()


def _load_model_and_tokenizer(args: argparse.Namespace):
    if args.backend in {"auto", "unsloth"}:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_name,
                max_seq_length=args.max_seq_length,
                load_in_4bit=args.load_in_4bit,
            )
            model = FastLanguageModel.for_inference(model)
            return model, tokenizer, "unsloth"
        except Exception as error:
            if args.backend == "unsloth":
                raise RuntimeError(
                    "Unsloth backend failed to load. Check your CUDA/CuDNN install or use --backend transformers."
                ) from error

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    return model, tokenizer, "transformers"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        model, tokenizer, backend = _load_model_and_tokenizer(args)
    except Exception as error:
        print(
            "Failed to start chat. This environment is missing a working GPU/torch stack (for example libcudnn.so.9).",
            file=sys.stderr,
        )
        print(
            "Install a compatible CUDA/CuDNN runtime or run in a supported environment.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error

    history: list[ChatTurn] = []
    print(f"backend: {backend}")
    print("Type /exit to quit, /reset to clear history.")

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
            print("history cleared")
            continue

        messages = build_messages(args.system_prompt, history, user_message)
        prompt = render_prompt(tokenizer, messages, args.prompt_style)

        inputs = tokenizer([prompt], return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        reply = trim_generated_text(prompt, decoded)
        print(f"assistant> {reply}")
        history.append(ChatTurn(role="user", content=user_message))
        history.append(ChatTurn(role="assistant", content=reply))


if __name__ == "__main__":
    main()
