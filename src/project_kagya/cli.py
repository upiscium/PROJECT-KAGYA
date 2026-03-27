from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .main import chat_once, load_runtime

MODEL_PRESETS: dict[str, str] = {
    "lightweight": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B-Instruct",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="project-kagya")
    parser.add_argument(
        "--preset",
        choices=tuple(MODEL_PRESETS),
        default="lightweight",
        help="Model preset to load",
    )
    parser.add_argument("--model-name", help="Override the preset model name")
    parser.add_argument("--adapter-path", default="./kagya_subjective_adapter")
    parser.add_argument("--input", dest="user_input", help="User message to process")
    parser.add_argument("--valence", type=float, default=0.0)
    parser.add_argument("--arousal", type=float, default=0.0)
    return parser


def resolve_model_name(preset: str, model_name: str | None) -> str:
    if model_name:
        return model_name
    return MODEL_PRESETS[preset]


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.user_input:
        parser.error("--input is required")

    runtime = load_runtime(
        model_name=resolve_model_name(args.preset, args.model_name),
        adapter_path=args.adapter_path,
    )
    response = chat_once(runtime, args.user_input, args.valence, args.arousal)
    print(response)
    return 0


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))
