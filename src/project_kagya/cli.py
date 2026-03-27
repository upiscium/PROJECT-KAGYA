from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .main import chat_once, load_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="project-kagya")
    parser.add_argument(
        "--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    parser.add_argument("--adapter-path", default="./kagya_subjective_adapter")
    parser.add_argument("--input", dest="user_input", help="User message to process")
    parser.add_argument("--valence", type=float, default=0.0)
    parser.add_argument("--arousal", type=float, default=0.0)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.user_input:
        parser.error("--input is required")

    runtime = load_runtime(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
    )
    response = chat_once(runtime, args.user_input, args.valence, args.arousal)
    print(response)
    return 0


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))
