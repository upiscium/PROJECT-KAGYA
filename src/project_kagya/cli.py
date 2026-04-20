from __future__ import annotations

import argparse
import sys
from pathlib import Path

from project_kagya.runtime import KagyaRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="project-kagya")
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("settings.toml"),
        help="path to settings.toml",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="run embodied emotion demo")
    mode.add_argument("--serve", action="store_true", help="serve the FastAPI app")
    mode.add_argument(
        "--consolidate",
        action="store_true",
        help="run sleep consolidation only",
    )
    mode.add_argument("--train", action="store_true", help="run training only")
    mode.add_argument(
        "--pipeline",
        choices=["full"],
        help="run the end-to-end integration pipeline",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    runtime = KagyaRuntime(args.settings)

    if args.serve:
        runtime.run_server()
        return 0
    if args.consolidate:
        runtime.run_consolidation()
        return 0
    if args.train:
        runtime.run_training()
        return 0
    if args.pipeline == "full":
        runtime.run_full_pipeline()
        return 0
    runtime.run_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
