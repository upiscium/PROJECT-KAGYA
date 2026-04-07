from __future__ import annotations

import argparse

from .main import (
    get_runtime_log_path,
    get_sleep_output_path,
    load_runtime_from_settings,
)
from .settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="project-kagya")
    parser.add_argument("--settings", default="settings.toml")
    args = parser.parse_args()
    settings = load_settings(args.settings)
    runtime = load_runtime_from_settings(settings)
    response = runtime.handle_turn(settings.runtime.input_text)
    log_path = get_runtime_log_path(settings)
    sleep_path = get_sleep_output_path(settings)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sleep_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(response + "\n", encoding="utf-8")
    print(response)


if __name__ == "__main__":
    main()
