"""Command-line configuration validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from kagya.config.compatibility import compatibility_report
from kagya.config.settings import DEFAULT_CONFIG_PATH, load_settings_with_notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate PROJECT-KAGYA config and report compatibility notes."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config YAML",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config)

    try:
        settings, migration_notes = load_settings_with_notes(config_path)
    except (OSError, ValidationError) as exc:
        print(f"Config check failed for {config_path}: {exc}")
        return 1

    print(f"Config OK: {config_path}")
    notes = [*migration_notes, *compatibility_report(settings)]
    if notes:
        print("Compatibility notes:")
        for note in notes:
            print(f"- {note}")
    else:
        print("Compatibility notes: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
