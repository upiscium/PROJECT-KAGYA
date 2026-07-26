"""Offline administration for encrypted backup and restoration."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

from kagya.config import load_settings
from kagya.security.backup import BackupError, BackupManager
from kagya.security.crypto import EncryptionError
from kagya.security.migration import migrate_live_state, reencrypt_live_state


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        manager = BackupManager(settings)
        result: Any
        if args.command == "create":
            result = manager.create(base_backup_id=args.base)
        elif args.command == "list":
            result = manager.list(args.limit)
        elif args.command == "verify":
            result = manager.verify(args.backup_id)
        elif args.command == "preview":
            result = manager.preview(args.backup_id)
        elif args.command == "restore":
            result = manager.restore(
                args.backup_id, expected_manifest_hash=args.manifest_hash
            )
        elif args.command == "rotate":
            result = manager.rotate(args.backup_id)
        elif args.command == "scheduled":
            statuses = manager.list(100)
            if statuses and (
                datetime.now(UTC) - statuses[0].created_at
            ).total_seconds() < settings.at_rest.backup.schedule_interval_seconds:
                result = {"status": "not_due"}
            else:
                full_every = settings.at_rest.backup.incremental_full_every
                base = None
                if statuses and len(statuses) % full_every:
                    base = statuses[0].backup_id
                result = manager.create(base_backup_id=base)
        elif args.command == "migrate-live":
            result = {"migrated_files": migrate_live_state(settings)}
        elif args.command == "rotate-live":
            result = {"reencrypted_files": reencrypt_live_state(settings)}
        else:
            parser.error("unsupported command")
            return 2
        if isinstance(result, list):
            payload: Any = [item.model_dump(mode="json") for item in result]
        elif hasattr(result, "model_dump"):
            payload = result.model_dump(mode="json")
        else:
            payload = result
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        return 0
    except (BackupError, EncryptionError, OSError, ValueError) as exc:
        print(f"encrypted state operation failed: {exc}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kagya-backup")
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--base")
    listing = commands.add_parser("list")
    listing.add_argument("--limit", type=int, default=50)
    for name in ("verify", "preview", "rotate"):
        command = commands.add_parser(name)
        command.add_argument("backup_id")
    restore = commands.add_parser("restore")
    restore.add_argument("backup_id")
    restore.add_argument("--manifest-hash", required=True)
    commands.add_parser("scheduled")
    commands.add_parser("migrate-live")
    commands.add_parser("rotate-live")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
