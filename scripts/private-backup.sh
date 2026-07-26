#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${KAGYA_APP_DIR:-${PWD}}"
CONFIG_PATH="${KAGYA_CONFIG_PATH:-${APP_DIR}/config.yaml}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/private-backup.sh [--base <backup-id>]
  scripts/private-backup.sh --verify <backup-id>
  scripts/private-backup.sh --restore <backup-id> <manifest-sha256>

This wrapper only invokes the streaming encrypted kagya-backup format. It never
creates a tar archive and never includes environment files or encryption keys.
The backup directory and key environment variable names come from config.yaml.
USAGE
}

run_backup() {
  (cd "${APP_DIR}" && exec uv run kagya-backup --config "${CONFIG_PATH}" "$@")
}

case "${1:-}" in
  --help|-h)
    usage
    ;;
  --verify)
    [[ -n "${2:-}" ]] || { usage >&2; exit 2; }
    run_backup verify "$2"
    ;;
  --restore)
    [[ -n "${2:-}" && -n "${3:-}" ]] || { usage >&2; exit 2; }
    run_backup restore "$2" --manifest-hash "$3"
    ;;
  --base)
    [[ -n "${2:-}" ]] || { usage >&2; exit 2; }
    run_backup create --base "$2"
    ;;
  "")
    run_backup create
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
