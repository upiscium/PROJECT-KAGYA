#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${KAGYA_APP_DIR:-/opt/project-kagya}"
RETENTION_DAYS="${KAGYA_RETENTION_DAYS:-30}"
MODE="dry-run"
CONFIRM=""

usage() {
  cat <<'USAGE'
Usage:
  scripts/private-prune.sh [--days N]
  scripts/private-prune.sh --apply --confirm PRUNE [--days N]

Environment:
  KAGYA_APP_DIR          Application checkout, default /opt/project-kagya
  KAGYA_RETENTION_DAYS   Candidate age threshold in days, default 30

The default mode is dry-run. Always run scripts/private-backup.sh first.
This helper never prunes Chroma memory or adapter_registry.json.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      MODE="apply"
      shift
      ;;
    --confirm)
      CONFIRM="${2:-}"
      shift 2
      ;;
    --days)
      RETENTION_DAYS="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "${RETENTION_DAYS}" =~ ^[0-9]+$ ]]; then
  printf 'Retention days must be a non-negative integer: %s\n' "${RETENTION_DAYS}" >&2
  exit 2
fi

if [[ "${MODE}" == "apply" && "${CONFIRM}" != "PRUNE" ]]; then
  printf 'Destructive pruning requires --apply --confirm PRUNE.\n' >&2
  exit 2
fi

KAGYA_DIR="${APP_DIR}/.kagya"
if [[ ! -d "${KAGYA_DIR}" ]]; then
  printf '.kagya directory not found: %s\n' "${KAGYA_DIR}" >&2
  exit 2
fi

print_or_delete() {
  local path="$1"
  if [[ "${MODE}" == "apply" ]]; then
    rm -rf -- "${path}"
    printf '[PRUNED] %s\n' "${path}"
  else
    printf '[CANDIDATE] %s\n' "${path}"
  fi
}

prune_files_under() {
  local dir="$1"
  local pattern="$2"
  [[ -d "${dir}" ]] || return 0
  while IFS= read -r -d '' path; do
    print_or_delete "${path}"
  done < <(find "${dir}" -type f -name "${pattern}" -mtime +"${RETENTION_DAYS}" -print0)
}

prune_archived_adapter_dirs() {
  local registry="${KAGYA_DIR}/adapter_registry.json"
  local activation_history="${KAGYA_DIR}/adapter_registry_activations.json"
  local adapters_dir="${KAGYA_DIR}/adapters"
  [[ -f "${registry}" && -d "${adapters_dir}" ]] || return 0
  python - "${registry}" "${activation_history}" "${adapters_dir}" "${RETENTION_DAYS}" <<'PY' | while IFS= read -r adapter_path; do
import json
import sys
import time
from pathlib import Path

registry = Path(sys.argv[1])
activation_history = Path(sys.argv[2])
adapters_dir = Path(sys.argv[3]).resolve()
retention_days = int(sys.argv[4])

if not activation_history.is_file():
    print("[SKIP] Adapter activation history is missing; archived adapters are protected.", file=sys.stderr)
    raise SystemExit(0)

try:
    data = json.loads(registry.read_text(encoding="utf-8"))
    history = json.loads(activation_history.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    print(f"[SKIP] Adapter lifecycle metadata is unreadable; archived adapters are protected: {exc}", file=sys.stderr)
    raise SystemExit(0)

entries = data.get("adapters", data) if isinstance(data, dict) else data
if isinstance(entries, dict):
    entries = [dict(value, adapter_id=key) for key, value in entries.items() if isinstance(value, dict)]
if not isinstance(entries, list):
    print("[SKIP] Adapter registry has an invalid shape; archived adapters are protected.", file=sys.stderr)
    raise SystemExit(0)

records = history.get("activations", []) if isinstance(history, dict) else []
if not isinstance(records, list):
    print("[SKIP] Activation history has an invalid shape; archived adapters are protected.", file=sys.stderr)
    raise SystemExit(0)

active_ids = {
    str(entry.get("adapter_id"))
    for entry in entries
    if isinstance(entry, dict)
    and entry.get("status", entry.get("state")) == "active"
    and entry.get("adapter_id")
}
rollback_ids = set()
for active_id in active_ids:
    for record in reversed(records):
        if not isinstance(record, dict) or record.get("adapter_id") != active_id:
            continue
        previous_id = record.get("previous_adapter_id")
        if previous_id:
            rollback_ids.add(str(previous_id))
        break

protected_ids = active_ids | rollback_ids
cutoff = time.time() - retention_days * 86400
for entry in entries if isinstance(entries, list) else []:
    if not isinstance(entry, dict):
        continue
    adapter_id = str(entry.get("adapter_id", ""))
    status = entry.get("status", entry.get("state"))
    raw_path = entry.get("path")
    if status != "archived" or not raw_path or adapter_id in protected_ids:
        continue
    adapter_path = Path(str(raw_path)).expanduser().resolve()
    if not adapter_path.is_relative_to(adapters_dir):
        print(f"[SKIP] Adapter path is outside the managed directory: {adapter_path}", file=sys.stderr)
        continue
    try:
        is_expired = adapter_path.stat().st_mtime < cutoff
    except OSError:
        continue
    if is_expired:
        print(adapter_path)
PY
    if [[ -d "${adapter_path}" ]]; then
      print_or_delete "${adapter_path}"
    fi
  done
}

printf 'PROJECT-KAGYA prune mode: %s, retention: %s days, app: %s\n' "${MODE}" "${RETENTION_DAYS}" "${APP_DIR}"
printf 'Run scripts/private-backup.sh before applying destructive pruning.\n'
prune_files_under "${KAGYA_DIR}/eval_results" '*.json'
prune_files_under "${KAGYA_DIR}/dreams" '*.jsonl'
prune_files_under "${KAGYA_DIR}/dreams" '*.json'
prune_archived_adapter_dirs
