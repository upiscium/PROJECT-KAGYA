#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${KAGYA_APP_DIR:-/opt/project-kagya}"
CONFIG_DIR="${KAGYA_CONFIG_DIR:-/etc/project-kagya}"
BACKUP_DIR="${KAGYA_BACKUP_DIR:-${PWD}/backups}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

usage() {
  cat <<'USAGE'
Usage:
  scripts/private-backup.sh
  scripts/private-backup.sh --restore <archive.tar.gz>

Environment:
  KAGYA_APP_DIR      Application checkout, default /opt/project-kagya
  KAGYA_CONFIG_DIR   Private env/config directory, default /etc/project-kagya
  KAGYA_BACKUP_DIR   Backup output directory, default ./backups

The backup archive includes .kagya runtime data and private deployment env files.
Treat archives as secrets and store them with restricted permissions.
USAGE
}

backup() {
  mkdir -p "${BACKUP_DIR}"
  chmod 700 "${BACKUP_DIR}"
  local archive="${BACKUP_DIR}/project-kagya-${TIMESTAMP}.tar.gz"
  tar -czf "${archive}" \
    --warning=no-file-changed \
    -C "${APP_DIR}" .kagya \
    -C "${CONFIG_DIR}" .
  chmod 600 "${archive}"
  printf '%s\n' "${archive}"
}

restore() {
  local archive="$1"
  if [[ ! -f "${archive}" ]]; then
    printf 'Backup archive not found: %s\n' "${archive}" >&2
    exit 2
  fi
  printf 'This will restore %s into %s and %s. Type RESTORE to continue: ' "${archive}" "${APP_DIR}" "${CONFIG_DIR}" >&2
  local confirmation
  read -r confirmation
  if [[ "${confirmation}" != "RESTORE" ]]; then
    printf 'Restore cancelled.\n' >&2
    exit 1
  fi
  mkdir -p "${APP_DIR}" "${CONFIG_DIR}"
  local restore_dir="/tmp/kagya-restore-${TIMESTAMP}"
  mkdir -p "${restore_dir}"
  tar -xzf "${archive}" -C "${restore_dir}"
  cp -a "${restore_dir}/.kagya" "${APP_DIR}/"
  find "${restore_dir}" -mindepth 1 -maxdepth 1 ! -name '.kagya' -exec cp -a {} "${CONFIG_DIR}/" \;
  rm -rf "${restore_dir}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--restore" ]]; then
  if [[ -z "${2:-}" ]]; then
    usage >&2
    exit 2
  fi
  restore "$2"
else
  backup
fi
