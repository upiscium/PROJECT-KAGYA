#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8080}"
CHECK_ADMIN_PROXY="${CHECK_ADMIN_PROXY:-1}"

status_code() {
  local method="$1"
  local url="$2"
  shift 2
  curl -sS -o /dev/null -w '%{http_code}' -X "${method}" "$@" "${url}"
}

expect_status() {
  local expected="$1"
  local method="$2"
  local url="$3"
  shift 3
  local actual
  actual="$(status_code "${method}" "${url}" "$@")"
  if [[ "${actual}" != "${expected}" ]]; then
    printf 'Expected %s %s to return %s, got %s.\n' "${method}" "${url}" "${expected}" "${actual}" >&2
    exit 1
  fi
  printf '[OK] %s %s -> %s\n' "${method}" "${url}" "${actual}"
}

expect_status 200 GET "${BASE_URL}/health"
expect_status 200 POST "${BASE_URL}/api/chat" \
  -H 'Content-Type: application/json' \
  --data '{"message":"deployment smoke","attachments":[],"debug":false}'
expect_status 200 GET "${BASE_URL}/api/state/emotion"

if [[ "${CHECK_ADMIN_PROXY}" != "0" ]]; then
  expect_status 200 GET "${BASE_URL}/admin-proxy/state/emotion"
fi

printf 'Private deployment smoke checks passed for %s.\n' "${BASE_URL}"
