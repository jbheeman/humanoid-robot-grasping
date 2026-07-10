#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${SERVER_URL:-http://127.0.0.1:8000}"
COUNT="${COUNT:-1}"
INTERVAL="${INTERVAL:-0.5}"
NOTE="${NOTE:-}"

for ((i = 1; i <= COUNT; i++)); do
  curl --fail --silent --show-error \
    --get "${SERVER_URL}/capture" \
    --data-urlencode "note=${NOTE}"

  if (( i < COUNT )); then
    sleep "${INTERVAL}"
  fi
done
