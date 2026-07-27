#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIEWER_DIR="${ROOT_DIR}/scripts/gb10/web"
VIEWER_PORT="${VIEWER_PORT:-8080}"
VIEWER_HOST="${VIEWER_HOST:-0.0.0.0}"
MAIN_STREAM_PORT="${MAIN_STREAM_PORT:-8000}"
CHEST_STREAM_PORT="${CHEST_STREAM_PORT:-8001}"
VIEW_HOST="${VIEW_HOST:-127.0.0.1}"

if [[ ! -d "${VIEWER_DIR}" ]]; then
  echo "Missing viewer assets at: ${VIEWER_DIR}"
  exit 1
fi

if [[ -z "${MAIN_STREAM_URL:-}" ]]; then
  MAIN_STREAM_URL="http://${VIEW_HOST}:${MAIN_STREAM_PORT}/stream.mjpg"
fi

if [[ -z "${CHEST_STREAM_URL:-}" ]]; then
  CHEST_STREAM_URL="http://${VIEW_HOST}:${CHEST_STREAM_PORT}/stream.mjpg"
fi

if [[ "${MAIN_STREAM_URL}" == "http://"*"/stream.mjpg" && "${CHEST_STREAM_URL}" == "http://"*"/stream.mjpg" ]]; then
  echo "Use this local viewer page:"
  printf '  http://%s:%s/unitree_dual_viewer.html?host=%s&mainPort=%s&chestPort=%s\n' \
    "${VIEWER_HOST}" "${VIEWER_PORT}" "${VIEW_HOST}" "${MAIN_STREAM_PORT}" "${CHEST_STREAM_PORT}"
else
  if command -v python3 >/dev/null 2>&1; then
    MAIN_STREAM_URL_PARAM="$(python3 -c 'import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "${MAIN_STREAM_URL}")"
    CHEST_STREAM_URL_PARAM="$(python3 -c 'import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "${CHEST_STREAM_URL}")"
  else
    MAIN_STREAM_URL_PARAM="${MAIN_STREAM_URL}"
    CHEST_STREAM_URL_PARAM="${CHEST_STREAM_URL}"
  fi
  echo "Use this local viewer page:"
  printf '  http://%s:%s/unitree_dual_viewer.html?main=%s&chest=%s\n' \
    "${VIEWER_HOST}" "${VIEWER_PORT}" "${MAIN_STREAM_URL_PARAM}" "${CHEST_STREAM_URL_PARAM}"
fi

if command -v xdg-open >/dev/null 2>&1 && [[ "${OPEN:-0}" == "1" ]]; then
  xdg-open "http://${VIEWER_HOST}:${VIEWER_PORT}/unitree_dual_viewer.html?host=${VIEW_HOST}&mainPort=${MAIN_STREAM_PORT}&chestPort=${CHEST_STREAM_PORT}" >/dev/null 2>&1 || true
fi

echo "Serving viewer from ${VIEWER_DIR} on http://${VIEWER_HOST}:${VIEWER_PORT}"
cd "${VIEWER_DIR}"
python3 -m http.server "${VIEWER_PORT}" --bind "${VIEWER_HOST}"
