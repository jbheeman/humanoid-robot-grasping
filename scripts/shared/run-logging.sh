#!/usr/bin/env bash

# Persistent detailed logging for launchers and one-shot robot commands.
# Quiet mode is the default: stdout/stderr go to ./logs while callers use
# g1_console helpers for a few readable operator-facing status lines.

g1_console() {
  printf '%s\n' "$*" >&3
}

g1_console_error() {
  printf 'ERROR: %s\n' "$*" >&4
}

g1_begin_run_log() {
  local root_dir="$1"
  local component="$2"
  local log_root timestamp log_dir log_file

  # Preserve the terminal before redirecting detailed process output. Nested
  # launchers inherit their parent's log here, which keeps child spam out of
  # the real terminal while retaining it in the combined parent log.
  exec 3>&1 4>&2

  if [[ "${G1_DISABLE_FILE_LOG:-0}" == "1" ]]; then
    G1_ACTIVE_LOG_FILE=""
    export G1_ACTIVE_LOG_FILE
    return 0
  fi

  log_root="${G1_LOG_ROOT:-${root_dir}/logs}"
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log_dir="${log_root}/${component}"
  log_file="${G1_LOG_FILE:-${log_dir}/${timestamp}-$$.log}"
  case "${log_file}" in
    *.log) ;;
    *) log_file="${log_file}.log" ;;
  esac

  mkdir -p "${log_dir}" "$(dirname "${log_file}")"
  ln -sfn "$(basename "${log_file}")" "${log_dir}/latest.log"
  G1_ACTIVE_LOG_FILE="${log_file}"
  export G1_ACTIVE_LOG_FILE

  if [[ "${G1_LOG_CONSOLE_MODE:-quiet}" == "full" ]]; then
    exec > >(tee -a "${log_file}") 2>&1
  else
    exec >>"${log_file}" 2>&1
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] log_started component=${component} pid=$$ host=$(hostname)"
  echo "Log file: ${log_file}"
  g1_console "Logging ${component}: ${log_file}"
}

g1_log_command() {
  local value
  printf '[%s] command=' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for value in "$@"; do
    printf ' %q' "${value}"
  done
  printf '\n'
}

g1_log_exit() {
  local status="$1"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] log_finished exit_code=${status}"
}
