#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_PARENT="$(dirname "$SCRIPT_DIR")"
GDK_INSTALL_DIR="${GDK_INSTALL_DIR:-/home/agi/app}"
VENDOR_DIR="$SCRIPT_DIR/vendor"

if test -f "$GDK_INSTALL_DIR/env.sh"; then
  # The vendor env script appends to these variables without default values,
  # while this launcher intentionally enables nounset.
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="${PYTHONPATH:-}"
  # Pass the install root explicitly. Without this argument the sourced vendor
  # script would read run.sh's first CLI argument (usually --host) as its DIR.
  source "$GDK_INSTALL_DIR/env.sh" "$GDK_INSTALL_DIR"
fi

if test -d "$VENDOR_DIR"; then
  export PYTHONPATH="$VENDOR_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

GDK_LOG_DIR="${GDK_LOG_DIR:-$PACKAGE_PARENT/logs/gdk}"
mkdir -p "$GDK_LOG_DIR"
export GLOG_log_dir="${GLOG_log_dir:-$GDK_LOG_DIR}"

cd "$PACKAGE_PARENT"
if [[ "${1:-}" == "--human-in-loop" ]]; then
  shift
  exec python3 -m robot_client.hitl.g2_runtime.service "$@"
fi
exec python3 -m robot_client "$@"
