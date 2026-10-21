#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_PARENT="$(dirname "$SCRIPT_DIR")"
GDK_INSTALL_DIR="${GDK_INSTALL_DIR:-/home/agi/app}"
VENDOR_DIR="$SCRIPT_DIR/vendor"

if ! test -f "$GDK_INSTALL_DIR/env.sh"; then
  echo "GDK environment script not found: $GDK_INSTALL_DIR/env.sh" >&2
  exit 1
fi

# The vendor script appends to these variables and is not nounset-safe.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source "$GDK_INSTALL_DIR/env.sh" "$GDK_INSTALL_DIR"

if test -d "$VENDOR_DIR"; then
  export PYTHONPATH="$VENDOR_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONPATH="$PACKAGE_PARENT${PYTHONPATH:+:$PYTHONPATH}"

GDK_LOG_DIR="${GDK_LOG_DIR:-$PACKAGE_PARENT/logs/gdk}"
mkdir -p "$GDK_LOG_DIR"
export GLOG_log_dir="${GLOG_log_dir:-$GDK_LOG_DIR}"

exec /usr/bin/python3 "$@"
