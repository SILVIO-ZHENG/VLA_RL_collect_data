#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_DIR="$SCRIPT_DIR/wheels"
VENDOR_DIR="$SCRIPT_DIR/vendor"

if python3 -c "import websockets, websockets.sync.client; major=int(websockets.__version__.split('.', 1)[0]); raise SystemExit(0 if 13 <= major < 17 else 1)" >/dev/null 2>&1; then
  echo "compatible system websockets is already available"
  exit 0
fi

if ! compgen -G "$WHEEL_DIR/websockets-*.whl" >/dev/null; then
  echo "No compatible websockets wheel found under $WHEEL_DIR" >&2
  exit 1
fi

mkdir -p "$VENDOR_DIR"
python3 -m pip install \
  --no-index \
  --find-links "$WHEEL_DIR" \
  --target "$VENDOR_DIR" \
  "websockets>=13,<17"
PYTHONPATH="$VENDOR_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -c "import websockets, websockets.sync.client; print('websockets offline install OK:', websockets.__version__)"
