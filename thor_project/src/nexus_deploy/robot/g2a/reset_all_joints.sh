#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_PARENT="$(dirname "$SCRIPT_DIR")"

if (( $# != 1 )); then
  echo "Usage: $0 <generated-safety-profile.json>" >&2
  echo "This command cycles both Dahuan grippers open-close-open, then resets 22 body joints." >&2
  exit 2
fi

SAFETY_PROFILE="$1"
if [[ ! "$SAFETY_PROFILE" =~ ^G2_dahuan_[A-Za-z0-9_.+-]+\.json$ ]]; then
  echo "Invalid generated G2 Dahuan safety profile name" >&2
  exit 2
fi
SAFETY_CONFIG="$SCRIPT_DIR/$SAFETY_PROFILE"
if [[ ! -f "$SAFETY_CONFIG" ]]; then
  echo "Calibrated reset safety profile is not deployed: $SAFETY_PROFILE" >&2
  exit 2
fi

cd "$PACKAGE_PARENT"

echo "Resetting G2A to packaged profile g2_sorting_home_20260817_v1:"
echo "  Dahuan effectors=open -> pause -> close -> pause -> open"
echo "  then arms=14, head=3, waist=5"

exec "$SCRIPT_DIR/debug_python.sh" \
  "$SCRIPT_DIR/reset_to_dataset_start.py" \
  --scope=24d \
  --effector-profile=dahuan \
  --cycle-effectors \
  --effector-reissue-s=0.75 \
  --effector-max-commands=3 \
  --effector-cycle-pause-s=0.15 \
  --startup-wait-s=1.0 \
  --safety-config="$SAFETY_CONFIG" \
  --safety-path-samples=64 \
  --execute \
  --output-dir="$PACKAGE_PARENT/logs/reset_to_dataset_start"
