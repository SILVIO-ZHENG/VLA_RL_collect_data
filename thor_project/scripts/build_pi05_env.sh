#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -d third_party/lerobot/source ]]; then
  if [[ -n "${NEXUS_LEROBOT_MIRROR:-}" ]]; then
    .venv/bin/python scripts/prepare_pi05_lerobot.py --source "$NEXUS_LEROBOT_MIRROR"
  else
    .venv/bin/python scripts/prepare_pi05_lerobot.py
  fi
fi
.venv/bin/python scripts/prepare_pi05_lerobot.py --verify

uv venv --allow-existing --python "${NEXUS_PYTHON312:-3.12}" .pi05-venv

constraints=()
if [[ -n "${NEXUS_PI05_CONSTRAINTS:-}" ]]; then
  constraints=(--constraint "$NEXUS_PI05_CONSTRAINTS")
elif [[ "$(uname -m)" == "x86_64" ]]; then
  constraints=(--constraint "$PROJECT_ROOT/third_party/lerobot/requirements-5090.lock")
fi

uv pip install --python .pi05-venv/bin/python \
  "${constraints[@]}" \
  -e './third_party/lerobot/source[pi,peft-dep]' \
  'torch==2.11.0' 'torchvision==0.26.0' \
  'transformers==5.5.4' 'peft==0.20.0' \
  'msgpack>=1,<2' 'websockets>=15,<17'

uv pip install --python .pi05-venv/bin/python --no-deps -e .

.pi05-venv/bin/python - <<'PY'
from pathlib import Path
import sys

assert sys.version_info[:2] == (3, 12), sys.version
config = (Path(sys.prefix) / "pyvenv.cfg").read_text()
assert "include-system-site-packages = false" in config, config
from nexus_deploy.vla.pi05 import policy, server
assert len(policy.CANONICAL_ACTION_NAMES) == 24
print(f"PI0.5 imports ready: {sys.executable}")
PY
