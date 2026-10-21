#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Supply NEXUS_PYTHON312 with a platform Python 3.12 path if uv cannot
# discover one. uv creates an isolated venv; it never enables system packages.
uv sync --locked --python "${NEXUS_PYTHON312:-3.12}" \
  --extra server --extra vision --extra alignment --extra dev

"$PROJECT_ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path
import sys

assert sys.version_info[:2] == (3, 12), sys.version
config = (Path(sys.prefix) / "pyvenv.cfg").read_text()
assert "include-system-site-packages = false" in config, config
import nexus_deploy.control_api
import ultralytics
from ultralytics.nn.modules.head import Segment26
print(f"Runtime ready: {sys.executable}; Ultralytics {ultralytics.__version__}; {Segment26.__name__}")
PY
