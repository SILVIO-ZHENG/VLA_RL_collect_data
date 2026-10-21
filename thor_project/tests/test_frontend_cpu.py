"""Run deterministic JS lifecycle tests without connecting to a robot."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_frontend_cpu_lifecycle():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for browser lifecycle unit tests")
    result = subprocess.run([node, "--test", str(Path(__file__).with_name("frontend_cpu.test.cjs"))],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
