from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

from nexus_simulator.safety_reference import EXPECTED_SOURCE_SHA256, SAFETY_SOURCE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEXUS_PROJECT_ROOT = PROJECT_ROOT.parents[1]


class HardwareIsolationTest(unittest.TestCase):
    def test_runtime_has_no_hardware_or_remote_control_imports(self) -> None:
        forbidden = {"gdk", "paramiko", "asyncssh", "fabric"}
        violations: list[str] = []
        for path in sorted((PROJECT_ROOT / "src" / "nexus_simulator").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    names = [node.module]
                for name in names:
                    if name.split(".", 1)[0] in forbidden:
                        violations.append(f"{path.name}:{node.lineno}:{name}")
        self.assertEqual(violations, [])

    def test_safety_reference_is_repository_owned_and_pinned(self) -> None:
        expected = (
            NEXUS_PROJECT_ROOT
            / "src"
            / "nexus_deploy"
            / "robot"
            / "g2a"
            / "safety_layer.py"
        )
        self.assertEqual(SAFETY_SOURCE, expected)
        self.assertTrue(SAFETY_SOURCE.is_file())
        self.assertEqual(hashlib.sha256(SAFETY_SOURCE.read_bytes()).hexdigest(), EXPECTED_SOURCE_SHA256)


if __name__ == "__main__":
    unittest.main()
