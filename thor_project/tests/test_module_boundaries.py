"""Keep the newly extracted control modules independent from the legacy facade."""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOT = PROJECT_ROOT / "src" / "nexus_deploy" / "control"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_control_modules_do_not_depend_on_legacy_control_facade() -> None:
    # cli.py keeps one deliberate lazy import so ``python -m
    # nexus_deploy.control_api`` remains compatible during the staged migration.
    modules = [path for path in CONTROL_ROOT.glob("*.py") if path.name != "cli.py"]
    for path in modules:
        assert "nexus_deploy.control_api" not in imported_modules(path), path.name


def test_state_models_have_no_transport_or_hardware_dependency() -> None:
    imports = imported_modules(CONTROL_ROOT / "models.py")
    assert "nexus_deploy.deployment" not in imports
    assert "nexus_deploy.control.http_api" not in imports
    assert "nexus_deploy.control.robot" not in imports
    assert "nexus_deploy.control.vision" not in imports
