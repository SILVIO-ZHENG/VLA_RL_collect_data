from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nexus_simulator.assets import AssetError, sync_assets, verify_assets


class AssetsTest(unittest.TestCase):
    def test_sync_and_verify_model_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            content = b"<robot name='fixture'/>\n"
            (source / "robot.urdf").write_bytes(content)
            asset_root = root / "assets" / "g2"
            asset_root.mkdir(parents=True)
            model_root = asset_root / "G2_omnipicker"
            model_root.mkdir()
            (model_root / "source_manifest.json").write_text(
                json.dumps(
                    {
                        "source_root": str(source),
                        "files": [
                            {
                                "name": "robot.urdf",
                                "sha256": hashlib.sha256(content).hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = sync_assets(root)
            self.assertEqual(result["file_count"], 1)
            self.assertEqual(verify_assets(root)["files"]["robot.urdf"], result["files"]["robot.urdf"])

            (model_root / "robot.urdf").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(AssetError, "模型资产不一致"):
                verify_assets(root)


if __name__ == "__main__":
    unittest.main()
