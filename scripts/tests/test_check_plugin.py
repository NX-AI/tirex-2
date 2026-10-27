"""Regression checks for package boundaries, cache contents, and release gating."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_plugin import MANIFESTS, PLUGIN, validate_cache, validate_package, validate_version

REPO = Path(__file__).resolve().parents[2]


class PluginChecksTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(REPO / PLUGIN, self.root / PLUGIN)
        for name in ("LICENSE", "NOTICE", ".claude-plugin/marketplace.json", ".agents/plugins/marketplace.json"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, target)

    def test_package_and_cache(self):
        validate_package(self.root)
        cache = self.root / "cache"
        shutil.copytree(self.root / PLUGIN, cache / "0.1.0")
        validate_cache(self.root, cache)
        (cache / "0.1.0/pixi.lock").write_text("accidentally copied repository")
        with self.assertRaisesRegex(ValueError, "Installed cache differs"):
            validate_cache(self.root, cache)

    def test_root_source_rejected(self):
        path = self.root / ".claude-plugin/marketplace.json"
        data = json.loads(path.read_text())
        data["plugins"][0]["source"] = "./"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "must package only"):
            validate_package(self.root)

    def test_version_bump(self):
        def git(*args):
            return subprocess.check_output(["git", "-C", str(self.root), *args], text=True).strip()

        git("init", "-q")
        git("add", ".")
        git(
            "-c",
            "user.name=CI",
            "-c",
            "user.email=ci@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "base",
        )
        base = git("rev-parse", "HEAD")
        validate_version(self.root, base)
        skill = self.root / PLUGIN / "skills/tirex2-forecasting/SKILL.md"
        skill.write_text(skill.read_text() + "\nChanged skill.\n")
        with self.assertRaisesRegex(ValueError, "bump both"):
            validate_version(self.root, base)
        manifest = self.root / PLUGIN / MANIFESTS[0]
        data = json.loads(manifest.read_text())
        major, minor, patch = map(int, data["version"].split("."))
        data["version"] = f"{major}.{minor}.{patch + 1}"
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "disagree on version"):
            validate_package(self.root)
        other = self.root / PLUGIN / MANIFESTS[1]
        other_data = json.loads(other.read_text())
        other_data["version"] = data["version"]
        other.write_text(json.dumps(other_data))
        validate_package(self.root)
        validate_version(self.root, base)


if __name__ == "__main__":
    unittest.main()
