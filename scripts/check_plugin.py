"""Validate the forecasting plugin, its release version, and an optional installed copy."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

PLUGIN = Path("plugins/tirex2")
MANIFESTS = (".claude-plugin/plugin.json", ".codex-plugin/plugin.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def version(value):
    require(isinstance(value, str) and re.fullmatch(r"\d+\.\d+\.\d+", value), "Use a stable X.Y.Z plugin version")
    return tuple(map(int, value.split(".")))


def inventory(directory):
    files = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), f"Plugin must be self-contained; symlink found: {path}")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def validate_package(root):
    plugin = root / PLUGIN
    manifests = [json.loads((plugin / path).read_text()) for path in MANIFESTS]
    claude, codex = manifests
    for key in ("name", "version", "skills"):
        require(claude[key] == codex[key], f"Plugin manifests disagree on {key}")
    version(claude["version"])
    for filename in (".claude-plugin/marketplace.json", ".agents/plugins/marketplace.json"):
        catalog = json.loads((root / filename).read_text())
        require(catalog["name"] == "tirex2-plugins", f"Unexpected marketplace name in {filename}")
        entries = [entry for entry in catalog["plugins"] if entry["name"] == claude["name"]]
        require(len(entries) == 1, f"Expected one tirex2 entry in {filename}")
        source = entries[0]["source"]
        if isinstance(source, dict):
            require(source["source"] == "local", f"Expected a local plugin source in {filename}")
            source = source["path"]
        require(source == "./plugins/tirex2", f"{filename} must package only ./plugins/tirex2")

    return inventory(plugin)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def validate_version(root, base):
    # A merge base avoids treating unrelated updates on the target branch as PR changes.
    base = git(root, "merge-base", base, "HEAD")
    manifest = (PLUGIN / MANIFESTS[0]).as_posix()
    if not git(root, "ls-tree", "--name-only", base, "--", manifest):
        return  # First introduction of the plugin.
    if git(root, "diff", "--name-only", base, "--", str(PLUGIN)):
        previous = json.loads(git(root, "show", f"{base}:{manifest}"))
        current = json.loads((root / manifest).read_text())
        require(
            version(current["version"]) > version(previous["version"]), "Plugin changed: bump both manifest versions"
        )


def validate_cache(root, cache_root):
    expected = inventory(root / PLUGIN)
    copies = [path for path in cache_root.iterdir() if path.is_dir()]
    require(len(copies) == 1, f"Expected one installed plugin version under {cache_root}")
    require(
        inventory(copies[0]) == expected,
        "Installed cache differs from plugin package (missing, extra, or changed files)",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="PR base commit to enforce plugin version bumps")
    parser.add_argument("--cache-root", type=Path, help="Installed tirex2 directory containing version subdirectories")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = validate_package(root)
    if args.base:
        validate_version(root, args.base)
    if args.cache_root:
        validate_cache(root, args.cache_root)
    print(f"Plugin checks passed ({len(files)} packaged files)")


if __name__ == "__main__":
    main()
