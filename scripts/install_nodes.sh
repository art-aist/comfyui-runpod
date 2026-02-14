#!/bin/bash
# =============================================================
# Install custom nodes from catalog
# Usage: install_nodes.sh <nodes_catalog.json> <max_tier> <comfyui_path>
# Example: install_nodes.sh config/nodes_catalog.json 2 /workspace/ComfyUI
# =============================================================

CATALOG="${1:?Usage: install_nodes.sh <catalog.json> [max_tier] [comfyui_path]}"
MAX_TIER="${2:-2}"
COMFYUI_PATH="${3:-/workspace/ComfyUI}"
NODES_DIR="$COMFYUI_PATH/custom_nodes"

if [ ! -d "$COMFYUI_PATH" ]; then
    echo "ERROR: ComfyUI path not found: $COMFYUI_PATH"
    exit 1
fi

mkdir -p "$NODES_DIR"

export CATALOG MAX_TIER NODES_DIR

echo "Installing custom nodes (tier <= $MAX_TIER)..."
echo "Nodes dir: $NODES_DIR"
echo ""

python3 << 'PYEOF'
import json
import subprocess
import os
import sys

catalog_path = os.environ["CATALOG"]
max_tier = int(os.environ["MAX_TIER"])
nodes_dir = os.environ["NODES_DIR"]

with open(catalog_path) as f:
    catalog = json.load(f)

nodes = catalog.get("nodes", [])
installed = 0
skipped = 0
failed = 0

for node in nodes:
    tier = node.get("tier", 3)
    if tier > max_tier:
        continue

    name = node["name"]
    url = node["repo_url"]
    commit = node.get("commit", "")
    dest = os.path.join(nodes_dir, os.path.basename(url).replace(".git", ""))

    # Already installed?
    if os.path.exists(dest):
        print(f"  [skip] {name} — already installed")
        skipped += 1
        continue

    print(f"  [install] {name}...", end=" ", flush=True)

    # Clone
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, dest],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"FAILED: {result.stderr.strip()}")
        failed += 1
        continue

    # Checkout specific commit if specified
    if commit:
        subprocess.run(
            ["git", "-C", dest, "fetch", "--depth", "1", "origin", commit],
            capture_output=True
        )
        subprocess.run(
            ["git", "-C", dest, "checkout", commit],
            capture_output=True
        )

    # Install requirements
    req_file = os.path.join(dest, "requirements.txt")
    if os.path.exists(req_file):
        pip_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file, "-q", "--no-warn-script-location"],
            capture_output=True, text=True
        )
        if pip_result.returncode != 0:
            print(f"OK (but pip install had warnings)")
        else:
            print("OK")
    else:
        print("OK (no requirements.txt)")

    installed += 1

print(f"\nDone: {installed} installed, {skipped} skipped, {failed} failed")
if failed > 0:
    sys.exit(1)
PYEOF
