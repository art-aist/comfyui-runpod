#!/bin/bash
# =============================================================
# Manager UI wrapper — auto-restarts on exit (for "Restart Manager UI" button)
# Usage: bash run_manager.sh --port 7860 --catalog /opt/config/models_catalog.json --comfyui-path /workspace/ComfyUI
# =============================================================

MANAGER_DIR="${MANAGER_DIR:-/opt/manager}"

while true; do
    echo "[run_manager] Starting Manager UI..."
    cd "$MANAGER_DIR"
    python3 app.py "$@"
    EXIT_CODE=$?
    echo "[run_manager] Manager UI exited with code $EXIT_CODE, restarting in 2s..."
    sleep 2
done
