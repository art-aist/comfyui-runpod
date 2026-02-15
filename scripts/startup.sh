#!/bin/bash
# =============================================================
# ComfyUI RunPod Universal Template — Startup Script
# Главный entrypoint для Docker контейнера
# =============================================================
set -e

echo ""
echo "============================================"
echo "  ComfyUI RunPod Universal Template"
echo "  $(date)"
echo "============================================"
echo ""

# --- Config ---
COMFYUI_PATH="${COMFYUI_PATH:-/opt/ComfyUI}"
WORKSPACE="${WORKSPACE:-/workspace}"
MANAGER_PORT="${MANAGER_PORT:-7860}"
COMFYUI_PORT="${COMFYUI_PORT:-8188}"
COMFYUI_ARGS="${COMFYUI_ARGS:-}"
MODEL_PROFILE="${MODEL_PROFILE:-default}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-false}"
AUTO_START_COMFYUI="${AUTO_START_COMFYUI:-true}"
SCRIPTS_DIR="${SCRIPTS_DIR:-/opt/scripts}"
CONFIG_DIR="${CONFIG_DIR:-/opt/config}"
MANAGER_DIR="${MANAGER_DIR:-/opt/manager}"

# ===== STEP 1: GPU Check =====
echo "[Step 1/5] Проверка GPU/CUDA..."
if [ "$SKIP_GPU_CHECK" = "true" ]; then
    echo "  Пропущено (SKIP_GPU_CHECK=true)"
else
    if ! bash "$SCRIPTS_DIR/gpu_check.sh"; then
        echo ""
        echo "================================================="
        echo "  GPU проверка не пройдена!"
        echo "  Модели НЕ скачиваются."
        echo "  Остановите Pod и выберите другую машину."
        echo ""
        echo "  Если хотите проигнорировать:"
        echo "  Установите SKIP_GPU_CHECK=true в template"
        echo "================================================="
        echo ""
        # Keep container running so user can see the error and SSH in
        tail -f /dev/null
    fi
fi

# ===== STEP 2: Workspace setup =====
echo ""
echo "[Step 2/5] Настройка workspace..."

WORK_COMFYUI="$WORKSPACE/ComfyUI"

if [ ! -d "$WORK_COMFYUI" ]; then
    echo "  ComfyUI не найден в workspace, копирую из образа..."
    cp -r "$COMFYUI_PATH" "$WORK_COMFYUI"
    echo "  Скопировано в $WORK_COMFYUI"
else
    echo "  ComfyUI уже есть в $WORK_COMFYUI"

    # Update models symlinks from Docker image if new models were added
    if [ -d "$COMFYUI_PATH/models" ]; then
        echo "  Синхронизирую Tier 1 модели из образа..."
        for f in "$COMFYUI_PATH/models"/**/*; do
            if [ -f "$f" ]; then
                rel="${f#$COMFYUI_PATH/}"
                dest="$WORK_COMFYUI/$rel"
                if [ ! -f "$dest" ]; then
                    mkdir -p "$(dirname "$dest")"
                    cp "$f" "$dest"
                    echo "    + $rel"
                fi
            fi
        done
    fi
fi

# ===== STEP 3: Sync configs from HF =====
echo ""
echo "[Step 3/8] Синхронизация настроек с HuggingFace..."
HF_CONFIG_REPO="${HF_CONFIG_REPO:-kucher7serg/comfyui-config}"
if [ -n "$HF_TOKEN" ]; then
    python3 << 'EOFSYNC' || echo "  Config sync skipped (repo may not exist yet)"
import os, shutil
try:
    from huggingface_hub import hf_hub_download
    repo = os.environ.get("HF_CONFIG_REPO", "kucher7serg/comfyui-config")
    token = os.environ.get("HF_TOKEN")
    comfyui = os.environ.get("WORKSPACE", "/workspace") + "/ComfyUI"
    config_dir = "/opt/config"

    # Sync comfy.settings.json
    try:
        path = hf_hub_download(
            repo_id=repo, filename="comfy.settings.json",
            repo_type="dataset", local_dir=config_dir, token=token,
        )
        dest = f"{comfyui}/user/default/comfy.settings.json"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(path, dest)
        print("  [ok] comfy.settings.json")
    except Exception as e:
        print(f"  [skip] comfy.settings.json: {str(e)[:100]}")

    # Sync loras_catalog.json
    try:
        hf_hub_download(
            repo_id=repo, filename="loras_catalog.json",
            repo_type="dataset", local_dir=config_dir, token=token,
        )
        print("  [ok] loras_catalog.json")
    except Exception as e:
        print(f"  [skip] loras_catalog.json: {str(e)[:100]}")

    # Sync models_catalog.json (override local with HF version if exists)
    try:
        hf_hub_download(
            repo_id=repo, filename="models_catalog.json",
            repo_type="dataset", local_dir=config_dir, token=token,
        )
        print("  [ok] models_catalog.json")
    except Exception as e:
        print(f"  [skip] models_catalog.json: {str(e)[:100]}")

    # Sync nodes_catalog.json (override local with HF version if exists)
    try:
        hf_hub_download(
            repo_id=repo, filename="nodes_catalog.json",
            repo_type="dataset", local_dir=config_dir, token=token,
        )
        print("  [ok] nodes_catalog.json")
    except Exception as e:
        print(f"  [skip] nodes_catalog.json: {str(e)[:100]}")

except Exception as e:
    print(f"  Config sync error: {e}")
EOFSYNC
else
    echo "  HF_TOKEN not set, skipping config sync"
fi

# ===== STEP 4: Custom Nodes (Tier 2) =====
echo ""
echo "[Step 4/8] Установка custom nodes..."
if [ -f "$CONFIG_DIR/nodes_catalog.json" ]; then
    bash "$SCRIPTS_DIR/install_nodes.sh" "$CONFIG_DIR/nodes_catalog.json" 99 "$WORK_COMFYUI"
else
    echo "  nodes_catalog.json не найден, пропускаю"
fi

# ===== STEP 5: JupyterLab + Manager UI =====
echo ""
echo "[Step 5/8] Запуск JupyterLab..."
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
if command -v jupyter &> /dev/null; then
    jupyter lab \
        --ip=0.0.0.0 \
        --port="$JUPYTER_PORT" \
        --no-browser \
        --allow-root \
        --ServerApp.token='' \
        --ServerApp.password='' \
        --ServerApp.allow_origin='*' \
        --ServerApp.allow_remote_access=True \
        --ServerApp.disable_check_xsrf=True \
        --notebook-dir="$WORKSPACE" \
        &
    echo "  JupyterLab запущен на порту $JUPYTER_PORT"
else
    echo "  JupyterLab не установлен, пропускаю"
fi

echo ""
echo "[Step 6/8] Запуск Manager UI..."
if [ -f "$MANAGER_DIR/app.py" ]; then
    bash "$SCRIPTS_DIR/run_manager.sh" \
        --port "$MANAGER_PORT" \
        --catalog "$CONFIG_DIR/models_catalog.json" \
        --comfyui-path "$WORK_COMFYUI" \
        &
    MANAGER_PID=$!
    echo "  Manager UI запущен на порту $MANAGER_PORT (PID: $MANAGER_PID)"
else
    echo "  Manager UI не найден, пропускаю"
fi

# ===== STEP 7: Auto-download models =====
echo ""
echo "[Step 7/8] Загрузка моделей (профиль: $MODEL_PROFILE, тиры: 1,2)..."
if [ -f "$CONFIG_DIR/models_catalog.json" ]; then
    python3 "$SCRIPTS_DIR/download_models.py" \
        --catalog "$CONFIG_DIR/models_catalog.json" \
        --profile "$MODEL_PROFILE" \
        --comfyui-path "$WORK_COMFYUI" \
        --tier 1 2 || echo "  ВНИМАНИЕ: Некоторые модели не скачались (см. ошибки выше)"
    echo "  Загрузка моделей завершена"
else
    echo "  models_catalog.json не найден, пропускаю"
fi

# ===== STEP 8: Start ComfyUI =====
if [ "$AUTO_START_COMFYUI" = "true" ]; then
    echo ""
    echo "============================================"
    echo "  [Step 8/8] Запуск ComfyUI"
    echo "  Порт: $COMFYUI_PORT"
    echo "  Args: $COMFYUI_ARGS"
    echo "============================================"
    echo ""

    cd "$WORK_COMFYUI"
    exec python3 main.py --listen 0.0.0.0 --port "$COMFYUI_PORT" $COMFYUI_ARGS
else
    echo ""
    echo "ComfyUI не запущен (AUTO_START_COMFYUI=false)"
    echo "Запустите вручную или через Manager UI"
    tail -f /dev/null
fi
