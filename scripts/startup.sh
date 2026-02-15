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

# ===== STEP 3: Custom Nodes (Tier 2) =====
echo ""
echo "[Step 3/5] Установка Tier 2 custom nodes..."
if [ -f "$CONFIG_DIR/nodes_catalog.json" ]; then
    bash "$SCRIPTS_DIR/install_nodes.sh" "$CONFIG_DIR/nodes_catalog.json" 2 "$WORK_COMFYUI"
else
    echo "  nodes_catalog.json не найден, пропускаю"
fi

# ===== STEP 4: JupyterLab + Manager UI =====
echo ""
echo "[Step 4/6] Запуск JupyterLab..."
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
echo "[Step 5/6] Запуск Manager UI..."
if [ -f "$MANAGER_DIR/app.py" ]; then
    cd "$MANAGER_DIR"
    python3 app.py \
        --port "$MANAGER_PORT" \
        --catalog "$CONFIG_DIR/models_catalog.json" \
        --comfyui-path "$WORK_COMFYUI" \
        &
    MANAGER_PID=$!
    echo "  Manager UI запущен на порту $MANAGER_PORT (PID: $MANAGER_PID)"
else
    echo "  Manager UI не найден, пропускаю"
fi

# ===== STEP 5: Sync LoRA catalog from HF + Auto-download =====
echo ""
echo "[Step 6/7] Синхронизация LoRA каталога с HuggingFace..."
HF_CONFIG_REPO="${HF_CONFIG_REPO:-kucher7serg/comfyui-config}"
if [ -n "$HF_TOKEN" ]; then
    python3 << 'EOFLORA' || echo "  LoRA sync skipped (repo may not exist yet)"
import json, os, sys
try:
    from huggingface_hub import hf_hub_download
    repo = os.environ.get("HF_CONFIG_REPO", "kucher7serg/comfyui-config")
    token = os.environ.get("HF_TOKEN")
    local_dir = "/opt/config"

    path = hf_hub_download(
        repo_id=repo, filename="loras_catalog.json",
        repo_type="dataset", local_dir=local_dir, token=token,
    )
    with open(path) as f:
        catalog = json.load(f)

    loras = catalog.get("loras", [])
    auto_loras = [l for l in loras if l.get("auto_download")]
    print(f"  LoRA catalog: {len(loras)} total, {len(auto_loras)} auto-download")

    if auto_loras:
        comfyui = os.environ.get("WORKSPACE", "/workspace") + "/ComfyUI"
        loras_dir = f"{comfyui}/models/loras"
        os.makedirs(loras_dir, exist_ok=True)

        for lora in auto_loras:
            dest = f"{loras_dir}/{lora['filename']}"
            if os.path.exists(dest):
                print(f"    [skip] {lora['name']}")
                continue
            print(f"    [download] {lora['name']} ({lora.get('size_gb', 0)}GB)...")
            try:
                hf_hub_download(
                    repo_id=lora["hf_repo"],
                    filename=lora.get("hf_file", lora["filename"]),
                    local_dir=loras_dir, token=token,
                )
                print(f"    [ok] {lora['name']}")
            except Exception as e:
                print(f"    [error] {lora['name']}: {str(e)[:150]}")

except Exception as e:
    print(f"  LoRA sync: {e}")
EOFLORA
else
    echo "  HF_TOKEN not set, skipping LoRA sync"
fi

# ===== STEP 6: Auto-download models =====
echo ""
echo "[Step 7/7] Загрузка моделей (профиль: $MODEL_PROFILE, тиры: 1,2)..."
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

# Start ComfyUI
if [ "$AUTO_START_COMFYUI" = "true" ]; then
    echo ""
    echo "============================================"
    echo "  Запуск ComfyUI"
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
