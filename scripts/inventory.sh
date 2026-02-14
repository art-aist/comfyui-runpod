#!/bin/bash
# =============================================================
# Inventory Script — Phase 0
# Собирает информацию с текущего Pod (READ-ONLY!)
# Запускать на Pod: bash inventory.sh
# Результаты сохраняются в /tmp/inventory/
# =============================================================

set -e

OUT_DIR="/tmp/inventory"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "  ComfyUI Pod Inventory"
echo "  Результаты: $OUT_DIR/"
echo "=========================================="
echo ""

COMFYUI_PATH="${COMFYUI_PATH:-/workspace/ComfyUI}"

if [ ! -d "$COMFYUI_PATH" ]; then
    echo "ERROR: ComfyUI не найден в $COMFYUI_PATH"
    echo "Укажи путь: COMFYUI_PATH=/path/to/ComfyUI bash inventory.sh"
    exit 1
fi

# --- 1. System info ---
echo "[1/6] Системная информация..."
{
    echo "=== Date ==="
    date
    echo ""
    echo "=== nvidia-smi ==="
    nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
    echo ""
    echo "=== PyTorch ==="
    python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA toolkit: {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
print(f'GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'Compute capability: {cap[0]}.{cap[1]}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f}GB')
" 2>/dev/null || echo "PyTorch not available"
    echo ""
    echo "=== Python ==="
    python3 --version
    echo ""
    echo "=== ComfyUI version ==="
    cd "$COMFYUI_PATH" && git log --oneline -1 2>/dev/null || echo "Not a git repo"
    echo ""
    echo "=== OS ==="
    cat /etc/os-release 2>/dev/null | head -5
} > "$OUT_DIR/system_info.txt"
echo "  → $OUT_DIR/system_info.txt"

# --- 2. Models ---
echo "[2/6] Модели (может занять минуту)..."
{
    echo "# Models inventory — $(date)"
    echo "# Format: SIZE_BYTES  SIZE_HUMAN  PATH"
    echo ""
    find "$COMFYUI_PATH/models" -type f \
        \( -name "*.safetensors" -o -name "*.ckpt" -o -name "*.pth" -o -name "*.bin" -o -name "*.pt" -o -name "*.onnx" -o -name "*.gguf" \) \
        -exec ls -l {} \; 2>/dev/null | \
        awk '{printf "%s  %.2fGB  %s\n", $5, $5/1073741824, $NF}' | \
        sort -t'G' -k2 -rn
} > "$OUT_DIR/models_raw.txt"
MODEL_COUNT=$(wc -l < "$OUT_DIR/models_raw.txt" | tr -d ' ')
echo "  → $OUT_DIR/models_raw.txt ($MODEL_COUNT файлов)"

# --- 3. Model sizes by directory ---
echo "[3/6] Размеры директорий моделей..."
{
    echo "# Model directory sizes — $(date)"
    echo ""
    du -sh "$COMFYUI_PATH/models"/*/ 2>/dev/null | sort -rh
    echo ""
    echo "# Total:"
    du -sh "$COMFYUI_PATH/models" 2>/dev/null
} > "$OUT_DIR/models_sizes.txt"
echo "  → $OUT_DIR/models_sizes.txt"

# --- 4. Custom nodes ---
echo "[4/6] Custom nodes..."
{
    echo "# Custom nodes inventory — $(date)"
    echo "# Format: DIRECTORY | GIT_URL | COMMIT"
    echo ""
    for d in "$COMFYUI_PATH/custom_nodes"/*/; do
        dirname=$(basename "$d")
        if [ -d "$d/.git" ]; then
            url=$(cd "$d" && git remote get-url origin 2>/dev/null || echo "NO_REMOTE")
            commit=$(cd "$d" && git rev-parse --short HEAD 2>/dev/null || echo "NO_COMMIT")
            branch=$(cd "$d" && git branch --show-current 2>/dev/null || echo "detached")
            echo "$dirname | $url | $commit | $branch"
        else
            echo "$dirname | NOT_GIT | - | -"
        fi
    done
} > "$OUT_DIR/nodes_raw.txt"
NODE_COUNT=$(grep -c "|" "$OUT_DIR/nodes_raw.txt" || echo "0")
echo "  → $OUT_DIR/nodes_raw.txt ($NODE_COUNT nodes)"

# --- 5. Pip freeze ---
echo "[5/6] Python пакеты (pip freeze)..."
pip freeze > "$OUT_DIR/pip_freeze.txt" 2>/dev/null || \
    pip3 freeze > "$OUT_DIR/pip_freeze.txt" 2>/dev/null || \
    echo "pip freeze failed" > "$OUT_DIR/pip_freeze.txt"
PKG_COUNT=$(wc -l < "$OUT_DIR/pip_freeze.txt" | tr -d ' ')
echo "  → $OUT_DIR/pip_freeze.txt ($PKG_COUNT пакетов)"

# --- 6. Startup script ---
echo "[6/6] Startup скрипт..."
{
    echo "# Startup scripts found — $(date)"
    echo ""
    for f in \
        /workspace/start.sh \
        /workspace/ComfyUI/startup.sh \
        /start.sh \
        /opt/start.sh \
        "$COMFYUI_PATH/start.sh"; do
        if [ -f "$f" ]; then
            echo "=== $f ==="
            cat "$f"
            echo ""
        fi
    done
} > "$OUT_DIR/startup_scripts.txt"
echo "  → $OUT_DIR/startup_scripts.txt"

# --- Summary ---
echo ""
echo "=========================================="
echo "  Инвентаризация завершена!"
echo "=========================================="
echo ""
echo "Файлы:"
ls -lh "$OUT_DIR/"
echo ""
echo "Теперь скопируй содержимое этих файлов и отправь мне."
echo "Можно так: cat $OUT_DIR/system_info.txt"
echo "Или архивом: tar czf /tmp/inventory.tar.gz -C /tmp inventory/"
