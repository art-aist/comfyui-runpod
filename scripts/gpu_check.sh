#!/bin/bash
# =============================================================
# GPU/CUDA Compatibility Check
# Runs BEFORE downloading models to avoid wasting time/money
# =============================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

echo ""
echo -e "${BOLD}=========================================="
echo "  GPU / CUDA Compatibility Check"
echo -e "==========================================${NC}"
echo ""

# --- 1. nvidia-smi ---
echo -n "[1/7] nvidia-smi... "
if ! nvidia-smi > /dev/null 2>&1; then
    echo -e "${RED}FAIL${NC}"
    echo ""
    echo -e "${RED}nvidia-smi не работает. GPU не обнаружен или драйвер не загружен.${NC}"
    echo "Эта машина не подходит. Остановите Pod и выберите другую."
    exit 1
fi
echo -e "${GREEN}OK${NC}"

# --- 2. GPU info ---
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
CUDA_DRIVER_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" || echo "unknown")
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
VRAM_GB=$(python3 -c "print(round(${VRAM_MB:-0}/1024, 1))" 2>/dev/null || echo "?")

echo ""
echo "  GPU:          $GPU_NAME"
echo "  VRAM:         ${VRAM_GB}GB"
echo "  Driver:       $DRIVER_VERSION"
echo "  CUDA (driver): $CUDA_DRIVER_VERSION"
echo ""

# --- 3. PyTorch CUDA ---
echo -n "[2/7] PyTorch CUDA... "
PYTORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "")
PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")

if [ -z "$PYTORCH_CUDA" ]; then
    echo -e "${RED}FAIL${NC} — PyTorch не видит CUDA"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}OK${NC} — PyTorch ${PYTORCH_VERSION}, CUDA toolkit ${PYTORCH_CUDA}"
fi

# --- 4. PyTorch видит GPU ---
echo -n "[3/7] PyTorch GPU access... "
GPU_TEST=$(python3 -c "
import torch
if not torch.cuda.is_available():
    print('FAIL:cuda_not_available')
else:
    try:
        x = torch.randn(100, device='cuda')
        y = x * 2
        del x, y
        torch.cuda.empty_cache()
        print('OK')
    except Exception as e:
        print(f'FAIL:{e}')
" 2>/dev/null || echo "FAIL:python_error")

if [[ "$GPU_TEST" == "OK" ]]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC} — $GPU_TEST"
    echo ""
    echo -e "${RED}PyTorch не может использовать GPU.${NC}"
    echo "Возможные причины:"
    echo "  - Несовместимость CUDA toolkit (PyTorch) и CUDA driver"
    echo "  - PyTorch CUDA: $PYTORCH_CUDA, Driver CUDA: $CUDA_DRIVER_VERSION"
    echo ""
    echo "Эта машина не подходит. Остановите Pod и выберите другую."
    exit 1
fi

# --- 5. Compute capability ---
echo -n "[4/7] Compute capability... "
COMPUTE_CAP=$(python3 -c "
import torch
cap = torch.cuda.get_device_capability(0)
print(f'{cap[0]}.{cap[1]}')
" 2>/dev/null || echo "unknown")
echo "$COMPUTE_CAP"

# Blackwell = 12.0 (sm_120), Ada = 8.9, Ampere = 8.0, Hopper = 9.0
MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)
if [ "$MAJOR" -ge 12 ] 2>/dev/null; then
    echo "  → Blackwell архитектура обнаружена"
    IS_BLACKWELL=true
else
    IS_BLACKWELL=false
fi

# --- 6. NVFP4 (только для CUDA 13.0 образа) ---
echo -n "[5/7] NVFP4 quantization... "
if [[ "$PYTORCH_CUDA" == 13.* ]]; then
    NVFP4_CHECK=$(python3 -c "
import torch
# Check for FP4 support (Blackwell feature)
try:
    if hasattr(torch, 'float4_e2m1fn_x2'):
        print('OK')
    elif hasattr(torch.ops, 'aten') and hasattr(torch.ops.aten, '_scaled_mm'):
        # Alternative check for NVFP4 via scaled_mm
        print('MAYBE')
    else:
        print('NO')
except:
    print('NO')
" 2>/dev/null || echo "NO")

    if [[ "$NVFP4_CHECK" == "OK" ]]; then
        echo -e "${GREEN}ДОСТУПЕН${NC} — 2x ускорение для Blackwell"
    elif [[ "$NVFP4_CHECK" == "MAYBE" ]]; then
        echo -e "${YELLOW}ВОЗМОЖНО${NC} — scaled_mm есть, NVFP4 может работать"
    else
        echo -e "${YELLOW}НЕ ДОСТУПЕН${NC} — будет использоваться FP8"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${YELLOW}N/A${NC} — CUDA $PYTORCH_CUDA (нужна 13.x для NVFP4)"
    if [ "$IS_BLACKWELL" = true ]; then
        echo -e "  ${YELLOW}ВНИМАНИЕ: Blackwell GPU без CUDA 13 — NVFP4 недоступен${NC}"
        WARNINGS=$((WARNINGS + 1))
    fi
fi

# --- 7. VRAM ---
echo -n "[6/7] VRAM... "
if [ "${VRAM_MB:-0}" -lt 16000 ]; then
    echo -e "${RED}${VRAM_GB}GB — недостаточно для видео-генерации${NC}"
    ERRORS=$((ERRORS + 1))
elif [ "${VRAM_MB:-0}" -lt 24000 ]; then
    echo -e "${YELLOW}${VRAM_GB}GB — может быть мало для некоторых моделей${NC}"
    WARNINGS=$((WARNINGS + 1))
elif [ "${VRAM_MB:-0}" -lt 32000 ]; then
    echo -e "${GREEN}${VRAM_GB}GB — достаточно для большинства задач${NC}"
else
    echo -e "${GREEN}${VRAM_GB}GB — отлично${NC}"
fi

# --- 8. Disk space ---
echo -n "[7/7] Disk space... "
DISK_AVAIL=$(df -k /workspace 2>/dev/null | tail -1 | awk '{print int($4/1048576)}' || echo "0")
if [ "${DISK_AVAIL:-0}" -lt 50 ]; then
    echo -e "${RED}${DISK_AVAIL}GB — мало места для моделей${NC}"
    ERRORS=$((ERRORS + 1))
elif [ "${DISK_AVAIL:-0}" -lt 100 ]; then
    echo -e "${YELLOW}${DISK_AVAIL}GB — хватит для базового набора${NC}"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}${DISK_AVAIL}GB — достаточно${NC}"
fi

# --- Summary ---
echo ""
echo -e "${BOLD}==========================================${NC}"
if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}  РЕЗУЛЬТАТ: $ERRORS ошибок, $WARNINGS предупреждений${NC}"
    echo ""
    echo -e "${RED}  Машина НЕ ПОДХОДИТ. Модели НЕ будут скачиваться.${NC}"
    echo "  Остановите Pod и выберите другую машину."
    echo -e "${BOLD}==========================================${NC}"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}  РЕЗУЛЬТАТ: OK с $WARNINGS предупреждениями${NC}"
    echo ""
    echo "  Машина подходит, но есть нюансы (см. выше)."
    echo -e "${BOLD}==========================================${NC}"
    exit 0
else
    echo -e "${GREEN}  РЕЗУЛЬТАТ: Все проверки пройдены!${NC}"
    echo ""
    echo "  GPU: $GPU_NAME (${VRAM_GB}GB)"
    echo "  CUDA: $PYTORCH_CUDA"
    echo "  Можно скачивать модели и запускать ComfyUI."
    echo -e "${BOLD}==========================================${NC}"
    exit 0
fi
