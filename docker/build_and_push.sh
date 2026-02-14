#!/bin/bash
# =============================================================
# Build & Push ComfyUI Docker Image
# Запускать НА RunPod Pod (не локально!)
#
# ПОДГОТОВКА (один раз):
# 1. Создай аккаунт на https://hub.docker.com
# 2. Создай Access Token: Settings → Security → New Access Token
#
# СБОРКА:
# 1. Запусти Pod на RunPod (100GB+ disk, GPU не обязателен)
# 2. SSH в Pod
# 3. git clone <repo> && cd comfyui-runpod-project
# 4. bash docker/build_and_push.sh <dockerhub_user>
#
# ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ (опционально):
#   HF_TOKEN=hf_xxx  — для скачивания Flux VAE (gated repo) при сборке
#
# ПОСЛЕ СБОРКИ:
# → Останови Build Pod (экономь деньги!)
# =============================================================

set -e

DOCKERHUB_USER="${1:?Использование: bash docker/build_and_push.sh <dockerhub_user>}"
TAG="${DOCKERHUB_USER}/comfyui-runpod:cuda13"

echo ""
echo "============================================"
echo "  ComfyUI Docker Image Build"
echo "  Image: $TAG"
echo "============================================"
echo ""

# --- Step 0: Check Docker ---
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker не установлен!"
    echo "На RunPod Pod Docker обычно есть. Если нет:"
    echo "  apt-get update && apt-get install -y docker.io"
    exit 1
fi

# --- Step 1: Docker Hub Login ---
echo "=== Step 1: Docker Hub Login ==="
echo "Введи свой Docker Hub Access Token"
echo "(создай на hub.docker.com → Settings → Security → New Access Token)"
echo ""
docker login -u "$DOCKERHUB_USER"

# --- Step 2: Check base image ---
echo ""
echo "=== Step 2: Проверка base image ==="

# Попробуем найти подходящий RunPod image
BASE_IMAGES=(
    "runpod/pytorch:2.9.1-cu1300-torch291-ubuntu2404"
    "runpod/pytorch:2.9.0-cu1300-torch290-ubuntu2404"
    "runpod/pytorch:2.8.0-cu1300-torch280-ubuntu2404"
)

FOUND_IMAGE=""
for img in "${BASE_IMAGES[@]}"; do
    echo "  Проверяю $img..."
    if docker manifest inspect "$img" > /dev/null 2>&1; then
        FOUND_IMAGE="$img"
        echo "  НАЙДЕН: $img"
        break
    fi
done

if [ -z "$FOUND_IMAGE" ]; then
    echo ""
    echo "  RunPod images с CUDA 13.0 не найдены."
    echo "  Пробую official PyTorch..."

    FALLBACK_IMAGES=(
        "pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel"
        "pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel"
    )

    for img in "${FALLBACK_IMAGES[@]}"; do
        echo "  Проверяю $img..."
        if docker manifest inspect "$img" > /dev/null 2>&1; then
            FOUND_IMAGE="$img"
            echo "  НАЙДЕН: $img"
            break
        fi
    done
fi

if [ -z "$FOUND_IMAGE" ]; then
    echo ""
    echo "ERROR: Не найден ни один base image с CUDA 13.0!"
    echo "Доступные RunPod images можно посмотреть:"
    echo "  https://hub.docker.com/r/runpod/pytorch/tags"
    echo ""
    echo "Доступные PyTorch images:"
    echo "  https://hub.docker.com/r/pytorch/pytorch/tags"
    exit 1
fi

# Update Dockerfile with found image
echo ""
echo "  Использую: $FOUND_IMAGE"

# --- Step 3: Build ---
echo ""
echo "=== Step 3: Building Docker Image ==="
echo "Это займёт 15-30 минут..."
echo ""

cd "$(dirname "$0")/.."

# Build with --build-arg to pass the base image and HF_TOKEN
docker build \
    -f docker/Dockerfile.cuda13 \
    --build-arg BASE_IMAGE="$FOUND_IMAGE" \
    --build-arg HF_TOKEN="${HF_TOKEN:-}" \
    -t "$TAG" \
    .

echo ""
echo "Build complete!"

# --- Step 4: Quick verify (без GPU) ---
echo ""
echo "=== Step 4: Verification ==="

docker run --rm "$TAG" python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA toolkit: {torch.version.cuda}')
assert torch.version.cuda.startswith('13'), f'WRONG CUDA: {torch.version.cuda}'
import os
comfyui = '/opt/ComfyUI'
nodes = os.listdir(f'{comfyui}/custom_nodes')
print(f'Custom nodes: {len(nodes)}')
for n in sorted(nodes):
    print(f'  - {n}')
# Check Tier 1 models
for f in ['models/loras', 'models/vae', 'models/vae_approx']:
    path = f'{comfyui}/{f}'
    if os.path.exists(path):
        files = os.listdir(path)
        print(f'{f}: {files}')
print()
print('Image verification: OK!')
"

# --- Step 5: Push ---
echo ""
echo "=== Step 5: Pushing to Docker Hub ==="
echo "Это займёт 5-15 минут..."
echo ""

docker push "$TAG"

# --- Step 6: Image size ---
echo ""
IMAGE_SIZE=$(docker image inspect "$TAG" --format='{{.Size}}' 2>/dev/null || echo "0")
IMAGE_SIZE_GB=$(python3 -c "print(f'{${IMAGE_SIZE}/1024**3:.1f}')" 2>/dev/null || echo "?")

echo "============================================"
echo "  BUILD & PUSH COMPLETE!"
echo ""
echo "  Image: $TAG"
echo "  Size:  ${IMAGE_SIZE_GB}GB"
echo ""
echo "  Следующий шаг: создай RunPod Template"
echo "  https://runpod.io/console/user/templates"
echo ""
echo "  Настройки template:"
echo "    Name:           ComfyUI CUDA 13.0"
echo "    Container Image: $TAG"
echo "    Container Disk:  50GB"
echo "    Volume Disk:     200GB (для моделей)"
echo "    Expose HTTP:     7860,8188"
echo "    Expose TCP:      22"
echo ""
echo "    Environment Variables:"
echo "      HF_TOKEN=<твой_HuggingFace_токен>"
echo "      MODEL_PROFILE=wan_video"
echo "      COMFYUI_ARGS=--highvram"
echo ""
echo "  ВАЖНО: Останови этот Build Pod!"
echo "============================================"
