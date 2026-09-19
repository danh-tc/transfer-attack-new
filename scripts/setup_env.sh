#!/usr/bin/env bash
# Setup script cho môi trường nghiên cứu (RTX 3090, Ubuntu, thuê GPU — máy luôn fresh).
# Chạy: bash scripts/setup_env.sh
# Idempotent: xóa .venv cũ nếu có và tạo lại từ đầu, vì máy thuê coi như luôn mới.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="$REPO_ROOT/.venv"
PY_VERSION_PIN="3.10"     # mặc định của Ubuntu 22.04 (base image phổ biến nhất ở GPU rental), giữa khoảng test kỹ nhất của mmcv 2.1.0/mmdet 3.3.0.
TORCH_VERSION="2.1.2"
TORCHVISION_VERSION="0.16.2"
CUDA_TAG="cu118"          # RTX 3090 (Ampere, sm_86) chạy tốt với CUDA 11.8; wheel torch cu118 tự mang CUDA runtime, không cần cài CUDA toolkit hệ thống.
MMCV_VERSION="2.1.0"
MMDET_TAG="v3.3.0"
MMDET_DIR="$REPO_ROOT/third_party/mmdetection"

log() { echo -e "\n[setup] $*"; }

log "Kiểm tra GPU/driver..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[setup][LỖI] Không tìm thấy nvidia-smi. Máy thuê chưa có driver NVIDIA — dừng lại." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

log "Cài apt dependencies..."
SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
$SUDO apt-get update -qq
# libgl1/libglib2.0-0: opencv-python cần để import trên server không có màn hình (headless).
# ninja-build: build mmcv custom ops nhanh hơn nhiều so với không có ninja.
$SUDO apt-get install -y -qq software-properties-common git build-essential ninja-build libgl1 libglib2.0-0

PY_BIN="python${PY_VERSION_PIN}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  log "Không tìm thấy $PY_BIN trên máy — thêm deadsnakes PPA để cài đúng version pin..."
  $SUDO add-apt-repository -y ppa:deadsnakes/ppa >/dev/null
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq "python${PY_VERSION_PIN}" "python${PY_VERSION_PIN}-venv"
else
  $SUDO apt-get install -y -qq "python${PY_VERSION_PIN}-venv"
fi
log "Dùng $PY_BIN ($($PY_BIN --version))"

log "Tạo virtualenv tại $VENV_DIR (xóa bản cũ nếu có)..."
rm -rf "$VENV_DIR"
"$PY_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q

log "Cài PyTorch $TORCH_VERSION+$CUDA_TAG..."
pip install -q \
  "torch==${TORCH_VERSION}+${CUDA_TAG}" \
  "torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

log "Cài mmengine + mmcv qua openmim (tự chọn đúng wheel theo torch/cuda đã cài)..."
pip install -q openmim
mim install -q mmengine
mim install -q "mmcv==${MMCV_VERSION}"
# mmpretrain: các config ConvNeXt/Swin trong MMDetection đăng ký backbone qua registry của
# mmpretrain (mmcls) — thiếu package này thì các config đó không import được dù mmdet đã cài.
mim install -q "mmpretrain>=1.2.0"

log "Cài numpy pin cứng (mmcv/mmdet ở version này chưa tương thích numpy>=2)..."
pip install -q "numpy==1.26.4"

log "Clone + cài MMDetection ($MMDET_TAG, editable) vào third_party/..."
mkdir -p "$REPO_ROOT/third_party"
if [ ! -d "$MMDET_DIR" ]; then
  git clone --branch "$MMDET_TAG" --depth 1 https://github.com/open-mmlab/mmdetection.git "$MMDET_DIR"
fi
pip install -q -v -e "$MMDET_DIR"

log "Cài các thư viện phụ trợ (dataset eval, ảnh, tiện ích)..."
pip install -q pycocotools opencv-python-headless tqdm matplotlib

log "Kiểm tra cài đặt..."
python3 - <<'PYEOF'
import torch, mmcv, mmdet, mmengine
print(f"torch          {torch.__version__}  cuda={torch.version.cuda}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu            {torch.cuda.get_device_name(0)}")
print(f"mmengine       {mmengine.__version__}")
print(f"mmcv           {mmcv.__version__}")
print(f"mmdet          {mmdet.__version__}")
PYEOF

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "[setup][LỖI] torch.cuda.is_available() == False. Driver/CUDA tag không khớp — xem docs/environment_setup.md phần Troubleshooting." >&2
  exit 1
fi

log "Ghi lại báo cáo môi trường vào environment_report.txt..."
{
  echo "# Environment report — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  echo "---"
  pip freeze
} > "$REPO_ROOT/environment_report.txt"

log "Xong. Kích hoạt venv bằng: source .venv/bin/activate"
log "Nhớ: ghi lại thông tin phiên làm việc này (GPU thuê, mục tiêu chạy) vào docs/progress_log.md."
