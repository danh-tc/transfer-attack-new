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

log "Cài numpy pin cứng (mmcv/mmdet ở version này chưa tương thích numpy>=2)..."
# Phải pin TRƯỚC khi gọi mim/mmcv: pip install torch ở trên kéo theo numpy>=2 (không pin) làm
# `mim` (nội bộ import torch để detect cuda version) dính lỗi ABI "_ARRAY_API not found" —
# không fatal nhưng không đảm bảo mim chọn đúng wheel cu118, vi phạm nguyên tắc tái lập được.
pip install -q "numpy==1.26.4"

log "Cài mmengine + mmcv qua openmim (tự chọn đúng wheel theo torch/cuda đã cài)..."
pip install -q openmim
mim install -q mmengine
mim install -q "mmcv==${MMCV_VERSION}"
# mmpretrain: các config ConvNeXt/Swin trong MMDetection đăng ký backbone qua registry của
# mmpretrain (mmcls) — thiếu package này thì các config đó không import được dù mmdet đã cài.
mim install -q "mmpretrain>=1.2.0"
# mmcv/mmengine có thể kéo theo numpy>=2 qua dependency của chúng (vd. opencv-python) — pin lại
# để đảm bảo numpy<2 vẫn thắng sau khi mọi mim install đã chạy xong.
pip install -q "numpy==1.26.4"

log "Clone + cài MMDetection ($MMDET_TAG, editable) vào third_party/..."
mkdir -p "$REPO_ROOT/third_party"
if [ ! -d "$MMDET_DIR" ]; then
  git clone --branch "$MMDET_TAG" --depth 1 https://github.com/open-mmlab/mmdetection.git "$MMDET_DIR"
fi
# openxlab (dependency gián tiếp của openmim) ghim setuptools~=60.2.0, đã đè lên bản upgrade ở đầu
# script. Với --no-build-isolation, pip dùng thẳng setuptools của venv để build editable nên cần
# ghim lại về đúng 1 khoảng an toàn: >=64 để có PEP 660 build_editable hook (setuptools 60.2.0
# không có, gây lỗi "missing the 'build_editable' hook"), nhưng <81 vì setuptools 81+ đã bỏ hẳn
# pkg_resources — module mà torch.utils.cpp_extension (setup.py của mmdetection dùng) vẫn cần.
pip install -q "setuptools==69.5.1" wheel
# --no-build-isolation: setup.py của mmdetection import torch trực tiếp (torch.utils.cpp_extension)
# ở build-time; build isolation mặc định của pip cô lập khỏi site-packages nên không thấy torch
# đã cài trong venv → ModuleNotFoundError. Torch đã có sẵn trong venv nên không cần build isolation.
pip install -q -v -e "$MMDET_DIR" --no-build-isolation

log "Cài các thư viện phụ trợ (dataset eval, ảnh, tiện ích)..."
pip install -q pycocotools tqdm matplotlib

log "Ghim opencv-python-headless và gỡ xung đột numpy/opencv..."
# opencv-python-headless "mới nhất" (opencv 5.x) đã chuyển sang yêu cầu numpy>=2, xung đột với
# numpy<2 mà mmcv/mmdet 3.3.0 cần — ghim bản 4.x cuối cùng còn tương thích numpy 1.26.4.
# mmcv/mmengine ở bước mim install cũng kéo theo "opencv-python" (bản GUI, không pin) như một
# dependency bắt buộc của chúng — gỡ nó, chỉ giữ bản headless (đúng ý định ban đầu: server không
# màn hình) để 2 package không cùng cung cấp module `cv2` và ghi đè lẫn nhau không kiểm soát được.
# Thứ tự bắt buộc: uninstall GUI variant TRƯỚC, install headless SAU — cả hai cùng ghi file vào
# chung thư mục cv2/ trong site-packages, nếu install-rồi-uninstall thì bước uninstall sẽ xóa theo
# danh sách file của bản GUI và xóa nhầm luôn file mà bản headless vừa ghi đè lên (hỏng cv2, mất
# hằng số như COLOR_BGR2RGB dù import không báo lỗi).
pip uninstall -y -q opencv-python
pip install -q --force-reinstall "opencv-python-headless==4.10.0.84"
# Bản pin cuối cùng: bất kỳ dependency nào ở trên (mmcv/mmengine/opencv) cũng có thể đã âm thầm
# nâng numpy lên >=2 qua transitive requirement không pin — đây là điểm chốt cuối trước khi verify.
pip install -q --force-reinstall --no-deps "numpy==1.26.4"

log "Kiểm tra cài đặt..."
python3 - <<'PYEOF'
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("error")  # numpy ABI mismatch chỉ raise UserWarning, không tự crash — coi nó như lỗi thật ở đây
    import torch
import numpy, mmcv, mmdet, mmengine, cv2
print(f"torch          {torch.__version__}  cuda={torch.version.cuda}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu            {torch.cuda.get_device_name(0)}")
print(f"numpy          {numpy.__version__}")
print(f"mmengine       {mmengine.__version__}")
print(f"mmcv           {mmcv.__version__}")
print(f"mmdet          {mmdet.__version__}")
print(f"cv2            {cv2.__version__}")
assert numpy.__version__.startswith("1."), f"numpy phải <2, đang là {numpy.__version__} — có dependency đã ghi đè pin"

# Không chỉ import — chạy thật 1 op CUDA compiled sẵn của mmcv để bắt lỗi ABI numpy/torch câm lặng
from mmcv.ops import nms
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]], device="cuda")
scores = torch.tensor([0.9, 0.8], device="cuda")
nms(boxes, scores, 0.5)
print("mmcv CUDA op (nms) chạy thật OK")
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
