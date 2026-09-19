# Environment Setup (RTX 3090, Ubuntu, GPU thuê)

Bối cảnh: mỗi lần chạy thí nghiệm là một máy GPU thuê mới hoàn toàn (ổ đĩa trống, không có gì được giữ lại giữa các lần thuê). Mọi thứ cần để dựng lại môi trường phải nằm trong repo (git-tracked) — không dựa vào bất cứ gì đã cài thủ công lần trước.

## Cách dùng

```bash
git clone <repo-url> transfer-attack-new
cd transfer-attack-new
bash scripts/setup_env.sh
source .venv/bin/activate
```

Script `scripts/setup_env.sh` tự làm hết: kiểm tra GPU/driver, cài apt deps, tạo venv mới (xóa venv cũ nếu có), cài PyTorch/MMCV/MMDetection theo version đã pin, verify bằng cách import + kiểm tra `torch.cuda.is_available()`, và ghi báo cáo môi trường vào `environment_report.txt` (không commit file này — xem `.gitignore`).

Chạy 1 lệnh, xong là có venv sẵn sàng — không cần thao tác tay nào khác.

## Vì sao chọn các thứ này

**`venv` chuẩn thay vì conda** — máy thuê chỉ sống 1 phiên, không cần quản lý nhiều môi trường song song hay CUDA toolkit riêng của conda. Wheel PyTorch bản `cu118` đã tự mang theo CUDA runtime, chỉ cần driver NVIDIA đủ mới trên máy chủ là chạy được. `venv` dựng nhanh hơn, ít phụ thuộc hơn Miniconda.

**Version pin cụ thể** (không dùng `latest`) — để mỗi lần setup trên máy mới ra kết quả giống hệt lần trước, tránh trường hợp "hôm qua chạy được, hôm nay lỗi" do 1 dependency vừa release bản mới.

| Package | Version | Ghi chú |
|---|---|---|
| Python | **3.10** (pin cứng) | Mặc định Ubuntu 22.04 — base image phổ biến nhất ở GPU rental; giữa khoảng test kỹ nhất của mmcv 2.1.0/mmdet 3.3.0. Script tự cài qua deadsnakes PPA nếu máy không có sẵn, không phụ thuộc `python3` mặc định của hệ thống |
| torch / torchvision | 2.1.2 / 0.16.2, build `cu118` | Tương thích RTX 3090 (Ampere, sm_86), ổn định với MMDetection 3.x |
| mmengine | mới nhất qua `mim` | mim tự chọn bản khớp torch/cuda đã cài |
| mmcv | 2.1.0 | Cài qua `mim`, không cài qua pip thường (dễ sai bản build) |
| mmpretrain | ≥1.2.0 | **Bắt buộc** để dùng được config ConvNeXt/Swin trong MMDetection — các config này đăng ký backbone qua registry của mmpretrain, thiếu là import lỗi dù mmdet đã cài đủ |
| mmdetection | v3.3.0, clone + editable install vào `third_party/mmdetection` | Clone source thay vì chỉ `pip install mmdet` để chắc chắn có đủ toàn bộ `configs/` cần cho việc inspect/verify backbone (research_plan.md §13 bước 3), và để có thể patch cục bộ nếu cần sau này |
| numpy | 1.26.4 (pin `<2`) | mmcv/mmdet ở version trên chưa tương thích đầy đủ numpy 2.x |
| pycocotools, opencv-python-headless | mới nhất | eval COCO mAP, đọc/ghi ảnh; bản `headless` vì server không có màn hình |

## Troubleshooting

- **`nvidia-smi` không chạy được** → image máy thuê chưa cài driver NVIDIA, hoặc container không được cấp quyền GPU. Kiểm tra lại cấu hình thuê máy trước khi chạy script.
- **`torch.cuda.is_available()` trả về `False`** sau khi cài xong → driver trên máy quá cũ so với CUDA 11.8 (cần driver ≥ 450.80.02), hoặc container không pass-through GPU đúng. Thử `nvidia-smi` xem driver version, đối chiếu [bảng tương thích CUDA/driver của NVIDIA].
- **Build `mmcv` rất lâu (5–10 phút)** → bình thường, mmcv build custom CUDA ops. Nếu quá 20 phút không xong, kiểm tra `ninja-build` đã cài chưa (script đã cài qua apt).
- **Import config ConvNeXt/Swin báo lỗi thiếu module** → thường do thiếu `mmpretrain`, script đã cài nhưng nếu tự thêm backbone/config mới, kiểm tra lại registry cần package nào.
- **`environment_report.txt`** ở root sau khi chạy xong ghi lại đầy đủ `pip freeze` + thông tin GPU — dùng file này để đối chiếu khi 1 thí nghiệm không tái lập được giữa 2 lần thuê máy khác nhau.

## Khi cần đổi version

Không sửa version trực tiếp trong đầu — nếu 1 package cần nâng cấp (vd. cần feature mới của mmdet), sửa biến ở đầu `scripts/setup_env.sh`, ghi lý do đổi vào `docs/progress_log.md`, rồi mới chạy lại.
