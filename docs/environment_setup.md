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
| mmdetection | v3.3.0, clone + editable install vào `third_party/mmdetection`, cài với `--no-build-isolation` | Clone source thay vì chỉ `pip install mmdet` để chắc chắn có đủ toàn bộ `configs/` cần cho việc inspect/verify backbone (research_plan.md §13 bước 3), và để có thể patch cục bộ nếu cần sau này. `--no-build-isolation` bắt buộc vì `setup.py` của mmdetection `import torch` ở build-time (qua `torch.utils.cpp_extension`) — build isolation mặc định của pip không thấy được torch đã cài trong venv |
| setuptools | 69.5.1 (pin cứng, ghim lại ngay trước bước cài mmdetection) | `openxlab` (dependency gián tiếp của `openmim`) ghim `setuptools~=60.2.0` — quá cũ, không có PEP 660 `build_editable` hook nên editable install mmdetection lỗi. Không thể chỉ upgrade lên bản mới nhất: setuptools ≥81 đã bỏ hẳn `pkg_resources`, mà `torch.utils.cpp_extension` vẫn cần module đó. 69.5.1 là điểm vừa đủ mới (có PEP 660) vừa chưa bỏ `pkg_resources` |
| numpy | 1.26.4 (pin `<2`, ghim lại nhiều lần trong script) | mmcv/mmdet ở version trên chưa tương thích đầy đủ numpy 2.x. mmcv/mmengine và `opencv-python-headless` bản mới đều kéo theo `numpy>=2` như dependency không pin — script phải pin lại numpy sau mỗi bước có nguy cơ bị ghi đè, chốt lần cuối ngay trước verify |
| pycocotools | mới nhất | eval COCO mAP |
| opencv-python-headless | 4.10.0.84 (pin cứng) | đọc/ghi ảnh; bản `headless` vì server không có màn hình. Bản "mới nhất" (opencv 5.x) đòi `numpy>=2`, xung đột thẳng với numpy pin ở trên nên phải ghim lại bản 4.x cuối cùng còn tương thích. mmcv/mmengine cũng tự kéo theo `opencv-python` (bản GUI, không pin) như dependency bắt buộc của chúng — script gỡ nó và chỉ giữ bản headless, **theo đúng thứ tự uninstall trước rồi mới install** (ngược lại sẽ hỏng `cv2`, vì cả hai package ghi đè cùng thư mục `cv2/` trong site-packages và uninstall sẽ xóa nhầm file mà bản còn lại vừa ghi) |
| tmux | apt, bản Ubuntu mặc định | Không phải dependency Python — dùng để chạy thí nghiệm dài trong session tách khỏi SSH/VSCode remote, attach lại xem tiến độ trực tiếp được thay vì chỉ tail log file. Xem docs/progress_log.md 2026-09-20 lý do chọn (không nohup+disown trần) |

## Troubleshooting

- **`nvidia-smi` không chạy được** → image máy thuê chưa cài driver NVIDIA, hoặc container không được cấp quyền GPU. Kiểm tra lại cấu hình thuê máy trước khi chạy script.
- **`torch.cuda.is_available()` trả về `False`** sau khi cài xong → driver trên máy quá cũ so với CUDA 11.8 (cần driver ≥ 450.80.02), hoặc container không pass-through GPU đúng. Thử `nvidia-smi` xem driver version, đối chiếu [bảng tương thích CUDA/driver của NVIDIA].
- **Build `mmcv` rất lâu (5–10 phút)** → bình thường, mmcv build custom CUDA ops. Nếu quá 20 phút không xong, kiểm tra `ninja-build` đã cài chưa (script đã cài qua apt).
- **Import config ConvNeXt/Swin báo lỗi thiếu module** → thường do thiếu `mmpretrain`, script đã cài nhưng nếu tự thêm backbone/config mới, kiểm tra lại registry cần package nào.
- **`environment_report.txt`** ở root sau khi chạy xong ghi lại đầy đủ `pip freeze` + thông tin GPU — dùng file này để đối chiếu khi 1 thí nghiệm không tái lập được giữa 2 lần thuê máy khác nhau.

## Khi cần đổi version

Không sửa version trực tiếp trong đầu — nếu 1 package cần nâng cấp (vd. cần feature mới của mmdet), sửa biến ở đầu `scripts/setup_env.sh`, ghi lý do đổi vào `docs/progress_log.md`, rồi mới chạy lại.
