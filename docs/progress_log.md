# Nhật ký tiến độ (Progress Log)

Chỉ thêm mới (append-only). Sau mỗi phiên làm việc có kết quả hoặc quyết định, thêm một entry mới có ngày ở cuối file. Không sửa lại entry cũ, trừ khi sửa lỗi sai sự kiện — nếu phân vân, gạch ngang phần cũ thay vì xóa.

---

## 2026-09-19 — Khởi tạo dự án

- Đã viết kế hoạch nghiên cứu (v1): tấn công né tránh có khả năng chuyển giao xuyên họ feature-extractor, dùng single-surrogate, cho object detection. Xem [research_plan.md](research_plan.md).
- Tổ chức lại docs: `CLAUDE.md` (root, pointer cho mỗi session) + `docs/research_plan.md` + `docs/progress_log.md` (file này) + `docs/model_registry.md`.
- Toàn bộ docs chuyển sang tiếng Việt.
- Chưa có code, chưa có môi trường (venv), chưa verify model nào.
- Bước tiếp theo: Thí nghiệm 1, danh sách việc ở research_plan.md §13 — bắt đầu từ kiểm tra repo/model registry (bước 1-3).

## 2026-09-19 — Quyết định môi trường (venv)

- Bối cảnh: chạy trên GPU thuê (RTX 3090, Ubuntu), mỗi lần thuê là máy hoàn toàn mới, không giữ lại gì giữa các phiên.
- Quyết định: dùng `venv` chuẩn (không conda) + version pin cứng, dựng qua 1 script duy nhất `scripts/setup_env.sh`. Lý do và version cụ thể: [environment_setup.md](environment_setup.md).
- Stack đã chốt: torch 2.1.2 + cu118, mmcv 2.1.0 qua `mim`, mmpretrain ≥1.2.0 (bắt buộc cho config ConvNeXt/Swin), mmdetection v3.3.0 clone editable vào `third_party/`, numpy pin `<2`.
- Thêm `.gitignore` cho `.venv/`, `third_party/`, checkpoint, dataset, output — các thứ này không commit, dựng lại mỗi lần qua script.
- Thêm cơ chế bootstrap trong CLAUDE.md: mỗi session mới phải đọc progress_log.md (entry mới nhất) + model_registry.md trước khi làm việc, vì memory riêng của Claude không tồn tại trên máy GPU mới.
- Chưa chạy thử script này trên máy GPU thật — cần verify ở lần thuê máy đầu tiên.
- Sửa: pin cứng Python 3.10 (trước đó script chỉ dùng `python3` mặc định hệ thống, không nhất quán với triết lý pin version). Script tự cài qua deadsnakes PPA nếu máy thuê không có sẵn 3.10.
