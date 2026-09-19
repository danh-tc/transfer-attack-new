# transfer-attack-new

Dự án nghiên cứu: tấn công né tránh (evasion) có khả năng chuyển giao xuyên họ feature-extractor, dùng single-surrogate, cho object detection (CNN → Transformer). Kế hoạch đầy đủ, định nghĩa, câu hỏi nghiên cứu: [docs/research_plan.md](docs/research_plan.md).

## Giai đoạn hiện tại

Trước Thí nghiệm 1 (Pre-Experiment-1). Chưa có code, chưa có môi trường, chưa verify model nào. Việc trước mắt là research_plan.md §13: kiểm tra repo/model registry, xác minh model nào thực sự dùng backbone ResNet / ConvNeXt / Swin, đề xuất bộ model kiểm soát nhỏ nhất cho Thí nghiệm 1.

## Docs

- [docs/research_plan.md](docs/research_plan.md) — kim chỉ nam: mục tiêu, định nghĩa, thiết kế thí nghiệm, câu hỏi nghiên cứu. Chỉ sửa khi có pivot phương pháp luận có chủ đích.
- [docs/progress_log.md](docs/progress_log.md) — nhật ký theo ngày, append-only, ghi lại các lần chạy/kết quả/quyết định. Kiểm tra file này trước để biết đã thử gì rồi.
- [docs/model_registry.md](docs/model_registry.md) — surrogate/target đã verify và họ backbone thật của chúng. Tin vào đây thay vì đoán từ tên model.
- [docs/environment_setup.md](docs/environment_setup.md) — vì sao venv được dựng như vậy, version đã pin, troubleshooting.

## Bootstrap khi bắt đầu session mới

Máy chạy thí nghiệm là GPU thuê, luôn setup lại từ đầu — không có gì tồn tại ngoài repo này. File CLAUDE.md này tự động load mỗi session nên đã đưa context cơ bản vào ngay, nhưng **trước khi bắt tay làm việc thực tế, luôn đọc**:
1. `docs/progress_log.md` — ít nhất entry mới nhất, để biết lần trước dừng ở đâu, quyết định gì.
2. `docs/model_registry.md` — nếu việc sắp làm liên quan đến model/backbone, không đoán lại từ đầu.

Không giả định memory riêng của Claude còn từ phiên trước — trên máy GPU mới, memory đó trống. Mọi context bắt buộc phải nằm trong docs/, không phải trong memory.

## Chạy môi trường

```bash
bash scripts/setup_env.sh   # 1 lệnh duy nhất, dựng venv từ đầu, idempotent trên máy fresh
source .venv/bin/activate
```

Chi tiết/troubleshooting: [docs/environment_setup.md](docs/environment_setup.md).

## Quy ước

- Không cài đặt phương pháp tấn công mới trước khi Thí nghiệm 1 (xác minh transfer gap) hoàn tất — xem research_plan.md §6.7, §8.
- Mỗi lần chạy thí nghiệm: cùng config tấn công cho mọi target, cố định budget/iterations/seed, kết quả ghi vào progress_log.md.
- Tất cả docs trong dự án này viết bằng tiếng Việt.
- Không commit `.venv/`, `third_party/`, checkpoint, dataset — xem `.gitignore`. Version môi trường đã pin trong `scripts/setup_env.sh`, không phụ thuộc gì cài thủ công ngoài đó.
