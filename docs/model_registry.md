# Model Registry (Danh mục mô hình)

Chỉ ghi các thông tin đã được xác minh — không điền một dòng chỉ dựa trên suy đoán hay dựa vào tên model. Kiểu detector head không đồng nghĩa với họ backbone (xem research_plan.md §3, ví dụ Deformable DETR + ResNet-50). Phải xác minh backbone bằng cách đọc config/kiến trúc thật trước khi ghi vào đây.

## Surrogate

| Model | Kiến trúc detector | Backbone | Họ backbone | Đường dẫn config | Nguồn checkpoint | Ngày verify |
|---|---|---|---|---|---|---|
| _chưa có_ | | | | | | |

## Targets

| Model | Kiến trúc detector | Backbone | Họ backbone | Đường dẫn config | Nguồn checkpoint | Ngày verify |
|---|---|---|---|---|---|---|
| _chưa có_ | | | | | | |

## Tham chiếu họ backbone

Dùng để phân loại nhanh sau khi đã kiểm tra model:

- **CNN — họ ResNet**: ResNet-50, ResNet-101
- **CNN — khác**: DarkNet, CSPNet, ConvNeXt
- **Transformer**: Swin Transformer, PVT, ViT

## Ghi chú / câu hỏi còn mở

- _(ghi lại đây bất kỳ trường hợp nào không rõ ràng, ví dụ model không xác định được backbone chỉ từ config)_
