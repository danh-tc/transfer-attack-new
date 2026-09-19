# Model Registry (Danh mục mô hình)

Chỉ ghi các thông tin đã được xác minh — không điền một dòng chỉ dựa trên suy đoán hay dựa vào tên model. Kiểu detector head không đồng nghĩa với họ backbone (xem research_plan.md §3, ví dụ Deformable DETR + ResNet-50). Phải xác minh backbone bằng cách đọc config/kiến trúc thật trước khi ghi vào đây.

## Surrogate

| Model | Kiến trúc detector | Backbone | Họ backbone | Đường dẫn config | Nguồn checkpoint | Ngày verify |
|---|---|---|---|---|---|---|
| Mask R-CNN + ResNet-50 | MaskRCNN (RPN + RoIHead 2-stage; chỉ dùng box output) | ResNet, depth=50 (`type='ResNet'`, verify trực tiếp trong `configs/_base_/models/mask-rcnn_r50_fpn.py`) | ResNet | `third_party/mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py` | https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth (box AP công bố: 38.2) | 2026-09-19 |

## Targets

| Model | Kiến trúc detector | Backbone | Họ backbone | Đường dẫn config | Nguồn checkpoint | Ngày verify |
|---|---|---|---|---|---|---|
| Mask R-CNN + ResNet-101 (same-family) | MaskRCNN, giống hệt surrogate (chỉ đổi `depth=101`) | ResNet, depth=101 | ResNet | `third_party/mmdetection/configs/mask_rcnn/mask-rcnn_r101_fpn_1x_coco.py` | https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth (box AP công bố: 40.0) | 2026-09-19 |
| Mask R-CNN + ConvNeXt-Tiny (cross-CNN-family) | MaskRCNN | ConvNeXt, arch='tiny' (`type='mmpretrain.ConvNeXt'`, verify trực tiếp trong config — đăng ký qua registry mmpretrain, **bắt buộc** cài mmpretrain, xem environment_setup.md) | ConvNeXt | `third_party/mmdetection/configs/convnext/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py` | https://download.openmmlab.com/mmdetection/v2.0/convnext/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco_20220426_154953-050731f4.pth (box AP công bố: 46.2) | 2026-09-19 |
| Mask R-CNN + Swin-Tiny (CNN → Transformer) | MaskRCNN | Swin Transformer, embed_dims=96, depths=[2,2,6,2] (`type='SwinTransformer'`, verify trực tiếp trong config) | Transformer (Swin) | `third_party/mmdetection/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py` | https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth (box AP công bố: 42.7) | 2026-09-19 |

## Tham chiếu họ backbone

Dùng để phân loại nhanh sau khi đã kiểm tra model:

- **CNN — họ ResNet**: ResNet-50, ResNet-101
- **CNN — khác**: DarkNet, CSPNet, ConvNeXt
- **Transformer**: Swin Transformer, PVT, ViT

## Ghi chú / câu hỏi còn mở

- MMDetection v3.3.0 không có Faster R-CNN + ConvNeXt hoặc Faster R-CNN + Swin — đã đổi cả bộ (surrogate + target) sang Mask R-CNN để giữ kiến trúc detector cố định xuyên cả 3 target. Chi tiết lý do: research_plan.md §6.1–§6.2, docs/progress_log.md entry 2026-09-19.
- **Chênh lệch training recipe giữa các target (chưa giải quyết, cần cân nhắc khi diễn giải kết quả)**: checkpoint chính thức của Mask R-CNN+ResNet-50/101 và Mask R-CNN+Swin-Tiny đều dùng schedule 1x (12 epoch, không augmentation mạnh). Checkpoint Mask R-CNN+ConvNeXt-Tiny chỉ có bản schedule 3x + AMP + multi-scale crop (36 epoch, augmentation mạnh hơn nhiều) — không có bản 1x tương đương trong MMDetection. Nghĩa là target ConvNeXt được train lâu hơn/kỹ hơn hẳn 2 target còn lại, một biến gây nhiễu nằm ngoài kiểm soát của thí nghiệm (không phải do kiến trúc backbone). Nếu ConvNeXt cho transfer tốt hơn kỳ vọng, cần cân nhắc khả năng đây là hiệu ứng từ training recipe mạnh hơn chứ không hẳn do họ backbone.
- `match_low_quality` trong RPN assigner khác nhau giữa base config Faster R-CNN (`False`) và Mask R-CNN (`True`) — không ảnh hưởng thí nghiệm hiện tại vì toàn bộ model đều dùng Mask R-CNN (đồng nhất), chỉ ghi lại để không nhầm là hai base config giống hệt 100%.
