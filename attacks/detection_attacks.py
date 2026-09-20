"""Baseline tấn công chuyển giao cho Thí nghiệm 1 + 1B (research_plan.md §6.3, §11).

Một hàm tấn công lặp L_inf tổng quát, tham số hóa 3 trục biến thể (Thí nghiệm 1B,
docs/progress_log.md 2026-09-19 "Exp1B"):

- `decay`: 0.0 = BIM/I-FGSM thuần (không momentum), 1.0 = MI-FGSM (Dong et al. 2018).
- `objective`: "cls" = chỉ cross-entropy tại RoI của GT (mặc định Thí nghiệm 1, xem lý do chọn
  trong attacks/mi_fgsm.py cũ / progress_log — tránh false-positive flooding); "cls_bbox" = thêm
  cả loss hồi quy bbox tại RoI của GT (kiểu DAG đầy đủ hơn — tấn công cả classification lẫn
  localization, không chỉ nhãn).
- `input_diversity`: bật DI (Xie et al. 2019, "Improving Transferability... with Input
  Diversity") — resize ngẫu nhiên + pad ngẫu nhiên mỗi vòng lặp trước khi tính gradient, một
  baseline "augmentation-based transfer" được research_plan.md §11 liệt kê.

Không gian tấn công / lý do dùng RoI-của-GT thay vì toàn bộ model.loss(): xem attacks/mi_fgsm.py
(giữ nguyên, không xóa — vẫn được experiments/experiment1.py dùng làm baseline chính).
"""
import random

import torch
import torch.nn.functional as F
from mmdet.structures.bbox import bbox2roi


def _bbox_cls_bbox_loss(model, feats, gt_bboxes, gt_labels):
    """CE trên cls_score + L1 trên bbox_pred tại đúng RoI của GT, cả hai maximize để evasion.

    Target hồi quy = vector 0: encode(roi, gt_bbox) với roi==gt_bbox luôn cho delta=0 (một RoI
    trùng khít GT không cần điều chỉnh gì) — không cần gọi bbox_coder.encode, có thể viết thẳng.
    """
    rois = bbox2roi([gt_bboxes])
    bbox_feats = model.roi_head.bbox_roi_extractor(
        feats[: model.roi_head.bbox_roi_extractor.num_inputs], rois)
    cls_score, bbox_pred = model.roi_head.bbox_head(bbox_feats)
    cls_loss = F.cross_entropy(cls_score, gt_labels)

    bh = model.roi_head.bbox_head
    if bh.reg_class_agnostic:
        bbox_pred_sel = bbox_pred
    else:
        n, num_classes = bbox_pred.shape[0], bh.num_classes
        bbox_pred_sel = bbox_pred.view(n, num_classes, 4)[torch.arange(n, device=bbox_pred.device), gt_labels]
    zero_target = torch.zeros_like(bbox_pred_sel)
    bbox_loss = bh.loss_bbox(bbox_pred_sel, zero_target)

    return cls_loss + bbox_loss


def _bbox_cls_loss(model, feats, gt_bboxes, gt_labels):
    rois = bbox2roi([gt_bboxes])
    bbox_feats = model.roi_head.bbox_roi_extractor(
        feats[: model.roi_head.bbox_roi_extractor.num_inputs], rois)
    cls_score, _ = model.roi_head.bbox_head(bbox_feats)
    return F.cross_entropy(cls_score, gt_labels)


def _apply_input_diversity(x, gt_bboxes, max_pad=16, prob=0.7):
    """DI-FGSM: resize ngẫu nhiên lớn hơn ảnh gốc rồi pad ngẫu nhiên — đồng thời biến đổi tọa độ
    GT box tương ứng (scale + offset) để RoI vẫn đúng vị trí vật thể trên ảnh đã biến đổi.
    Trả về (x_transformed, gt_bboxes_transformed). Với xác suất (1-prob), trả về nguyên trạng."""
    if random.random() > prob:
        return x, gt_bboxes

    c, h, w = x.shape
    new_h = random.randint(h, h + max_pad)
    new_w = random.randint(w, w + max_pad)
    resized = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
    scale_h, scale_w = new_h / h, new_w / w

    pad_h, pad_w = (h + max_pad) - new_h, (w + max_pad) - new_w
    pad_top = random.randint(0, pad_h)
    pad_left = random.randint(0, pad_w)
    padded = F.pad(resized, [pad_left, pad_w - pad_left, pad_top, pad_h - pad_top])

    scale = torch.tensor([scale_w, scale_h, scale_w, scale_h], device=x.device, dtype=gt_bboxes.dtype)
    offset = torch.tensor([pad_left, pad_top, pad_left, pad_top], device=x.device, dtype=gt_bboxes.dtype)
    transformed_boxes = gt_bboxes * scale + offset
    return padded, transformed_boxes


def iterative_linf_attack(model, x, data_sample, epsilon=8.0, num_iter=10, decay=1.0,
                           objective="cls", input_diversity=False):
    """Tấn công lặp L_inf, không target, trên objective kiểu DAG (RoI của GT).

    Args:
        model, x, data_sample, epsilon, num_iter: xem attacks/mi_fgsm.py.
        decay: 0.0 = BIM/I-FGSM (không momentum), 1.0 = MI-FGSM (momentum gốc, mặc định).
        objective: "cls" (mặc định, dùng cho Thí nghiệm 1) hoặc "cls_bbox".
        input_diversity: bật biến đổi DI-FGSM mỗi vòng lặp (mặc định tắt).

    Returns:
        x_adv: tensor cùng shape với x, thang pixel [0,255].
    """
    gt_bboxes = data_sample.gt_instances.bboxes
    if hasattr(gt_bboxes, "tensor"):
        gt_bboxes = gt_bboxes.tensor
    gt_bboxes = gt_bboxes.to(x.device)
    gt_labels = data_sample.gt_instances.labels.to(x.device)

    if gt_bboxes.numel() == 0:
        return x.detach().clone()

    # BUG FIX 2026-09-20: gt_instances.bboxes ở tọa độ ẢNH GỐC (ori_shape) — test pipeline chạy
    # LoadAnnotations SAU Resize nên GT không được resize theo, trong khi feats tính từ ảnh ĐÃ
    # resize (img_shape, dùng bởi model.extract_feat). Không scale lại thì RoIAlign trích sai vùng
    # feature (lệch tỷ lệ = scale_factor, ~1.8x quan sát được, nặng hơn với object xa gốc tọa độ).
    # Phát hiện bằng cách so trực tiếp gt_instances.bboxes với annotation gốc trong file COCO —
    # khớp tuyệt đối, xác nhận chưa hề qua resize. Xem docs/progress_log.md 2026-09-20.
    w_scale, h_scale = data_sample.metainfo["scale_factor"]
    scale = torch.tensor([w_scale, h_scale, w_scale, h_scale], device=x.device, dtype=gt_bboxes.dtype)
    gt_bboxes = gt_bboxes * scale

    loss_fn = {"cls": _bbox_cls_loss, "cls_bbox": _bbox_cls_bbox_loss}[objective]

    alpha = epsilon / num_iter
    x_clean = x.detach().clone()
    x_adv = x.detach().clone()
    g = torch.zeros_like(x)

    for _ in range(num_iter):
        x_adv.requires_grad_(True)

        if input_diversity:
            x_in, boxes_in = _apply_input_diversity(x_adv, gt_bboxes)
        else:
            x_in, boxes_in = x_adv, gt_bboxes

        data = dict(inputs=[x_in], data_samples=[data_sample])
        batch = model.data_preprocessor(data, training=False)
        feats = model.extract_feat(batch["inputs"])
        loss = loss_fn(model, feats, boxes_in, gt_labels)

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad_norm = grad / (grad.abs().mean() + 1e-12)
        g = decay * g + grad_norm

        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.clamp(x_adv, x_clean - epsilon, x_clean + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 255.0)

    return x_adv.detach()
