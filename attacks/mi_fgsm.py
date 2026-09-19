"""Tấn công chuyển giao baseline cho Thí nghiệm 1 (research_plan.md §6.3): MI-FGSM (Dong et al.,
2018) — momentum L_inf iterative — áp dụng lên một objective kiểu DAG (Xie et al., 2017) chỉ
nhắm đúng RoI của ground-truth thay vì toàn bộ loss huấn luyện.

**Vì sao không dùng thẳng model.loss() (toàn bộ RPN+RCNN loss)**: đã thử trước (xem
docs/progress_log.md entry cùng ngày) — tăng tổng loss huấn luyện KHÔNG tương đương evasion.
RPN/RCNN loss cộng dồn trên hàng trăm/nghìn anchor/proposal nền (background), nên gradient ascent
"ăn gian" bằng cách biến vùng nền thành vật thể giả (confidence ~1.0, hàng chục box mới) — tăng
loss thật nhưng đó là fabrication, không phải né tránh (research_plan.md dùng đúng từ "né tránh").

**Fix**: chỉ tính cross-entropy trên đúng RoI ứng với GT box thật (dùng bbox_roi_extractor +
bbox_head có sẵn của mmdet, bỏ qua RPN/sampler) rồi maximize CE của đúng label GT tại đó — đẩy
predicted class ra khỏi GT (thành lớp khác hoặc thành background), không tạo tín hiệu ở vùng nền
vì không có RoI nền nào tham gia loss. Ảnh hưởng đã verify thực nghiệm: số detection giảm/vật thể
biến mất đúng như kỳ vọng evasion, không còn flood false-positive.
"""
import torch
import torch.nn.functional as F
from mmdet.structures.bbox import bbox2roi


def mi_fgsm_attack(model, x, data_sample, epsilon=8.0, num_iter=10, decay=1.0):
    """MI-FGSM, L-infinity, không target, objective kiểu DAG (chỉ RoI của GT).

    Args:
        model: detector mmdet đã eval(), dùng làm surrogate (white-box).
        x: tensor [C,H,W], float32, thang pixel gốc [0,255], đã qua Resize của test_pipeline
           (chưa pad/normalize — 2 bước đó nằm trong model.data_preprocessor).
        data_sample: DetDataSample tương ứng (đã có gt_instances từ pipeline).
        epsilon: ngân sách L_inf tính trên thang pixel [0,255] (8/255 * 255 = 8, theo
                 research_plan.md §6.4 — epsilon chuẩn là 8/255 trên ảnh [0,1]).
        num_iter: số vòng lặp (mặc định 10, alpha = epsilon/num_iter theo đúng quy ước gốc
                  MI-FGSM/BIM: alpha*num_iter vừa đủ phủ hết ngân sách epsilon).
        decay: hệ số momentum (mặc định 1.0, giá trị gốc trong paper MI-FGSM).

    Returns:
        x_adv: tensor [C,H,W], float32, thang pixel [0,255], cùng shape với x.
        Nếu ảnh không có GT box nào, trả về x không đổi (không có gì để tấn công).
    """
    gt_bboxes = data_sample.gt_instances.bboxes
    if hasattr(gt_bboxes, "tensor"):
        gt_bboxes = gt_bboxes.tensor
    gt_bboxes = gt_bboxes.to(x.device)
    gt_labels = data_sample.gt_instances.labels.to(x.device)

    if gt_bboxes.numel() == 0:
        return x.detach().clone()

    rois = bbox2roi([gt_bboxes])

    alpha = epsilon / num_iter
    x_clean = x.detach().clone()
    x_adv = x.detach().clone()
    g = torch.zeros_like(x)

    for _ in range(num_iter):
        x_adv.requires_grad_(True)
        data = dict(inputs=[x_adv], data_samples=[data_sample])
        batch = model.data_preprocessor(data, training=False)
        feats = model.extract_feat(batch["inputs"])
        bbox_feats = model.roi_head.bbox_roi_extractor(
            feats[: model.roi_head.bbox_roi_extractor.num_inputs], rois)
        cls_score, _ = model.roi_head.bbox_head(bbox_feats)
        # evasion = tăng loss (gradient ASCENT): đẩy cls_score ra khỏi đúng nhãn GT tại đúng vị
        # trí GT — không có RoI nền nào tham gia nên không có đường "ăn gian" bằng fabrication.
        loss = F.cross_entropy(cls_score, gt_labels)

        grad = torch.autograd.grad(loss, x_adv)[0]
        # L1-normalize gradient rồi cộng dồn momentum — đúng công thức MI-FGSM gốc.
        grad_norm = grad / (grad.abs().mean() + 1e-12)
        g = decay * g + grad_norm

        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.clamp(x_adv, x_clean - epsilon, x_clean + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 255.0)

    return x_adv.detach()
