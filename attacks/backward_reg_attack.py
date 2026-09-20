"""Thí nghiệm 3A (research_plan.md, chốt 2026-09-20 sau khi khóa RQ2 — xem docs/progress_log.md):
causal intervention — nếu backward sensitivity ở Stage 3-4 THỰC SỰ là cơ chế chi phối transfer gap
(không chỉ correlation như 2A/2C), thì regularize (giảm variance) gradient CHÍNH XÁC tại Stage 3-4
trong lúc craft attack phải tăng cross-family ASR nhiều hơn regularize ở Stage 1-2.

Baseline regularizer CỐ Ý đơn giản, lấy tinh thần TGR (Zhang et al., CVPR 2023 — "Transferable
Adversarial Attacks on Vision Transformers with Token Gradient Regularization": giảm variance của
gradient lan truyền ngược tại block trung gian bằng cách loại bỏ giá trị cực trị). TGR gốc làm ở
mức token/attention-block cho ViT — ở đây áp dụng ở granularity BACKBONE STAGE (feature map, không
phải token), để dùng thống nhất được cho cả CNN (ResNet) lẫn Transformer (Swin) lẫn ConvNeXt.
KHÔNG phải reimplementation TGR, chỉ mượn nguyên lý "clip gradient theo variance tại điểm trung
gian" làm regularizer đơn giản nhất có thể biện minh được, đúng tinh thần "chưa tạo method novel
ngay, chỉ test causal hypothesis" (docs/progress_log.md 2026-09-20).

Cơ chế: đăng ký backward hook trực tiếp lên tensor feature map của stage cần regularize (KHÔNG
phải hook trên Module — cần hook trên tensor để chỉ chỉnh sửa đúng gradient chảy qua điểm đó, xem
torch.Tensor.register_hook). Hook chạy trong lúc autograd backward tự nhiên khi tính gradient của
loss đối với input — không cần đổi gì ở phần forward hay ở loss function.

Vì cần chèn hook giữa backbone và neck, không dùng model.extract_feat() (gộp cả 2 bước) mà tách ra
gọi model.backbone() rồi model.neck() thủ công — 2 bước này CHÍNH XÁC là những gì extract_feat làm
bên trong (mmdet BaseDetector.extract_feat), không thay đổi hành vi gì khác ngoài việc chèn hook.

---

Thí nghiệm 3B (chốt 2026-09-20 sau khi 3A confirm causal + literature search overlap — xem
docs/progress_log.md): 3A dùng 1 kiểu regularize DUY NHẤT (clip variance đồng đều trên toàn bộ
feature map của stage). Câu hỏi 3B: object-conditioned weighting — tập trung regularize vào đúng
vùng feature tương ứng với object (thay vì đều tay trên toàn bộ feature map, lẫn cả background) —
có giúp ích hơn không? Thêm 2 mode mới, dùng chung `_forward_feats_with_reg` qua tham số
`stage_modes` (dict stage -> mode, ưu tiên hơn `reg_stages` cũ nếu cả 2 cùng truyền):

  - "clip": y hệt 3A (biến `reg_stages` cũ tương đương mode="clip" cho từng stage trong đó).
  - "weight": KHÔNG clip, chỉ nhân gradient với mask không gian (1.0 trong vùng object theo GT box
    đã scale về đúng resolution của stage đó qua stride, `OBJECT_MASK_BG_WEIGHT` ở ngoài) — cô lập
    tác dụng của "tập trung vào object" khỏi tác dụng của "giảm variance".
  - "clip_weight": áp clip (như "clip") nhưng CHỈ TRONG vùng object (blend với gradient gốc bên
    ngoài theo mask) — kết hợp cả 2 ý tưởng, test xem có tốt hơn "clip" thuần hay không.

Mask xây dựng đơn giản (hard box theo đúng GT box, không làm mượt biên bằng Gaussian) — đủ dùng cho
1 kiểm tra targeted, không phải thiết kế cuối cùng.

---

Method v0.1 (formalized, chốt 2026-09-20 — xem docs/research_plan.md §9.1 để có công thức đầy đủ
khớp chính xác với code dưới đây). 2 hằng số trước đây hard-code (`3*std` cho clip bound,
`OBJECT_MASK_BG_WEIGHT=0.3` cho mask) nay tổng quát hóa thành tham số `k` và `lam` để chạy Ablation
A (clip bound) / Ablation B (object-weight strength) mà không cần sửa code — mặc định giữ nguyên
giá trị cũ nên mọi lời gọi hiện có (Thí nghiệm 3A/3B) tái lập y hệt kết quả đã chạy.

`lam` (λ, research_plan.md §9.1): β(λ) = 1/(1+λ). λ=0 → β=1 → mask đồng nhất → công thức "clip_weight"
suy biến CHÍNH XÁC về "clip" (= reg_s3s4, mọi pixel đều clip đều tay). λ tăng → β giảm → blend ngoài
vùng object nghiêng dần về gradient gốc. `DEFAULT_LAM` dưới đây là giá trị λ tương ứng đúng
β=0.3 đã dùng ở Thí nghiệm 3B (λ ≈ 2.333) — giữ mặc định này để không đổi hành vi cũ.
"""
import torch

from attacks.detection_attacks import _bbox_cls_bbox_loss, _bbox_cls_loss, _apply_input_diversity

# Giống hệt experiments/experiment2b.py STAGE_STRIDES (đã verify 2026-09-20: cả 4 model cùng
# stride) — định nghĩa lại ở đây (không import từ experiments/) để attacks/ không phụ thuộc
# ngược vào experiments/.
STAGE_STRIDES = [4, 8, 16, 32]
OBJECT_MASK_BG_WEIGHT = 0.3  # beta dùng ở Thí nghiệm 3B (giữ lại làm hằng số tham chiếu/tài liệu)
DEFAULT_K = 3  # clip bound mặc định = k*std (Thí nghiệm 3A/3B) — đối tượng Ablation A
DEFAULT_LAM = 1.0 / OBJECT_MASK_BG_WEIGHT - 1  # ≈2.333, tái lập đúng beta=0.3 — đối tượng Ablation B


def _bg_weight_from_lambda(lam):
    """beta(lambda) = 1/(1+lambda) — xem docs/research_plan.md §9.1. lambda=0 -> beta=1 -> mask
    đồng nhất -> "clip_weight" suy biến đúng về "clip" (reg_s3s4)."""
    return 1.0 / (1.0 + lam)


def _clip_extreme_grad_hook(grad):
    """Tinh thần TGR: loại bỏ giá trị cực trị của gradient bằng cách clip theo mean +/- 3*std của
    CHÍNH gradient đó (thích nghi theo scale riêng từng stage/feature map, không cần ngưỡng cố định
    thủ công qua các stage có channel/scale khác nhau)."""
    std, mean = grad.std(), grad.mean()
    bound = 3 * std
    return grad.clamp(mean - bound, mean + bound)


def _build_object_mask(shape, gt_bboxes_net, stride, device, bg_weight=OBJECT_MASK_BG_WEIGHT):
    """Mask không gian (1,1,H,W) tại đúng resolution của 1 stage: 1.0 trong vùng hợp (union) các GT
    box (đã quy đổi tọa độ theo stride), bg_weight ngoài vùng đó. Hard box, không làm mượt biên —
    đơn giản, đủ dùng để kiểm tra targeted (3B), không phải thiết kế method cuối cùng."""
    _, _, h, w = shape
    mask = torch.full((1, 1, h, w), bg_weight, device=device)
    boxes_stage = (gt_bboxes_net / stride).round().long()
    for i in range(boxes_stage.shape[0]):
        x1, y1, x2, y2 = boxes_stage[i].tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2 + 1), min(h, y2 + 1)
        if x2 > x1 and y2 > y1:
            mask[0, 0, y1:y2, x1:x2] = 1.0
    return mask


def _make_reg_hook(mode, mask=None, k=DEFAULT_K):
    def hook(grad):
        if mode == "clip":
            std, mean = grad.std(), grad.mean()
            bound = k * std
            return grad.clamp(mean - bound, mean + bound)
        if mode == "weight":
            return grad * mask
        if mode == "clip_weight":
            std, mean = grad.std(), grad.mean()
            bound = k * std
            clipped = grad.clamp(mean - bound, mean + bound)
            return mask * clipped + (1 - mask) * grad
        raise ValueError(f"unknown mode: {mode}")
    return hook


def _forward_feats_with_reg(model, x, reg_stages=frozenset(), stage_modes=None, gt_bboxes_net=None,
                             k=DEFAULT_K, lam=DEFAULT_LAM):
    """model.backbone(x) -> hook lên đúng stage -> model.neck(...).

    reg_stages: set stage (1-indexed) — API cũ (Thí nghiệm 3A), tương đương mode="clip" mọi stage.
    stage_modes: dict {stage: mode} — API mới (Thí nghiệm 3B), mode trong {"clip","weight",
        "clip_weight"}. Nếu truyền, ƯU TIÊN hơn reg_stages cho đúng stage đó. mode "weight"/
        "clip_weight" cần gt_bboxes_net (tọa độ network-space, đã scale theo scale_factor) để xây
        mask object — bắt buộc truyền nếu dùng 2 mode này.
    k: clip bound = k*std (Ablation A, research_plan.md §9.1). lam: object-weight strength, chuyển
        sang beta qua _bg_weight_from_lambda (Ablation B)."""
    stage_modes = dict(stage_modes or {})
    for s in reg_stages:
        stage_modes.setdefault(s, "clip")

    bg_weight = _bg_weight_from_lambda(lam)
    stage_feats = model.backbone(x)
    if stage_modes:
        stage_feats = list(stage_feats)
        for s, mode in stage_modes.items():
            feat = stage_feats[s - 1]
            mask = None
            if mode in ("weight", "clip_weight"):
                mask = _build_object_mask(feat.shape, gt_bboxes_net, STAGE_STRIDES[s - 1], feat.device, bg_weight)
            feat.register_hook(_make_reg_hook(mode, mask, k))
        stage_feats = tuple(stage_feats)
    if model.with_neck:
        return model.neck(stage_feats)
    return stage_feats


def iterative_linf_attack_reg(model, x, data_sample, epsilon=8.0, num_iter=10, decay=1.0,
                               objective="cls", input_diversity=False, reg_stages=frozenset(),
                               stage_modes=None, k=DEFAULT_K, lam=DEFAULT_LAM):
    """Giống hệt attacks/detection_attacks.py.iterative_linf_attack, chỉ thêm:
      - reg_stages: set stage cần regularize kiểu "clip" (Thí nghiệm 3A, giữ nguyên để không phá
        kết quả đã chạy).
      - stage_modes: dict {stage: mode} tổng quát hơn (Thí nghiệm 3B, xem _forward_feats_with_reg).
      - k, lam: 2 hyperparameter tổng quát hóa (Ablation A/B, research_plan.md §9.1) — mặc định
        giữ nguyên hành vi 3A/3B (k=3, lam≈2.333 tương đương beta=0.3).
    Cả reg_stages/stage_modes rỗng/None = baseline (y hệt attack gốc, không có gì khác)."""
    gt_bboxes = data_sample.gt_instances.bboxes
    if hasattr(gt_bboxes, "tensor"):
        gt_bboxes = gt_bboxes.tensor
    gt_bboxes = gt_bboxes.to(x.device)
    gt_labels = data_sample.gt_instances.labels.to(x.device)

    if gt_bboxes.numel() == 0:
        return x.detach().clone()

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
        feats = _forward_feats_with_reg(model, batch["inputs"], reg_stages, stage_modes, boxes_in, k=k, lam=lam)
        loss = loss_fn(model, feats, boxes_in, gt_labels)

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad_norm = grad / (grad.abs().mean() + 1e-12)
        g = decay * g + grad_norm

        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.clamp(x_adv, x_clean - epsilon, x_clean + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 255.0)

    return x_adv.detach()
