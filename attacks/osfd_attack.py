"""OSFD (Wu et al., AAAI 2024, "Transferable Adversarial Attacks for Object Detection using
Object-Aware Significant Feature Distortion") — port MATCHED-BUDGET vào pipeline mmdet v3.3.0 của
dự án này, để so sánh công bằng với method v0.1 (attacks/backward_reg_attack.py) dưới cùng threat
model (epsilon=8.0, num_iter=10). Chốt 2026-09-21 — xem docs/progress_log.md entry cùng ngày.

Đọc TRỰC TIẾP official source code (github.com/wakuwu/OSFD, user đã clone vào /workspace/OSFD) để
port đúng thuật toán thật, không đoán từ paper text:

  - `attack/ours/OSFD.py`: loss = MSE(k * feat_clean, feat_adv), k=3.0 mặc định, tính TRÊN CẢ 4
    STAGE BACKBONE (KHÔNG mask theo GT box — "Object-Aware" trong tên paper nằm ở RRB bên dưới,
    không phải ở loss này).
  - `attack/base/RRB.py` (Random Rotation + adaptive Resizing + Blur — 1 trong các base_attack họ
    combine cùng MI): mỗi iteration tạo 2 "view" của ảnh adv, NỐI TIẾP nhau (không phải 2 nhánh độc
    lập): view1 = rotate quanh tâm 1 GT box ngẫu nhiên (góc ngẫu nhiên trong [-theta,theta]); view2
    = resize thích ứng theo kích thước 1 GT box ngẫu nhiên ÁP LÊN view1 (không phải lên ảnh gốc).
    Cả 2 view sau đó qua Gaussian blur (sigma cố định) rồi mới forward qua backbone. Loss cộng dồn
    qua cả 2 view (không trung bình).
  - `attack/base/MI.py`: momentum y hệt MI-FGSM đã dùng ở mọi nơi khác trong dự án này
    (grad_norm = grad/mean(|grad|), g = decay*g_prev + grad_norm).

Khác biệt CỐ Ý với official repo (matched-budget, không phải sai sót — xem docs/progress_log.md):
  - epsilon=8.0 (pixel scale), num_iter=10, alpha=epsilon/num_iter — convention của TOÀN BỘ attack
    khác trong dự án — thay vì epsilon=5 gốc + 20 epoch x 10 step (200 step tích lũy trong cùng 1
    quả bóng epsilon) + alpha=1.0 cố định mỗi bước. Native-config đầy đủ không khả thi thời gian
    trên máy hiện tại — dùng "OSFD-extended" (attacks/osfd_attack.py cùng module, budget lớn hơn)
    làm sanity check phụ, không dùng để claim thắng/thua chính.
  - Hyperparameter RRB/k lấy từ `config/attack_faster_rcnn.yaml` của official repo (surrogate gần
    nhất với R50 của dự án — repo gốc không có sẵn config Mask R-CNN R50): k=3.0, theta=7.0,
    l_s=10, rho=0.8, s_max=1.10, sigma=6.0. RRB_PROB=1.0 là default của class RRB (không override
    trong yaml đó).
"""
import random

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate

K = 3.0
RRB_THETA = 7.0
RRB_L_S = 10
RRB_RHO = 0.8
RRB_S_MAX = 1.10
RRB_SIGMA = 6.0
RRB_PROB = 1.0


def _random_axis_rotation(x, gt_bboxes, theta, l_s):
    """x: (1,C,H,W). gt_bboxes: (N,4) xyxy, tọa độ network-space (đã scale theo scale_factor).

    LƯU Ý FIDELITY (verify 2026-09-21, xem docs/progress_log.md): official RRB.random_axis_rotation
    (attack/base/RRB.py dòng 64) dùng fallback "tâm ảnh" = [H//2, W//2] (shape[-2]=H, shape[-1]=W)
    trong khi các box center khác ở dạng (x,y)=(cx,cy) — ĐẢO thứ tự trục so với box center, và so
    với tham số `center=[x,y]` mà torchvision.transforms.functional.rotate mong đợi. Đây là 1 quirk
    thật trong code gốc (không phải cách hiểu sai của mình) — cố ý REPLICATE ĐÚNG y hệt (không "sửa
    lỗi") vì mục tiêu port là fidelity với thuật toán họ THỰC SỰ chạy, không phải bản đã tối ưu lại."""
    device = x.device
    boxes_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
    img_center = torch.tensor([[x.shape[-2] // 2, x.shape[-1] // 2]], device=device, dtype=boxes_centers.dtype)
    centers = torch.cat([boxes_centers, img_center], dim=0)
    if l_s > 0:
        centers = centers + torch.randint_like(centers, low=-l_s, high=l_s)
    cx, cy = centers[random.randrange(centers.shape[0])].tolist()
    angle = random.random() * 2 * theta - theta
    return rotate(x, angle, center=[int(cx), int(cy)])


def _adaptive_random_resizing(x, gt_bboxes, rho, s_max):
    ori_h, ori_w = x.shape[2], x.shape[3]
    boxes = gt_bboxes.detach().cpu().numpy()
    box = boxes[random.randrange(len(boxes))]
    box_w, box_h = box[2] - box[0], box[3] - box[1]

    scale_h = min(1 + rho * (box_h / ori_h), s_max)
    scale_w = min(1 + rho * (box_w / ori_w), s_max)
    new_h = random.randint(ori_h, max(ori_h, int(scale_h * ori_h)))
    new_w = random.randint(ori_w, max(ori_w, int(scale_w * ori_w)))
    rescaled = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=True)

    rem_h = max(0, int(scale_h * ori_h) - new_h)
    rem_w = max(0, int(scale_w * ori_w) - new_w)
    pad_top = random.randint(0, rem_h)
    pad_left = random.randint(0, rem_w)
    padded = F.pad(rescaled, (pad_left, rem_w - pad_left, pad_top, rem_h - pad_top), mode="constant", value=0.0)
    return F.interpolate(padded, size=(ori_h, ori_w), mode="bilinear", align_corners=True)


def _gaussian_blur(x, sigma):
    return torch.clamp(x + torch.randn_like(x) * sigma, 0.0, 255.0)


def _rrb_views(x_adv, gt_bboxes, prob=RRB_PROB, theta=RRB_THETA, l_s=RRB_L_S, rho=RRB_RHO,
               s_max=RRB_S_MAX, sigma=RRB_SIGMA):
    """Trả về batch (2,C,H,W): view1=rotate(x_adv), view2=resize(view1) — NỐI TIẾP, đúng
    attack/base/RRB.py::RRB.preprocess_data (không phải 2 nhánh độc lập từ cùng gốc)."""
    cur = x_adv.unsqueeze(0)
    views = []
    if random.random() < prob:
        cur = _random_axis_rotation(cur, gt_bboxes, theta, l_s)
    views.append(cur)
    if random.random() < prob:
        cur = _adaptive_random_resizing(cur, gt_bboxes, rho, s_max)
    views.append(cur)
    return _gaussian_blur(torch.cat(views, dim=0), sigma)


def iterative_osfd_attack(model, x, data_sample, epsilon=8.0, num_iter=10, decay=1.0, k=K,
                           use_rrb=True):
    """OSFD: feature-distortion attack thuần túy (KHÔNG dùng bbox_head/RoI cls loss như các attack
    khác trong dự án — đúng thiết kế gốc, chỉ maximize MSE(k*feat_clean, feat_adv) trên backbone)."""
    gt_bboxes = data_sample.gt_instances.bboxes
    if hasattr(gt_bboxes, "tensor"):
        gt_bboxes = gt_bboxes.tensor
    gt_bboxes = gt_bboxes.to(x.device)
    if gt_bboxes.numel() == 0:
        return x.detach().clone()

    w_scale, h_scale = data_sample.metainfo["scale_factor"]
    scale = torch.tensor([w_scale, h_scale, w_scale, h_scale], device=x.device, dtype=gt_bboxes.dtype)
    gt_bboxes_net = gt_bboxes * scale

    with torch.no_grad():
        data_cln = dict(inputs=[x], data_samples=[data_sample])
        batch_cln = model.data_preprocessor(data_cln, training=False)
        feats_cln = model.backbone(batch_cln["inputs"])

    alpha = epsilon / num_iter
    x_clean = x.detach().clone()
    x_adv = x.detach().clone()
    g = torch.zeros_like(x)

    for _ in range(num_iter):
        x_adv.requires_grad_(True)

        x_views = _rrb_views(x_adv, gt_bboxes_net) if use_rrb else x_adv.unsqueeze(0)

        data = dict(inputs=list(x_views), data_samples=[data_sample] * x_views.shape[0])
        batch = model.data_preprocessor(data, training=False)
        feats_adv = model.backbone(batch["inputs"])

        # official: mse_loss (reduction="mean") RIÊNG mỗi (stage, view) rồi SUM tất cả. Ở đây gộp
        # các view cùng stage thành 1 batch rồi mean 1 lần — nhân lại N view để "mean của batch N"
        # == "tổng N mean riêng lẻ" (đúng vì mọi view cùng kích thước không gian).
        loss = sum(
            F.mse_loss(k * feats_cln[l][0:1].expand_as(feats_adv[l]), feats_adv[l]) * feats_adv[l].shape[0]
            for l in range(len(feats_cln))
        )

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad_norm = grad / (grad.abs().mean() + 1e-12)
        g = decay * g + grad_norm

        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.clamp(x_adv, x_clean - epsilon, x_clean + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 255.0)

    return x_adv.detach()
