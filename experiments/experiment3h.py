"""Thí nghiệm 3H (chốt 2026-09-21, nối tiếp Thí nghiệm 3G — xem docs/progress_log.md): 3G xác nhận
regularization (Stage 3-4, k=3, lam=0.5, mode clip_weight) giảm mạnh concentration/tail của backward
signal SURROGATE ở 1 lần forward+backward duy nhất (không phải attack thật). 3H hỏi tiếp: gradient
bớt concentrated có làm CHÍNH TRAJECTORY của attack thật (10 iteration, FGSM-style sign-update, y
hệt mọi thí nghiệm trước) ổn định hơn qua các bước hay không?

KHÁC 3G: ở đây không thể tính REG offline từ RAW bằng 1 lần backward — vì tại iteration t>1, x_adv
đã phụ thuộc vào gradient (có thể đã bị regularize) của các iteration trước, nên phải chạy 2 chuỗi
10-iteration ĐỘC LẬP HOÀN TOÀN trên cùng ảnh sạch: baseline (stage_modes rỗng — y hệt attack gốc,
không regularize gì) và reg (Stage 3-4, k=3, lam=0.5, mode clip_weight — đúng config method hiện
tại, khớp 3B/3D/3E/3F/3G). KHÔNG cần target model nào — đây thuần là thuộc tính nội tại của quỹ đạo
gradient trên surrogate, y hệt tinh thần 3G. Loop attack copy lại từ
attacks/backward_reg_attack.py.iterative_linf_attack_reg (KHÔNG sửa file đó, chỉ thêm logging mỗi
iteration — giữ nguyên logic gốc tuyệt đối, đã so khớp từng dòng).

Ghi log per iteration t=1..10 giá trị `grad_norm` (= grad/(|grad|.mean()+eps), đúng tensor được cộng
dồn vào momentum g = decay*g + grad_norm trong attack thật — không phải grad thô, vì grad_norm mới
là "hướng" thực sự chi phối quỹ đạo x_adv qua momentum, và cos/sign-flip bất biến với việc chia cho
1 hằng số dương nên dùng grad thô hay grad_norm cho ra cos/sign-flip giống hệt nhau; chỉ có
rel_change_norm là nhạy với lựa chọn này — dùng grad_norm cho nhất quán với "cái mà attack thật
dùng").

4 thống kê mỗi cặp iteration liên tiếp (t, t+1), t=1..9:
  - cos(g_t, g_{t+1}): cosine giữa 2 gradient liên tiếp — trajectory mượt hay giật cục.
  - sign_flip_rate: tỷ lệ phần tử đổi dấu giữa g_t và g_{t+1}.
  - rel_change_norm: ||g_{t+1} - g_t||_2 / ||g_t||_2.
Và so với gradient ban đầu (drift), t=2..10:
  - cos(g_1, g_t): trôi dạt so với hướng ban đầu.

Mỗi ảnh tổng hợp thành 5 scalar (mean qua các cặp/iteration + final drift) cho MỖI variant
(baseline/reg) — 1 ẢNH = 1 ROW (giống nguyên tắc đơn vị phân tích của 3G, không có vấn đề nhiều
object/ảnh không độc lập vì đây là thuộc tính của cả ảnh, không phải per-object). Kiểm định:
Wilcoxon signed-rank (paired theo ảnh) baseline vs reg cho từng scalar, giống 3G.

Hypothesis: reg ⇒ mean_cos_consecutive↑, mean_sign_flip_rate↓, mean_rel_change_norm↓,
mean_cos_drift↑ (trôi dạt so với hướng ban đầu chậm hơn).

Output: outputs/experiment3h/records.jsonl (1 dòng/ảnh, đủ cả 2 variant + per-iteration breakdown)
+ summary.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import wilcoxon

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attacks.backward_reg_attack import _forward_feats_with_reg  # noqa: E402
from attacks.detection_attacks import _bbox_cls_loss  # noqa: E402
from experiments import common  # noqa: E402
from experiments.experiment1 import MODELS  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402
register_all_modules(init_default_scope=True)

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment3h")

EPSILON = 8.0
NUM_ITER = 10
DECAY = 1.0
K = 3
LAM = 0.5
REG_STAGE_MODES = {3: "clip_weight", 4: "clip_weight"}
BASELINE_STAGE_MODES = {}
VARIANTS = {"baseline": BASELINE_STAGE_MODES, "reg": REG_STAGE_MODES}
SCALAR_METRICS = ["mean_cos_consecutive", "mean_sign_flip_rate", "mean_rel_change_norm", "mean_cos_drift"]


def _attack_trajectory(model, x, data_sample, stage_modes):
    """Copy logic y hệt attacks/backward_reg_attack.py.iterative_linf_attack_reg (objective="cls",
    input_diversity=False — cấu hình dùng ở mọi thí nghiệm trước), CHỈ thêm: trả về list grad_norm
    mỗi iteration thay vì chỉ trả về x_adv cuối."""
    gt_bboxes = data_sample.gt_instances.bboxes
    if hasattr(gt_bboxes, "tensor"):
        gt_bboxes = gt_bboxes.tensor
    gt_bboxes = gt_bboxes.to(x.device)
    gt_labels = data_sample.gt_instances.labels.to(x.device)
    if gt_bboxes.numel() == 0:
        return None

    w_scale, h_scale = data_sample.metainfo["scale_factor"]
    scale = torch.tensor([w_scale, h_scale, w_scale, h_scale], device=x.device, dtype=gt_bboxes.dtype)
    gt_bboxes = gt_bboxes * scale

    alpha = EPSILON / NUM_ITER
    x_clean = x.detach().clone()
    x_adv = x.detach().clone()
    g = torch.zeros_like(x)
    grad_norms = []

    for _ in range(NUM_ITER):
        x_adv.requires_grad_(True)
        data = dict(inputs=[x_adv], data_samples=[data_sample])
        batch = model.data_preprocessor(data, training=False)
        feats = _forward_feats_with_reg(model, batch["inputs"], stage_modes=stage_modes,
                                         gt_bboxes_net=gt_bboxes, k=K, lam=LAM)
        loss = _bbox_cls_loss(model, feats, gt_bboxes, gt_labels)
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad_norm = grad / (grad.abs().mean() + 1e-12)
        grad_norms.append(grad_norm.detach().clone())

        g = DECAY * g + grad_norm
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.clamp(x_adv, x_clean - EPSILON, x_clean + EPSILON)
        x_adv = torch.clamp(x_adv, 0.0, 255.0)

    return grad_norms


def _traj_metrics(grad_list):
    flat = [g.flatten() for g in grad_list]
    cos_consec, sign_flip, rel_change = [], [], []
    for t in range(len(flat) - 1):
        a, b = flat[t], flat[t + 1]
        cos_consec.append(float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)))
        sign_flip.append(float((torch.sign(a) != torch.sign(b)).float().mean()))
        rel_change.append(float((b - a).norm() / (a.norm() + 1e-12)))

    g1 = flat[0]
    cos_drift = [float(torch.dot(g1, flat[t]) / (g1.norm() * flat[t].norm() + 1e-12))
                 for t in range(1, len(flat))]

    return {
        "mean_cos_consecutive": float(np.mean(cos_consec)),
        "mean_sign_flip_rate": float(np.mean(sign_flip)),
        "mean_rel_change_norm": float(np.mean(rel_change)),
        "mean_cos_drift": float(np.mean(cos_drift)),
        "final_cos_drift": cos_drift[-1],
        "per_iter_cos_consecutive": cos_consec,
        "per_iter_sign_flip_rate": sign_flip,
        "per_iter_cos_drift": cos_drift,
    }


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[exp3h] loading surrogate...")
    cfg, ckpt, _family = MODELS["surrogate_r50"]
    surrogate = common.load_model(cfg, ckpt)

    print(f"[exp3h] building dataset (limit={limit})...")
    ds, n = common.build_dataset(cfg, limit=limit)
    print(f"[exp3h] {n} images (dataset total: {len(ds)})")

    records = []
    t0 = time.time()
    n_used = 0
    for i in range(n):
        sample = ds[i]
        ds_sample = sample["data_samples"]
        img_id = int(ds_sample.metainfo["img_id"])
        if ds_sample.gt_instances.labels.numel() == 0:
            continue

        x_clean = sample["inputs"].float().to(common.DEVICE)
        row = {"image_id": img_id}
        ok = True
        for variant, stage_modes in VARIANTS.items():
            grad_norms = _attack_trajectory(surrogate, x_clean, ds_sample, stage_modes)
            if grad_norms is None:
                ok = False
                break
            metrics = _traj_metrics(grad_norms)
            for k_, v in metrics.items():
                row[f"{variant}__{k_}"] = v
        if not ok:
            continue
        n_used += 1
        records.append(row)

        if (i + 1) % 20 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp3h] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút, "
                  f"{n_used} ảnh dùng được)")

    with open(os.path.join(OUT_DIR, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print("[exp3h] tổng hợp thống kê (paired Wilcoxon baseline vs reg theo ảnh)...")
    summary = {}
    for metric in SCALAR_METRICS + ["final_cos_drift"]:
        base_vals = np.array([r[f"baseline__{metric}"] for r in records])
        reg_vals = np.array([r[f"reg__{metric}"] for r in records])
        diff = reg_vals - base_vals
        w_stat, p_value = (float("nan"), float("nan"))
        if np.any(diff != 0):
            w_stat, p_value = wilcoxon(reg_vals, base_vals)
        entry = {
            "mean_baseline": float(base_vals.mean()), "mean_reg": float(reg_vals.mean()),
            "median_baseline": float(np.median(base_vals)), "median_reg": float(np.median(reg_vals)),
            "mean_diff_reg_minus_baseline": float(diff.mean()),
            "wilcoxon_p": float(p_value),
        }
        summary[metric] = entry
        print(f"[exp3h] {metric}: baseline={entry['mean_baseline']:.4g} reg={entry['mean_reg']:.4g} "
              f"diff={entry['mean_diff_reg_minus_baseline']:+.4g} (p={entry['wilcoxon_p']:.2e})")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"epsilon": EPSILON, "num_iter": NUM_ITER, "decay": DECAY, "k": K, "lam": LAM,
                    "reg_stage_modes": REG_STAGE_MODES, "n_images": n, "n_images_used": n_used,
                    "summary": summary}, f, indent=2)
    print(f"[exp3h] Xong. Kết quả: {OUT_DIR}/records.jsonl, {OUT_DIR}/summary.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
