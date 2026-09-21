"""Thí nghiệm 3G (chốt 2026-09-21, sau khi Thí nghiệm 3F cho negative/inconclusive finding — xem
docs/progress_log.md entry 2026-09-21): 3F đo cos(g_s, g_t) — cosine giữa gradient surrogate và
target — và KHÔNG xác nhận được rằng regularization làm tăng alignment đó ở n=300. 3G lùi lại 1
bước, hỏi câu hỏi hẹp hơn và rẻ hơn: **regularization thực sự làm gì với backward signal của chính
SURROGATE** (không cần target nào cả, không cần cos_sim với ai) — trước khi hỏi tiếp nó có giúp
transfer bằng cách nào.

Thiết kế (thống nhất với user 2026-09-21, ưu tiên rẻ trước — đây là bước #1 trong 4 bước đề xuất,
#2 là iteration stability, #3 là transformation consistency, để sau):

Với mỗi ảnh (population: toàn bộ ảnh có >=1 GT box trong subset 300 ảnh cố định, KHÔNG cần match
với target nào — đây là thuộc tính nội tại của surrogate, không phụ thuộc target), 1 lần forward +
backward DUY NHẤT trên surrogate (loss = attack objective "cls" y hệt mọi thí nghiệm trước, tính
trên TOÀN BỘ object của ảnh — đúng loss thật mà attack tối ưu, không phải per-object như 2A/2C/3F):

  - Đăng ký "spy" hook (chỉ ghi lại gradient, KHÔNG sửa gì, trả về grad y nguyên) lên feature map
    thô của Stage 3 và Stage 4 (trước neck, giống backward_reg_attack._forward_feats_with_reg).
  - Sau backward, có g_l RAW (l=3,4) — CHÍNH XÁC tensor mà hook thật (backward_reg_attack) sẽ nhận
    được nếu đăng ký regularize thật.
  - Vì _make_reg_hook (attacks/backward_reg_attack.py) là 1 hàm THUẦN TÚY của (grad, mask, k) —
    không phụ thuộc gì khác trong graph — tính REG offline bằng cách gọi thẳng hàm đó lên g_l RAW,
    KHÔNG cần chạy lại forward/backward lần 2. Tính cả 2 biến thể REG để so sánh:
      - "reg_clip": mode="clip" thuần (= reg_s3s4, Thí nghiệm 3A/3B) — clip đều cả tensor, không mask.
      - "reg_clip_weight": mode="clip_weight", k=3, lam=0.5 (= method hiện tại, khớp 3E/3F/3D) —
        clip + object-weighted blend.

Với mỗi (ảnh, stage, variant in {raw, reg_clip, reg_clip_weight}), tính 6 thống kê mô tả TOÀN BỘ
tensor (không phải per-pixel/per-channel — đúng granularity mà hook thật dùng để tính mean/std):
  - std: độ lệch chuẩn.
  - max_abs: |g|_max.
  - kurtosis: excess kurtosis (Fisher, 0 = phân phối chuẩn) — đo "đuôi nặng" (heavy-tailedness).
  - frac_outside_3sigma: tỷ lệ phần tử |x - mean| > 3*std (CHÍNH XÁC ngưỡng mà mode "clip" cắt).
  - top1pct_energy_frac: tỷ lệ năng lượng (tổng bình phương) nằm trong top 1% phần tử |giá trị| lớn
    nhất — đo mức độ gradient bị "thống trị" bởi vài giá trị cực trị.
  - linf_over_l2: max_abs / ||g||_2 — proxy rẻ cho độ tập trung (concentration), 1.0 = toàn bộ năng
    lượng dồn vào 1 phần tử, thấp = năng lượng trải đều.

Prediction (research_plan.md, đặt ra bởi user 2026-09-21): reg (đặc biệt reg_clip_weight) phải làm
GIẢM cả 4 đại lượng "đuôi/tập trung" (max_abs, kurtosis, top1pct_energy_frac, linf_over_l2) so với
raw, trong khi std không collapse về 0 (tín hiệu gradient tổng thể không bị triệt tiêu hoàn toàn).

Đơn vị phân tích: MỖI ẢNH 1 ROW (không phải per-object) — g_l là gradient của loss TOÀN ẢNH (tổng
qua mọi object), không có vấn đề "nhiều object cùng ảnh không độc lập" như 2A/2C/3F (nơi đơn vị là
per-object). Kiểm định: Wilcoxon signed-rank (paired, ảnh nào cũng có đúng 1 cặp raw/reg) — không
cần bootstrap/cluster-correction gì thêm vì thiết kế đã tự nhiên độc lập ở mức ảnh.

Output: outputs/experiment3g/records.jsonl (1 dòng/ảnh/stage, đủ 3 variant) + summary.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import kurtosis as _kurtosis_fn
from scipy.stats import wilcoxon

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attacks.backward_reg_attack import (  # noqa: E402
    STAGE_STRIDES, _bg_weight_from_lambda, _build_object_mask, _make_reg_hook,
)
from attacks.detection_attacks import _bbox_cls_loss  # noqa: E402
from experiments import common  # noqa: E402
from experiments.experiment1 import MODELS  # noqa: E402
from experiments.experiment2a import _gt_bboxes_net  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402
register_all_modules(init_default_scope=True)

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment3g")

STAGES = [3, 4]
K = 3
LAM = 0.5
VARIANTS = ["raw", "reg_clip", "reg_clip_weight"]
METRICS = ["std", "max_abs", "kurtosis", "frac_outside_3sigma", "top1pct_energy_frac", "linf_over_l2"]


def _tensor_stats(t, k=K):
    x = t.detach().flatten().float().cpu().numpy()
    mean = float(x.mean())
    std = float(x.std())
    max_abs = float(np.abs(x).max())
    l2 = float(np.sqrt(np.sum(x.astype(np.float64) ** 2)))
    linf_over_l2 = max_abs / (l2 + 1e-12)
    frac_outside = float(np.mean(np.abs(x - mean) > k * std)) if std > 0 else 0.0
    kurt = float(_kurtosis_fn(x, fisher=True))
    n_top = max(1, int(np.ceil(0.01 * x.size)))
    sq = (x.astype(np.float64)) ** 2
    total_energy = float(sq.sum())
    if total_energy > 0:
        top_idx = np.argpartition(sq, -n_top)[-n_top:]
        top1pct_energy_frac = float(sq[top_idx].sum() / total_energy)
    else:
        top1pct_energy_frac = 0.0
    return {
        "std": std, "max_abs": max_abs, "kurtosis": kurt,
        "frac_outside_3sigma": frac_outside, "top1pct_energy_frac": top1pct_energy_frac,
        "linf_over_l2": linf_over_l2,
    }


def _raw_and_reg_stats(model, x_leaf, ds_sample, gt_bboxes_net, gt_labels):
    data = dict(inputs=[x_leaf], data_samples=[ds_sample])
    batch = model.data_preprocessor(data, training=False)
    stage_feats = list(model.backbone(batch["inputs"]))

    spy = {}

    def make_spy(stage):
        def hook(grad):
            spy[stage] = grad.detach().clone()
            return grad
        return hook

    handles = []
    for s in STAGES:
        handles.append(stage_feats[s - 1].register_hook(make_spy(s)))

    feats = model.neck(tuple(stage_feats)) if model.with_neck else tuple(stage_feats)
    loss = _bbox_cls_loss(model, feats, gt_bboxes_net, gt_labels)
    torch.autograd.grad(loss, x_leaf)
    for h in handles:
        h.remove()

    bg_weight = _bg_weight_from_lambda(LAM)
    out = {}
    for s in STAGES:
        raw_grad = spy[s]
        mask = _build_object_mask(raw_grad.shape, gt_bboxes_net, STAGE_STRIDES[s - 1], raw_grad.device, bg_weight)
        reg_clip = _make_reg_hook("clip", None, K)(raw_grad)
        reg_clip_weight = _make_reg_hook("clip_weight", mask, K)(raw_grad)
        out[s] = {
            "raw": _tensor_stats(raw_grad),
            "reg_clip": _tensor_stats(reg_clip),
            "reg_clip_weight": _tensor_stats(reg_clip_weight),
        }
    return out


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[exp3g] loading surrogate...")
    cfg, ckpt, _family = MODELS["surrogate_r50"]
    surrogate = common.load_model(cfg, ckpt)

    print(f"[exp3g] building dataset (limit={limit})...")
    ds, n = common.build_dataset(cfg, limit=limit)
    print(f"[exp3g] {n} images (dataset total: {len(ds)})")

    records = []
    t0 = time.time()
    n_used = 0
    for i in range(n):
        sample = ds[i]
        ds_sample = sample["data_samples"]
        img_id = int(ds_sample.metainfo["img_id"])
        gt_labels_full = ds_sample.gt_instances.labels
        if gt_labels_full.numel() == 0:
            continue
        n_used += 1
        gt_labels_full = gt_labels_full.to(common.DEVICE)
        gt_bboxes_net = _gt_bboxes_net(ds_sample, common.DEVICE)

        x_clean = sample["inputs"].float().to(common.DEVICE)
        x_leaf = x_clean.detach().clone().requires_grad_(True)

        stats_by_stage = _raw_and_reg_stats(surrogate, x_leaf, ds_sample, gt_bboxes_net, gt_labels_full)
        for s in STAGES:
            row = {"image_id": img_id, "stage": s}
            for variant in VARIANTS:
                for metric in METRICS:
                    row[f"{variant}__{metric}"] = stats_by_stage[s][variant][metric]
            records.append(row)

        if (i + 1) % 20 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp3g] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút, "
                  f"{n_used} ảnh dùng được, {len(records)} record)")

    with open(os.path.join(OUT_DIR, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print("[exp3g] tổng hợp thống kê theo stage...")
    summary = {}
    for s in STAGES:
        rs = [r for r in records if r["stage"] == s]
        stage_summary = {}
        for metric in METRICS:
            raw_vals = np.array([r[f"raw__{metric}"] for r in rs])
            entry = {"mean_raw": float(raw_vals.mean()), "median_raw": float(np.median(raw_vals))}
            for variant in ("reg_clip", "reg_clip_weight"):
                reg_vals = np.array([r[f"{variant}__{metric}"] for r in rs])
                diff = reg_vals - raw_vals
                w_stat, p_value = (float("nan"), float("nan"))
                if np.any(diff != 0):
                    w_stat, p_value = wilcoxon(reg_vals, raw_vals)
                entry[f"mean_{variant}"] = float(reg_vals.mean())
                entry[f"median_{variant}"] = float(np.median(reg_vals))
                entry[f"mean_diff_{variant}_minus_raw"] = float(diff.mean())
                entry[f"median_diff_{variant}_minus_raw"] = float(np.median(diff))
                entry[f"wilcoxon_p_{variant}"] = float(p_value)
            stage_summary[metric] = entry
            print(f"[exp3g] stage{s} {metric}: raw={entry['mean_raw']:.4g} "
                  f"reg_clip={entry['mean_reg_clip']:.4g} (p={entry['wilcoxon_p_reg_clip']:.2e}) "
                  f"reg_clip_weight={entry['mean_reg_clip_weight']:.4g} (p={entry['wilcoxon_p_reg_clip_weight']:.2e})")
        summary[f"stage{s}"] = stage_summary

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"k": K, "lam": LAM, "stages": STAGES, "n_images": n, "n_images_used": n_used,
                    "metrics": METRICS, "summary": summary}, f, indent=2)
    print(f"[exp3g] Xong. Kết quả: {OUT_DIR}/records.jsonl, {OUT_DIR}/summary.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
