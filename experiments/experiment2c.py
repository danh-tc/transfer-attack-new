"""Thí nghiệm 2C (chốt 2026-09-20, sau negative finding ở 2B/2B.1 — xem docs/progress_log.md):
raw forward feature similarity (CKA) KHÔNG giải thích được transfer success — CKA cho thứ tự
R101 > Swin > ConvNeXt (2B) và thậm chí CKA_evaded < CKA_not_evaded ở deep stage (2B.1), ngược
hoàn toàn hướng hypothesis. Trong khi đó gradient alignment ở RoI cuối (2A, dùng loss classification
tại bbox_head) lại đi ĐÚNG hướng: cos_sim cao hơn ở nhóm evaded, và mean cos_sim theo target đúng
thứ tự R101 > ConvNeXt > Swin khớp ASR.

Câu hỏi 2C: khác biệt đó bắt đầu từ đâu trong mạng? Tính "backward sensitivity" (feature-gradient)
ở TỪNG STAGE backbone — không phải chỉ ở RoI head cuối như 2A — rồi so cos similarity giữa
surrogate và target, xem có STAGE NÀO mà thứ tự khớp đúng R101 > ConvNeXt > Swin không (và nếu có,
từ stage nào bắt đầu).

Định nghĩa "feature-gradient" ở đây: với mỗi object, tại mỗi stage l, pool feature (RoIAlign 7x7 +
global average, giống 2B) ra vector f^l(x) (C chiều). Lấy scalar = ||f^l(x)||_2^2 (tổng bình
phương — lựa chọn tự nhiên không cần projection ngẫu nhiên tùy tiện, không cần nhãn, đo "độ nhạy
của năng lượng activation" tại stage đó theo input). Backprop scalar này về x (ẢNH SẠCH, KHÔNG chạy
attack — 2C giống 2B, chỉ cần độ nhạy/gradient CỦA MODEL SẠCH, không phải adversarial example) ra
gradient g^l(x) trong không gian pixel input — CÙNG shape cho mọi model/stage nên so cos_sim được
trực tiếp, đúng kiểu đã dùng ở 2A nhưng áp dụng cho từng stage trung gian thay vì chỉ RoI head cuối.

Population: TÁI SỬ DỤNG đúng object đã dùng ở 2A/2B (đọc từ outputs/experiment2a/records.jsonl).

Output: outputs/experiment2c/records.jsonl (per-object per-stage) + summary.json (mean cos_sim
theo stage x target, so trực tiếp với thứ tự ASR đã biết).
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from mmcv.ops import roi_align
from mmdet.structures.bbox import bbox2roi

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from experiments import common  # noqa: E402
from experiments.experiment1 import MODELS  # noqa: E402
from experiments.experiment2a import OUT_DIR as EXP2A_DIR  # noqa: E402
from experiments.experiment2a import _gt_bboxes_net  # noqa: E402
from experiments.experiment2b import STAGE_STRIDES, ROI_OUTPUT_SIZE  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402
register_all_modules(init_default_scope=True)

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment2c")


def _load_2a_population():
    """Đọc outputs/experiment2a/records.jsonl -> {img_id: {target_name: {object_idx: evaded}}}."""
    records_path = os.path.join(EXP2A_DIR, "records.jsonl")
    by_image = defaultdict(lambda: defaultdict(dict))
    with open(records_path) as f:
        for line in f:
            r = json.loads(line)
            by_image[r["image_id"]][r["target"]][r["object_idx"]] = r["evaded"]
    print(f"[exp2c] đọc {len(by_image)} ảnh có object từ {records_path}")
    return by_image


def _forward_stages_grad(model, x, ds_sample):
    """Giống experiments/experiment2b._forward_stages nhưng KHÔNG no_grad — cần giữ graph để
    backprop feature-gradient. x phải là leaf tensor requires_grad=True."""
    data = dict(inputs=[x], data_samples=[ds_sample])
    batch = model.data_preprocessor(data, training=False)
    return model.backbone(batch["inputs"])


def _object_stage_grad(x, feat, roi_j, stride):
    """Gradient của scalar=||pooled_feature||_2^2 tại 1 object, 1 stage, đối với x. retain_graph để
    tái dùng qua nhiều object/stage (cùng 1 forward feats)."""
    pooled = roi_align(feat, roi_j, (ROI_OUTPUT_SIZE, ROI_OUTPUT_SIZE), 1.0 / stride, 0, "avg", True)
    vec = pooled.mean(dim=(2, 3)).flatten()
    scalar = (vec ** 2).sum()
    grad = torch.autograd.grad(scalar, x, retain_graph=True)[0]
    return grad.detach().flatten()


def _cos_sim(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    by_image = _load_2a_population()

    print("[exp2c] loading models...")
    models = {name: common.load_model(cfg, ckpt) for name, (cfg, ckpt, _f) in MODELS.items()}
    surrogate = models["surrogate_r50"]
    targets = {name: m for name, m in models.items() if name != "surrogate_r50"}

    print(f"[exp2c] building dataset (limit={limit})...")
    ds, n = common.build_dataset(MODELS["surrogate_r50"][0], limit=limit)
    print(f"[exp2c] {n} images (dataset total: {len(ds)})")

    n_stages = len(STAGE_STRIDES)
    records = []
    t0 = time.time()
    n_images_used = 0
    for i in range(n):
        sample = ds[i]
        ds_sample = sample["data_samples"]
        img_id = int(ds_sample.metainfo["img_id"])
        needed_by_target = by_image.get(img_id)
        if not needed_by_target:
            continue

        union_needed = sorted(set().union(*[set(d.keys()) for d in needed_by_target.values()]))
        if not union_needed:
            continue
        n_images_used += 1

        x_clean = sample["inputs"].float().to(common.DEVICE)
        gt_bboxes_net = _gt_bboxes_net(ds_sample, common.DEVICE)
        rois_all = bbox2roi([gt_bboxes_net])

        x_leaf = x_clean.detach().clone().requires_grad_(True)
        feats_s = _forward_stages_grad(surrogate, x_leaf, ds_sample)
        surrogate_grads = {s: {} for s in range(n_stages)}
        for s in range(n_stages):
            for j in union_needed:
                surrogate_grads[s][j] = _object_stage_grad(x_leaf, feats_s[s], rois_all[j:j + 1], STAGE_STRIDES[s])
        del feats_s

        for t_name, t_model in targets.items():
            t_obj_evaded = needed_by_target.get(t_name, {})
            t_needed = sorted(t_obj_evaded.keys())
            if not t_needed:
                continue
            family = MODELS[t_name][2]
            feats_t = _forward_stages_grad(t_model, x_leaf, ds_sample)
            for s in range(n_stages):
                for j in t_needed:
                    g_t = _object_stage_grad(x_leaf, feats_t[s], rois_all[j:j + 1], STAGE_STRIDES[s])
                    cos_sim = _cos_sim(surrogate_grads[s][j], g_t)
                    records.append({
                        "image_id": img_id, "target": t_name, "family": family,
                        "stage": s + 1, "object_idx": j, "cos_sim": cos_sim,
                        "evaded": bool(t_obj_evaded[j]),
                    })
            del feats_t

        if (i + 1) % 20 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp2c] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút, "
                  f"{n_images_used} ảnh dùng được, {len(records)} record)")

    with open(os.path.join(OUT_DIR, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print("[exp2c] tổng hợp mean cos_sim theo stage x target...")
    summary = {}
    for s in range(1, n_stages + 1):
        summary[f"stage{s}"] = {}
        for t_name in targets:
            rs = [r for r in records if r["target"] == t_name and r["stage"] == s]
            if not rs:
                continue
            cos_all = np.array([r["cos_sim"] for r in rs])
            cos_ev = np.array([r["cos_sim"] for r in rs if r["evaded"]])
            cos_not = np.array([r["cos_sim"] for r in rs if not r["evaded"]])
            summary[f"stage{s}"][t_name] = {
                "family": rs[0]["family"], "n": len(rs),
                "mean_cos_sim": float(cos_all.mean()),
                "mean_cos_sim_evaded": float(cos_ev.mean()) if len(cos_ev) else float("nan"),
                "mean_cos_sim_not_evaded": float(cos_not.mean()) if len(cos_not) else float("nan"),
            }
            print(f"[exp2c] stage{s} {t_name}: mean_cos_sim={summary[f'stage{s}'][t_name]['mean_cos_sim']:.4f} "
                  f"(evaded={summary[f'stage{s}'][t_name]['mean_cos_sim_evaded']:.4f} "
                  f"not_evaded={summary[f'stage{s}'][t_name]['mean_cos_sim_not_evaded']:.4f}, n={len(rs)})")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"n_images": n, "n_images_used": n_images_used, "summary": summary}, f, indent=2)
    print(f"[exp2c] Xong. Kết quả: {OUT_DIR}/records.jsonl, {OUT_DIR}/summary.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
