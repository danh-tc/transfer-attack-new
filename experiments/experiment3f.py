"""Mechanism Validation (chốt 2026-09-20, sau Component Ablation — xem docs/progress_log.md): đo
lại metric đã khóa ở RQ2 — cos(g_s, g_t) — TRƯỚC và SAU khi áp backward regularization (Stage 3-4,
k=3, lambda=0.5, mode="clip_weight" — candidate tốt nhất từ Ablation A+B) lên gradient của
SURROGATE khi tính loss. Gradient của TARGET KHÔNG regularize (luôn raw) — đúng threat model
single-surrogate: chỉ kiểm soát được surrogate, target luôn black-box, không bao giờ có quyền chỉnh
sửa nội bộ tính toán của nó (kể cả trong thực nghiệm chẩn đoán này).

Hypothesis (cụ thể hóa RQ2 -> RQ3): regularization -> cross-family gradient alignment tăng ->
transfer ASR tăng. Nếu cos(g_s_reg, g_t) > cos(g_s_raw, g_t) rõ rệt trên ConvNeXt/Swin nhưng KHÔNG
cần tăng (hoặc tăng ít hơn nhiều) trên R101, đây là bằng chứng cơ chế mạnh nhất nối RQ2 -> RQ3:
    regularization -> backward alignment cross-family tăng -> transfer ASR cross-family tăng
(đã biết từ Ablation A/B/Component Ablation — ASR cross-family tăng, R101 gần như không đổi/giảm nhẹ).

Population: TÁI SỬ DỤNG đúng object đã dùng ở Thí nghiệm 2A (đọc từ
outputs/experiment2a/records.jsonl) — cùng ảnh sạch, cùng object clean-correct theo từng target —
để so sánh trực tiếp được với cos_sim đã khóa ở RQ2 (2A: mean cos_sim theo target = 0.0878-0.1397
R101, 0.0524-0.0905 ConvNeXt, 0.0282-0.0491 Swin — xem đối chiếu cos_before ở đây với các số đó).

Output: outputs/experiment3f/records.jsonl (per-object, cos_before/cos_after) + summary.json.
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attacks.backward_reg_attack import _forward_feats_with_reg  # noqa: E402
from experiments import common  # noqa: E402
from experiments.experiment1 import MODELS  # noqa: E402
from experiments.experiment2a import OUT_DIR as EXP2A_DIR  # noqa: E402
from experiments.experiment2a import _cos_sim, _gt_bboxes_net, _object_grads  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402
register_all_modules(init_default_scope=True)

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment3f")

STAGE_MODES = {3: "clip_weight", 4: "clip_weight"}
K = 3
LAM = 0.5


def _load_2a_population():
    records_path = os.path.join(EXP2A_DIR, "records.jsonl")
    by_image = defaultdict(lambda: defaultdict(dict))
    with open(records_path) as f:
        for line in f:
            r = json.loads(line)
            by_image[r["image_id"]][r["target"]][r["object_idx"]] = r["evaded"]
    print(f"[exp3f] đọc {len(by_image)} ảnh từ {records_path}")
    return by_image


def _forward_feats_variant(model, x_leaf, ds_sample, regularize, gt_bboxes_net=None):
    data = dict(inputs=[x_leaf], data_samples=[ds_sample])
    batch = model.data_preprocessor(data, training=False)
    if regularize:
        return _forward_feats_with_reg(model, batch["inputs"], stage_modes=STAGE_MODES,
                                        gt_bboxes_net=gt_bboxes_net, k=K, lam=LAM)
    return model.extract_feat(batch["inputs"])


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    by_image = _load_2a_population()

    print("[exp3f] loading models...")
    models = {name: common.load_model(cfg, ckpt) for name, (cfg, ckpt, _f) in MODELS.items()}
    surrogate = models["surrogate_r50"]
    targets = {name: m for name, m in models.items() if name != "surrogate_r50"}

    print(f"[exp3f] building dataset (limit={limit})...")
    ds, n = common.build_dataset(MODELS["surrogate_r50"][0], limit=limit)
    print(f"[exp3f] {n} images (dataset total: {len(ds)})")

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
        gt_labels_full = ds_sample.gt_instances.labels.to(common.DEVICE)
        gt_bboxes_net = _gt_bboxes_net(ds_sample, common.DEVICE)

        x_leaf = x_clean.detach().clone().requires_grad_(True)

        feats_s_raw = _forward_feats_variant(surrogate, x_leaf, ds_sample, regularize=False)
        grads_s_raw = _object_grads(surrogate, x_leaf, feats_s_raw, gt_bboxes_net, gt_labels_full, union_needed)
        del feats_s_raw

        feats_s_reg = _forward_feats_variant(surrogate, x_leaf, ds_sample, regularize=True,
                                              gt_bboxes_net=gt_bboxes_net)
        grads_s_reg = _object_grads(surrogate, x_leaf, feats_s_reg, gt_bboxes_net, gt_labels_full, union_needed)
        del feats_s_reg

        for t_name, t_model in targets.items():
            t_obj_evaded = needed_by_target.get(t_name, {})
            t_needed = sorted(t_obj_evaded.keys())
            if not t_needed:
                continue
            family = MODELS[t_name][2]
            feats_t = _forward_feats_variant(t_model, x_leaf, ds_sample, regularize=False)
            grads_t = _object_grads(t_model, x_leaf, feats_t, gt_bboxes_net, gt_labels_full, t_needed)
            del feats_t

            for j in t_needed:
                cos_before = _cos_sim(grads_s_raw[j], grads_t[j])
                cos_after = _cos_sim(grads_s_reg[j], grads_t[j])
                records.append({
                    "image_id": img_id, "target": t_name, "family": family, "object_idx": j,
                    "cos_before": cos_before, "cos_after": cos_after,
                    "evaded": bool(t_obj_evaded[j]),
                })

        if (i + 1) % 20 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp3f] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút, "
                  f"{n_images_used} ảnh dùng được, {len(records)} record)")

    with open(os.path.join(OUT_DIR, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print("[exp3f] tổng hợp mean cos_before/cos_after theo target...")
    summary = {}
    for t_name in targets:
        rs = [r for r in records if r["target"] == t_name]
        if not rs:
            continue
        before = np.array([r["cos_before"] for r in rs])
        after = np.array([r["cos_after"] for r in rs])
        summary[t_name] = {
            "family": rs[0]["family"], "n": len(rs),
            "mean_cos_before": float(before.mean()), "mean_cos_after": float(after.mean()),
            "delta": float(after.mean() - before.mean()),
        }
        print(f"[exp3f] {t_name}: mean_cos_before={summary[t_name]['mean_cos_before']:.4f} "
              f"mean_cos_after={summary[t_name]['mean_cos_after']:.4f} delta={summary[t_name]['delta']:+.4f}")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"k": K, "lam": LAM, "stage_modes": STAGE_MODES, "n_images": n,
                    "n_images_used": n_images_used, "summary": summary}, f, indent=2)
    print(f"[exp3f] Xong. Kết quả: {OUT_DIR}/records.jsonl, {OUT_DIR}/summary.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
