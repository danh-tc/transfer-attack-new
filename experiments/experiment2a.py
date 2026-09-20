"""Thí nghiệm 2A (research_plan.md §7.A, chốt 2026-09-20): gradient alignment giữa surrogate và
từng target, đo PER-OBJECT (không phải chỉ tương quan ở mức 3 target — quá ít điểm để kết luận
thống kê), tương quan với evaded/not-evaded.

Với mỗi object GT đã được 1 target detect đúng ở điều kiện sạch (clean-correct — cùng định nghĩa
"clean_correct_objects" dùng để tính ASR ở experiments/common.compute_asr), tính:

  - g_s: gradient của cross-entropy loss (giống objective="cls" của attack, xem
    attacks/detection_attacks.py._bbox_cls_loss) của SURROGATE, tại đúng RoI GT của object đó,
    trên ảnh SẠCH — KHÔNG chạy attack lúc tính gradient này, chỉ 1 lần forward+backward.
  - g_t: gradient tương tự nhưng dùng backbone/head của chính TARGET model đó (không phải
    surrogate) — CÙNG ảnh sạch, CÙNG RoI GT.
  - cos(g_s, g_t): cosine similarity 2 gradient trong không gian pixel input x (cùng shape bất kể
    kiến trúc backbone khác nhau, nên so sánh được).
  - evaded: object này có bị attack (crafted trên surrogate, config y hệt Thí nghiệm 1/1B —
    epsilon=8.0, num_iter=10, decay=1.0, objective="cls") làm target đó né tránh hay không (đúng
    nhãn ở ảnh sạch, mất đúng nhãn ở ảnh adversarial theo IoU/score threshold trong common.py).

QUAN TRỌNG: gt_instances.bboxes ở tọa độ ẢNH GỐC (bug đã fix 2026-09-20, xem docs/progress_log.md)
— phải nhân scale_factor để ra tọa độ mạng trước khi bbox2roi (giống hệt cách attacks/*.py đã fix).
Việc match evaded/not-evaded dùng gt_list ở tọa độ GỐC (đúng, khớp predict_coco_format rescale=True).

Output: outputs/experiment2a/records.jsonl (1 dòng/object-target) + outputs/experiment2a/summary.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.structures.bbox import bbox2roi

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attacks.detection_attacks import iterative_linf_attack  # noqa: E402
from experiments import common  # noqa: E402
from experiments.experiment1 import MODELS  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402
register_all_modules(init_default_scope=True)

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment2a")

# Config attack — y hệt Thí nghiệm 1/1B để nhãn evaded/not-evaded so sánh được với ASR đã báo cáo.
EPSILON = 8.0
NUM_ITER = 10
DECAY = 1.0
OBJECTIVE = "cls"


def _gt_bboxes_net(ds_sample, device):
    """gt_bboxes ở tọa độ ảnh gốc -> scale theo scale_factor để khớp feats (tính từ ảnh đã resize)."""
    gt_bboxes = ds_sample.gt_instances.bboxes
    if hasattr(gt_bboxes, "tensor"):
        gt_bboxes = gt_bboxes.tensor
    gt_bboxes = gt_bboxes.to(device)
    w_scale, h_scale = ds_sample.metainfo["scale_factor"]
    scale = torch.tensor([w_scale, h_scale, w_scale, h_scale], device=device, dtype=gt_bboxes.dtype)
    return gt_bboxes * scale


def _gt_list_orig(ds_sample, cat_ids):
    """gt_list ở tọa độ ảnh GỐC, xywh — dùng cho common.match_greedy (khớp predict_coco_format
    rescale=True). Thứ tự index giữ nguyên như gt_instances để khớp trực tiếp với object_idx dùng
    khi tính gradient (bbox2roi trên cùng gt_bboxes/gt_labels, cùng thứ tự)."""
    gt_bboxes = ds_sample.gt_instances.bboxes
    if hasattr(gt_bboxes, "tensor"):
        gt_bboxes = gt_bboxes.tensor
    gt_labels = ds_sample.gt_instances.labels
    out = []
    for box, label in zip(gt_bboxes.tolist(), gt_labels.tolist()):
        x1, y1, x2, y2 = box
        out.append({"category_id": int(cat_ids[label]), "bbox": [x1, y1, x2 - x1, y2 - y1]})
    return out


def _forward_feats(model, x, ds_sample):
    data = dict(inputs=[x], data_samples=[ds_sample])
    batch = model.data_preprocessor(data, training=False)
    return model.extract_feat(batch["inputs"])


def _object_grads(model, x, feats, gt_bboxes_net, gt_labels, obj_indices):
    """Gradient riêng cho từng object trong obj_indices, dùng feats đã forward sẵn (retain_graph
    để tái dùng qua nhiều lần backward). x phải là leaf tensor requires_grad=True đã dùng để tính
    feats. Mỗi object 1 forward roi_head + 1 backward riêng — KHÔNG cộng dồn loss nhiều object lại
    (mới cô lập được gradient của đúng 1 object, đúng yêu cầu per-object của research_plan §7.A)."""
    if not obj_indices:
        return {}
    rois_all = bbox2roi([gt_bboxes_net])
    grads = {}
    for j in obj_indices:
        roi_j = rois_all[j:j + 1]
        label_j = gt_labels[j:j + 1]
        bbox_feats = model.roi_head.bbox_roi_extractor(
            feats[: model.roi_head.bbox_roi_extractor.num_inputs], roi_j)
        cls_score, _ = model.roi_head.bbox_head(bbox_feats)
        loss = F.cross_entropy(cls_score, label_j)
        grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
        grads[j] = grad.detach().flatten()
    return grads


def _cos_sim(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[exp2a] loading models...")
    models = {name: common.load_model(cfg, ckpt) for name, (cfg, ckpt, _f) in MODELS.items()}
    surrogate = models["surrogate_r50"]
    targets = {name: m for name, m in models.items() if name != "surrogate_r50"}

    print(f"[exp2a] building dataset (limit={limit})...")
    ds, n = common.build_dataset(MODELS["surrogate_r50"][0], limit=limit)
    cat_ids = ds.cat_ids
    print(f"[exp2a] {n} images (dataset total: {len(ds)})")

    records = []
    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        x_clean = sample["inputs"].float().to(common.DEVICE)
        ds_sample = sample["data_samples"]
        img_id = int(ds_sample.metainfo["img_id"])

        gt_labels_full = ds_sample.gt_instances.labels
        if gt_labels_full.numel() == 0:
            continue
        gt_labels_full = gt_labels_full.to(common.DEVICE)
        gt_bboxes_net = _gt_bboxes_net(ds_sample, common.DEVICE)
        gt_list_orig = _gt_list_orig(ds_sample, cat_ids)
        n_obj = gt_labels_full.numel()

        # Attack trên surrogate — config y hệt Thí nghiệm 1/1B (đã fix bug tọa độ RoI).
        x_adv = iterative_linf_attack(surrogate, x_clean, ds_sample, epsilon=EPSILON,
                                       num_iter=NUM_ITER, decay=DECAY, objective=OBJECTIVE)

        # Gradient surrogate cho TẤT CẢ object trong ảnh (dùng lại chung cho mọi target).
        x_leaf = x_clean.detach().clone().requires_grad_(True)
        feats_s = _forward_feats(surrogate, x_leaf, ds_sample)
        grads_s = _object_grads(surrogate, x_leaf, feats_s, gt_bboxes_net, gt_labels_full, list(range(n_obj)))
        del feats_s

        for t_name, t_model in targets.items():
            family = MODELS[t_name][2]
            clean_preds = common.predict_coco_format(t_model, x_clean, ds_sample, cat_ids)
            adv_preds = common.predict_coco_format(t_model, x_adv, ds_sample, cat_ids)
            clean_matched = common.match_greedy(gt_list_orig, clean_preds, common.IOU_THR, common.SCORE_THR)
            adv_matched = common.match_greedy(gt_list_orig, adv_preds, common.IOU_THR, common.SCORE_THR)
            if not clean_matched:
                continue

            feats_t = _forward_feats(t_model, x_leaf, ds_sample)
            grads_t = _object_grads(t_model, x_leaf, feats_t, gt_bboxes_net, gt_labels_full, sorted(clean_matched))
            del feats_t

            for j in sorted(clean_matched):
                cos_sim = _cos_sim(grads_s[j], grads_t[j])
                records.append({
                    "image_id": img_id, "target": t_name, "family": family,
                    "object_idx": j, "category_id": int(cat_ids[gt_labels_full[j].item()]),
                    "cos_sim": cos_sim, "evaded": bool(j not in adv_matched),
                })

        if (i + 1) % 10 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp2a] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút, {len(records)} record)")

    with open(os.path.join(OUT_DIR, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary = {}
    for t_name in targets:
        rs = [r for r in records if r["target"] == t_name]
        if not rs:
            continue
        cos_evaded = [r["cos_sim"] for r in rs if r["evaded"]]
        cos_not = [r["cos_sim"] for r in rs if not r["evaded"]]
        all_cos = np.array([r["cos_sim"] for r in rs])
        all_evaded = np.array([1.0 if r["evaded"] else 0.0 for r in rs])
        corr = float(np.corrcoef(all_cos, all_evaded)[0, 1]) if len(rs) > 1 and all_cos.std() > 0 else float("nan")
        summary[t_name] = {
            "family": rs[0]["family"],
            "n_objects": len(rs),
            "n_evaded": len(cos_evaded),
            "n_not_evaded": len(cos_not),
            "mean_cos_sim_evaded": float(np.mean(cos_evaded)) if cos_evaded else float("nan"),
            "mean_cos_sim_not_evaded": float(np.mean(cos_not)) if cos_not else float("nan"),
            "std_cos_sim_evaded": float(np.std(cos_evaded)) if cos_evaded else float("nan"),
            "std_cos_sim_not_evaded": float(np.std(cos_not)) if cos_not else float("nan"),
            "point_biserial_corr_cos_vs_evaded": corr,
            "mean_cos_sim_overall": float(np.mean(all_cos)),
        }
        print(f"[exp2a] {t_name}: n={len(rs)} mean_cos(evaded)={summary[t_name]['mean_cos_sim_evaded']:.4f} "
              f"mean_cos(not_evaded)={summary[t_name]['mean_cos_sim_not_evaded']:.4f} corr={corr:.4f}")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "epsilon": EPSILON, "num_iter": NUM_ITER, "decay": DECAY, "objective": OBJECTIVE,
            "score_thr": common.SCORE_THR, "iou_thr": common.IOU_THR, "n_images": n,
            "summary": summary,
        }, f, indent=2)
    print(f"[exp2a] Xong. Kết quả: {OUT_DIR}/records.jsonl, {OUT_DIR}/summary.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
