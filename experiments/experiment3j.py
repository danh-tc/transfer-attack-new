"""Thí nghiệm 3J (chốt 2026-09-21, sau chuỗi mechanism 3F-3I — xem docs/progress_log.md và
research_plan.md §9.2): Mask/Control Ablation — roadmap bước 2 (đã hoãn từ entry HANDOFF
2026-09-20, giờ làm lại sau khi ưu tiên mechanism-search bước 1+3 xong).

Câu hỏi: "object-conditioned" trong method v0.1 có thực sự cần đúng VỊ TRÍ ngữ nghĩa của object hay
chỉ cần "có weighting nào đó khác 1.0 đều tay" là đủ? 4 variant, cùng Stage 3-4, k=3, lam=0.5:

  1. `clip_only` — clip đều cả tensor, KHÔNG weighting (= `reg_s3s4`/`clip_s3s4`, đã có sẵn từ
     Thí nghiệm 3A/3B/3E — TÁI SỬ DỤNG từ outputs/experiment3e/results.json, không chạy lại).
  2. `real_object_mask` — method hiện tại: mask=1.0 trong GT box, beta=1/(1+lam) ngoài box (= `reg_s3s4_object_weight`/`clip_s3s4_objw`, đã có sẵn — TÁI SỬ DỤNG, không chạy lại).
  3. `uniform_weight` — MỚI: mask = hằng số c trên toàn bộ feature map, với c = mean(real_mask)
     của đúng ảnh đó (area-weighted, tự động xử lý union nếu nhiều box chồng nhau) — cùng "tổng
     ngân sách weighting" với real_object_mask, nhưng trải ĐỀU, không phân biệt object/background.
  4. `random_mask` — MỚI: mask hình dạng/diện tích giống hệt real_object_mask (cùng box size mỗi
     GT box) nhưng đặt ở vị trí NGẪU NHIÊN trong feature map (không nhất thiết trùng object thật) —
     seed cố định theo image index để tái lập được.

Câu hỏi chính: `real_object_mask` có thắng rõ `uniform_weight` và `random_mask` trên cross-family
ASR không? Nếu có → "vị trí ngữ nghĩa của object" thực sự quan trọng, không chỉ là hiệu ứng của
"có weighting nào đó". Không cần đúng thứ tự tuyệt đối `real > random ≈ uniform > clip_only`.

Code: viết lại 1 bản `_forward_feats_variant`/`_attack_with_mask_variant` cục bộ trong file này
(KHÔNG sửa attacks/backward_reg_attack.py — tái sử dụng `_make_reg_hook`/`_bg_weight_from_lambda`/
`STAGE_STRIDES` trực tiếp, chỉ thêm cách xây mask mới cho 2 variant MỚI, giữ logic attack loop y
hệt `iterative_linf_attack_reg`).

Output: outputs/experiment3j/results.json (đủ cả 4 variant, 2 tái sử dụng + 2 mới).
"""
import json
import os
import random
import sys
import time

import torch
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attacks.backward_reg_attack import STAGE_STRIDES, _bg_weight_from_lambda, _make_reg_hook  # noqa: E402
from attacks.detection_attacks import _bbox_cls_loss  # noqa: E402
from experiments import common  # noqa: E402
from experiments.experiment1 import MODELS  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402
register_all_modules(init_default_scope=True)

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment3j")

EPSILON = 8.0
NUM_ITER = 10
DECAY = 1.0
OBJECTIVE = "cls"
K = 3
LAM = 0.5
REG_STAGES = (3, 4)
NEW_VARIANTS = ["uniform_weight", "random_mask"]
REUSED_VARIANTS = {"clip_only": ("experiment3e", "clip_s3s4"), "real_object_mask": ("experiment3e", "clip_s3s4_objw")}


def _build_mask_from_boxes(shape, boxes_stage, device, bg_weight):
    _, _, h, w = shape
    mask = torch.full((1, 1, h, w), bg_weight, device=device)
    for i in range(boxes_stage.shape[0]):
        x1, y1, x2, y2 = boxes_stage[i].tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2 + 1), min(h, y2 + 1)
        if x2 > x1 and y2 > y1:
            mask[0, 0, y1:y2, x1:x2] = 1.0
    return mask


def _random_shift_boxes(boxes_stage, h, w, rng):
    shifted = boxes_stage.clone()
    for i in range(boxes_stage.shape[0]):
        x1, y1, x2, y2 = boxes_stage[i].tolist()
        bw = min(max(1, x2 - x1 + 1), w)
        bh = min(max(1, y2 - y1 + 1), h)
        new_x1 = rng.randint(0, max(0, w - bw))
        new_y1 = rng.randint(0, max(0, h - bh))
        shifted[i, 0], shifted[i, 1] = new_x1, new_y1
        shifted[i, 2], shifted[i, 3] = new_x1 + bw - 1, new_y1 + bh - 1
    return shifted


def _build_mask(mask_kind, feat_shape, gt_bboxes_net, stride, device, bg_weight, rng):
    boxes_stage = (gt_bboxes_net / stride).round().long()
    real_mask = _build_mask_from_boxes(feat_shape, boxes_stage, device, bg_weight)
    if mask_kind == "real":
        return real_mask
    if mask_kind == "uniform":
        return torch.full_like(real_mask, float(real_mask.mean()))
    if mask_kind == "random":
        _, _, h, w = feat_shape
        shifted = _random_shift_boxes(boxes_stage, h, w, rng)
        return _build_mask_from_boxes(feat_shape, shifted, device, bg_weight)
    raise ValueError(mask_kind)


def _forward_feats_variant(model, x, mask_kind, gt_bboxes_net, k, lam, rng):
    bg_weight = _bg_weight_from_lambda(lam)
    stage_feats = list(model.backbone(x))
    for s in REG_STAGES:
        feat = stage_feats[s - 1]
        mask = _build_mask(mask_kind, feat.shape, gt_bboxes_net, STAGE_STRIDES[s - 1], feat.device, bg_weight, rng)
        feat.register_hook(_make_reg_hook("clip_weight", mask, k))
    stage_feats = tuple(stage_feats)
    return model.neck(stage_feats) if model.with_neck else stage_feats


def _attack_with_mask_variant(model, x, data_sample, mask_kind, rng):
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

    alpha = EPSILON / NUM_ITER
    x_clean = x.detach().clone()
    x_adv = x.detach().clone()
    g = torch.zeros_like(x)

    for _ in range(NUM_ITER):
        x_adv.requires_grad_(True)
        data = dict(inputs=[x_adv], data_samples=[data_sample])
        batch = model.data_preprocessor(data, training=False)
        feats = _forward_feats_variant(model, batch["inputs"], mask_kind, gt_bboxes, K, LAM, rng)
        loss = _bbox_cls_loss(model, feats, gt_bboxes, gt_labels)

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad_norm = grad / (grad.abs().mean() + 1e-12)
        g = DECAY * g + grad_norm

        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.clamp(x_adv, x_clean - EPSILON, x_clean + EPSILON)
        x_adv = torch.clamp(x_adv, 0.0, 255.0)

    return x_adv.detach()


def run_one(mask_kind, ds, n, cat_ids, coco_gt, img_ids, gt_by_image, models):
    print(f"\n[exp3j] === {mask_kind} ===")
    surrogate = models["surrogate_r50"]
    results = {name: {"clean": [], "adv": []} for name in MODELS}
    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        x_clean = sample["inputs"].float().to(common.DEVICE)
        ds_sample = sample["data_samples"]
        rng = random.Random(1000 + i)

        x_adv = _attack_with_mask_variant(surrogate, x_clean, ds_sample, mask_kind, rng)

        for name, model in models.items():
            results[name]["clean"].extend(common.predict_coco_format(model, x_clean, ds_sample, cat_ids))
            results[name]["adv"].extend(common.predict_coco_format(model, x_adv, ds_sample, cat_ids))

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp3j][{mask_kind}] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút)")

    summary = {}
    for name, (_, _, family) in MODELS.items():
        clean_map = common.compute_map(coco_gt, results[name]["clean"], img_ids)
        adv_map = common.compute_map(coco_gt, results[name]["adv"], img_ids)
        clean_by_img = common.group_by_image(results[name]["clean"])
        adv_by_img = common.group_by_image(results[name]["adv"])
        asr = common.compute_asr(gt_by_image, clean_by_img, adv_by_img)
        summary[name] = common.summarize_model_result(family, clean_map, adv_map, asr)
        print(f"[exp3j][{mask_kind}] {name} ({family}): clean AP50={clean_map['AP50']:.4f} -> "
              f"adv AP50={adv_map['AP50']:.4f} | ASR={asr['ASR']:.4f} "
              f"({asr['evaded_objects']}/{asr['clean_correct_objects']})")

    return summary


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)

    reused = {}
    for key, (exp_dir, setting_key) in REUSED_VARIANTS.items():
        with open(os.path.join(REPO_ROOT, "outputs", exp_dir, "results.json")) as f:
            data = json.load(f)
        reused[key] = data["settings"][setting_key]
    print(f"[exp3j] tái sử dụng {list(REUSED_VARIANTS.keys())} từ {[v[0] for v in REUSED_VARIANTS.values()]}")

    print("[exp3j] loading models...")
    models = {name: common.load_model(cfg, ckpt) for name, (cfg, ckpt, _f) in MODELS.items()}

    print(f"[exp3j] building dataset (limit={limit})...")
    ds, n = common.build_dataset(MODELS["surrogate_r50"][0], limit=limit)
    cat_ids = ds.cat_ids
    print(f"[exp3j] {n} images (dataset total: {len(ds)})")

    coco_gt = COCO(common.SUBSET_ANNO)
    img_ids = [ds[i]["data_samples"].metainfo["img_id"] for i in range(n)]
    gt_by_image = common.load_gt_by_image(coco_gt, img_ids)

    all_results = dict(reused)
    for mask_kind_key, mask_kind in [("uniform_weight", "uniform"), ("random_mask", "random")]:
        all_results[mask_kind_key] = run_one(mask_kind, ds, n, cat_ids, coco_gt, img_ids, gt_by_image, models)
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump({
                "epsilon": EPSILON, "num_iter": NUM_ITER, "decay": DECAY, "objective": OBJECTIVE,
                "k": K, "lam": LAM, "reg_stages": list(REG_STAGES), "n_images": n,
                "reused_from": {k: f"outputs/{v[0]}/results.json#{v[1]}" for k, v in REUSED_VARIANTS.items()},
                "settings": all_results,
            }, f, indent=2)

    print("\n[exp3j] Bảng theo mask variant:")
    order = ["clip_only", "uniform_weight", "random_mask", "real_object_mask"]
    for key in order:
        s = all_results[key]
        cross_avg = (s["target_convnext_t"]["ASR"] + s["target_swin_t"]["ASR"]) / 2
        gap = s["target_r101"]["ASR"] - cross_avg
        print(f"  {key:16s} ConvNeXt={s['target_convnext_t']['ASR']:.4f} Swin={s['target_swin_t']['ASR']:.4f} "
              f"CrossAvg={cross_avg:.4f} | R101={s['target_r101']['ASR']:.4f} | "
              f"WhiteBox={s['surrogate_r50']['ASR']:.4f} | TransferGap={gap:+.4f}")

    print(f"\n[exp3j] Xong. Kết quả: {OUT_DIR}/results.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
