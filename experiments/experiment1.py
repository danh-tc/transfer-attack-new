"""Thí nghiệm 1 (research_plan.md §6, §13): xác minh transfer gap xuyên họ feature-extractor.

Surrogate: Mask R-CNN + ResNet-50 (white-box, dùng để craft adversarial example).
Target: Mask R-CNN + {ResNet-101 (same-family), ConvNeXt-Tiny (cross-CNN), Swin-Tiny (CNN->Transformer)}.
Tấn công: MI-FGSM (momentum, L_inf, epsilon=8/255) trên objective kiểu DAG — xem attacks/mi_fgsm.py.
Dataset: subset 1000 ảnh COCO val2017 cố định (configs/coco_val2017_subset_1000.json).

Với mỗi ảnh: craft đúng 1 adversarial image trên surrogate, dùng lại y hệt ảnh đó (không craft
riêng cho từng target — đúng threat model black-box single-surrogate của research_plan.md §2) để
suy luận (predict) trên cả 4 model, điều kiện clean và adversarial.

Output: docs/progress_log.md ghi số liệu tóm tắt; file đầy đủ (không commit) tại
outputs/experiment1/results.json.

Seed: chỉ có tính ngẫu nhiên ở việc chọn subset ảnh (đã cố định, xem scripts/download_dataset.sh).
Bản thân MI-FGSM ở đây là deterministic (không có bước random nào: không random start, không
random augmentation) nên không cần seed riêng cho attack.
"""
import json
import os
import sys
import time

from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attacks.mi_fgsm import mi_fgsm_attack  # noqa: E402
from experiments import common  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402
register_all_modules(init_default_scope=True)

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment1")

# Threat model — research_plan.md §6.4
EPSILON = 8.0       # L_inf, thang pixel [0,255] (= 8/255 trên ảnh [0,1])
NUM_ITER = 10
DECAY = 1.0

MODELS = {
    "surrogate_r50": (
        f"{common.MMDET_DIR}/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py",
        f"{REPO_ROOT}/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth",
        "surrogate",
    ),
    "target_r101": (
        f"{common.MMDET_DIR}/configs/mask_rcnn/mask-rcnn_r101_fpn_1x_coco.py",
        f"{REPO_ROOT}/checkpoints/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth",
        "same-family",
    ),
    "target_convnext_t": (
        f"{common.MMDET_DIR}/configs/convnext/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py",
        f"{REPO_ROOT}/checkpoints/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco_20220426_154953-050731f4.pth",
        "cross-cnn-family",
    ),
    "target_swin_t": (
        f"{common.MMDET_DIR}/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py",
        f"{REPO_ROOT}/checkpoints/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth",
        "cnn-to-transformer",
    ),
}


def load_models():
    return {name: common.load_model(cfg, ckpt) for name, (cfg, ckpt, _family) in MODELS.items()}


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[exp1] loading models...")
    models = load_models()
    print(f"[exp1] building dataset (limit={limit})...")
    ds, n = common.build_dataset(MODELS["surrogate_r50"][0], limit=limit)
    cat_ids = ds.cat_ids
    print(f"[exp1] {n} images (dataset total: {len(ds)})")

    coco_gt = COCO(common.SUBSET_ANNO)
    img_ids = [ds[i]["data_samples"].metainfo["img_id"] for i in range(n)]

    results = {name: {"clean": [], "adv": []} for name in MODELS}

    surrogate = models["surrogate_r50"]
    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        x_clean = sample["inputs"].float().to(common.DEVICE)
        ds_sample = sample["data_samples"]

        x_adv = mi_fgsm_attack(surrogate, x_clean, ds_sample, epsilon=EPSILON, num_iter=NUM_ITER, decay=DECAY)

        for name, model in models.items():
            results[name]["clean"].extend(common.predict_coco_format(model, x_clean, ds_sample, cat_ids))
            results[name]["adv"].extend(common.predict_coco_format(model, x_adv, ds_sample, cat_ids))

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp1] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút)")

    with open(os.path.join(OUT_DIR, "raw_predictions.json"), "w") as f:
        json.dump(results, f)

    gt_by_image = common.load_gt_by_image(coco_gt, img_ids)

    summary = {}
    for name, (_, _, family) in MODELS.items():
        clean_map = common.compute_map(coco_gt, results[name]["clean"], img_ids)
        adv_map = common.compute_map(coco_gt, results[name]["adv"], img_ids)
        clean_by_img = common.group_by_image(results[name]["clean"])
        adv_by_img = common.group_by_image(results[name]["adv"])
        asr = common.compute_asr(gt_by_image, clean_by_img, adv_by_img)
        summary[name] = {
            "family": family,
            "clean_AP": clean_map["AP"], "clean_AP50": clean_map["AP50"],
            "adv_AP": adv_map["AP"], "adv_AP50": adv_map["AP50"],
            "delta_AP50": adv_map["AP50"] - clean_map["AP50"],
            **asr,
        }
        print(f"[exp1] {name} ({family}): clean AP50={clean_map['AP50']:.4f} -> adv AP50={adv_map['AP50']:.4f} "
              f"| ASR={asr['ASR']:.4f} ({asr['evaded_objects']}/{asr['clean_correct_objects']})")

    config = {
        "epsilon": EPSILON, "num_iter": NUM_ITER, "decay": DECAY,
        "score_thr": common.SCORE_THR, "iou_thr": common.IOU_THR, "n_images": n,
        "subset_file": "configs/coco_val2017_subset_1000.json",
        "attack": "MI-FGSM (momentum L_inf) trên objective kiểu DAG (RoI của GT box) — attacks/mi_fgsm.py",
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"config": config, "summary": summary}, f, indent=2)
    print(f"[exp1] Xong. Kết quả: {OUT_DIR}/results.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
