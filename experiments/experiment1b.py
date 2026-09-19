"""Thí nghiệm 1B (docs/progress_log.md 2026-09-19, đề xuất sau khi có kết quả Thí nghiệm 1):
kiểm tra 3 confound trước khi coi transfer gap ở Thí nghiệm 1 là do KIẾN TRÚC backbone gây ra.

Claim Thí nghiệm 1 chỉ dừng ở: "cross-family transfer gap tồn tại rõ ràng trong 1 thiết lập cụ
thể (MI-FGSM, objective classification-only, 4 model cố định)". Thí nghiệm 1B kiểm tra claim đó
có đứng vững khi thay đổi các biến sau không:

  Run B — model-strength confound: mở rộng model zoo trong CÙNG họ ResNet/ResNeXt (thêm
    ResNeXt-101-32x4d và ResNet-50 train 3x+multi-scale) để tách "backbone family" khỏi "model
    mạnh/yếu" — nếu clean AP cao hơn luôn đi cùng ASR thấp hơn NGAY CẢ trong cùng 1 họ, kiến trúc
    backbone chưa chắc là nguyên nhân chính.
  Run C — attack robustness: BIM/I-FGSM (bỏ momentum) thay MI-FGSM, trên 4 model gốc — xem thứ tự
    same-family > cross-CNN > CNN->Transformer có giữ nguyên không khi đổi attack.
  Run D — objective robustness: thêm loss hồi quy bbox vào objective (không chỉ classification).
  Run E — augmentation-based transfer: DI-FGSM (input diversity) — baseline "mạnh hơn" theo
    research_plan.md §11.

Tất cả dùng chung 1000 ảnh subset, epsilon/num_iter giống Thí nghiệm 1 (xem experiment1.py).
"""
import json
import os
import sys
import time

from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attacks.detection_attacks import iterative_linf_attack  # noqa: E402
from experiments import common  # noqa: E402
from experiments.experiment1 import MODELS as ORIGINAL_MODELS  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402
register_all_modules(init_default_scope=True)

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment1b")

EPSILON = 8.0
NUM_ITER = 10

EXTRA_MODELS = {
    "target_x101": (
        f"{common.MMDET_DIR}/configs/mask_rcnn/mask-rcnn_x101-32x4d_fpn_1x_coco.py",
        f"{REPO_ROOT}/checkpoints/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205-478d0b67.pth",
        "same-family",  # ResNeXt vẫn tính là họ ResNet (grouped conv, không đổi stage design)
    ),
    "target_r50_3x": (
        f"{common.MMDET_DIR}/configs/mask_rcnn/mask-rcnn_r50_fpn_ms-poly-3x_coco.py",
        f"{REPO_ROOT}/checkpoints/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth",
        "same-family",  # CÙNG backbone R50 với surrogate, chỉ train recipe khác (3x+ms vs 1x)
    ),
}

ZOO_MODELS = {**ORIGINAL_MODELS, **EXTRA_MODELS}

RUNS = [
    {"key": "runB_mi_fgsm_zoo", "decay": 1.0, "objective": "cls", "input_diversity": False,
     "models": ZOO_MODELS, "desc": "MI-FGSM cls-only, model zoo mở rộng (6 model, kiểm tra model-strength confound)"},
    {"key": "runC_bim", "decay": 0.0, "objective": "cls", "input_diversity": False,
     "models": ORIGINAL_MODELS, "desc": "BIM/I-FGSM (không momentum), 4 model gốc"},
    {"key": "runD_cls_bbox", "decay": 1.0, "objective": "cls_bbox", "input_diversity": False,
     "models": ORIGINAL_MODELS, "desc": "MI-FGSM, objective cls+bbox, 4 model gốc"},
    {"key": "runE_di_fgsm", "decay": 1.0, "objective": "cls", "input_diversity": True,
     "models": ORIGINAL_MODELS, "desc": "DI-MI-FGSM (input diversity), 4 model gốc"},
]


def run_one(run, ds, n, cat_ids, coco_gt, img_ids, gt_by_image):
    print(f"\n[exp1b] === {run['key']}: {run['desc']} ===")
    models = {name: common.load_model(cfg, ckpt) for name, (cfg, ckpt, _f) in run["models"].items()}
    surrogate = models["surrogate_r50"]

    results = {name: {"clean": [], "adv": []} for name in run["models"]}
    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        x_clean = sample["inputs"].float().to(common.DEVICE)
        ds_sample = sample["data_samples"]

        x_adv = iterative_linf_attack(
            surrogate, x_clean, ds_sample, epsilon=EPSILON, num_iter=NUM_ITER,
            decay=run["decay"], objective=run["objective"], input_diversity=run["input_diversity"])

        for name, model in models.items():
            results[name]["clean"].extend(common.predict_coco_format(model, x_clean, ds_sample, cat_ids))
            results[name]["adv"].extend(common.predict_coco_format(model, x_adv, ds_sample, cat_ids))

        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp1b][{run['key']}] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút)")

    summary = {}
    for name, (_, _, family) in run["models"].items():
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
        print(f"[exp1b][{run['key']}] {name} ({family}): clean AP50={clean_map['AP50']:.4f} -> "
              f"adv AP50={adv_map['AP50']:.4f} | ASR={asr['ASR']:.4f} "
              f"({asr['evaded_objects']}/{asr['clean_correct_objects']})")

    del models
    import torch
    torch.cuda.empty_cache()
    return summary


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[exp1b] building dataset (limit={limit})...")
    ds, n = common.build_dataset(ORIGINAL_MODELS["surrogate_r50"][0], limit=limit)
    cat_ids = ds.cat_ids
    print(f"[exp1b] {n} images (dataset total: {len(ds)})")

    coco_gt = COCO(common.SUBSET_ANNO)
    img_ids = [ds[i]["data_samples"].metainfo["img_id"] for i in range(n)]
    gt_by_image = common.load_gt_by_image(coco_gt, img_ids)

    all_results = {}
    for run in RUNS:
        all_results[run["key"]] = {
            "desc": run["desc"],
            "decay": run["decay"], "objective": run["objective"], "input_diversity": run["input_diversity"],
            "summary": run_one(run, ds, n, cat_ids, coco_gt, img_ids, gt_by_image),
        }
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump({"epsilon": EPSILON, "num_iter": NUM_ITER, "n_images": n, "runs": all_results}, f, indent=2)

    print(f"\n[exp1b] Xong toàn bộ. Kết quả: {OUT_DIR}/results.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
