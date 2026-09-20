"""Thí nghiệm 3A (research_plan.md, chốt 2026-09-20 sau khi khóa RQ2 — xem docs/progress_log.md):
causal intervention. RQ2 đã khóa dựa trên CORRELATION (2A/2C: gradient/backward-sensitivity
alignment đi cùng ASR, mạnh nhất từ Stage 3-4). 3A kiểm tra xem đó có phải QUAN HỆ NHÂN QUẢ không:
nếu regularize (giảm variance) gradient CHÍNH XÁC tại Stage 3-4 trong lúc craft attack, cross-family
ASR (ConvNeXt, Swin) có tăng nhiều hơn regularize ở Stage 1-2 không?

4 setting, cùng 4 model + cùng config attack (epsilon=8.0, num_iter=10, decay=1.0, objective=cls —
y hệt Thí nghiệm 1) như baseline duy nhất khác là reg_stages:
  - baseline: không regularize (y hệt attacks/detection_attacks.py gốc)
  - reg_s1s2: regularize Stage 1-2 (shallow) — nhóm đối chứng, dự đoán ít/không tăng cross-family ASR
  - reg_s3s4: regularize Stage 3-4 (deep) — dự đoán tăng cross-family ASR NHIỀU HƠN s1s2
  - reg_all: regularize cả 4 stage — tham khảo (không phải trọng tâm hypothesis)

Hypothesis chính (research_plan.md, docs/progress_log.md 2026-09-20):
    Gain_{S3-4}^{cross-family} > Gain_{S1-2}^{cross-family}
với Gain = ASR_reg - ASR_baseline, đặc biệt trên target_convnext_t và target_swin_t.

Không dùng để craft attack thật sự tốt hơn (chưa phải method) — chỉ để kiểm tra causal hypothesis
trước khi đầu tư thiết kế method (research_plan.md §8 bước 3-4, dùng lại đúng model/subset đã có).
"""
import json
import os
import sys
import time

from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attacks.backward_reg_attack import iterative_linf_attack_reg  # noqa: E402
from experiments import common  # noqa: E402
from experiments.experiment1 import MODELS  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402
register_all_modules(init_default_scope=True)

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment3a")

EPSILON = 8.0
NUM_ITER = 10
DECAY = 1.0
OBJECTIVE = "cls"

SETTINGS = [
    {"key": "baseline", "reg_stages": frozenset(), "desc": "Baseline (không regularize, y hệt Thí nghiệm 1)"},
    {"key": "reg_s1s2", "reg_stages": frozenset({1, 2}), "desc": "Regularize Stage 1-2 (shallow, nhóm đối chứng)"},
    {"key": "reg_s3s4", "reg_stages": frozenset({3, 4}), "desc": "Regularize Stage 3-4 (deep, hypothesis chính)"},
    {"key": "reg_all", "reg_stages": frozenset({1, 2, 3, 4}), "desc": "Regularize cả 4 stage (tham khảo)"},
]


def run_one(setting, ds, n, cat_ids, coco_gt, img_ids, gt_by_image, models):
    print(f"\n[exp3a] === {setting['key']}: {setting['desc']} ===")
    surrogate = models["surrogate_r50"]
    results = {name: {"clean": [], "adv": []} for name in MODELS}
    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        x_clean = sample["inputs"].float().to(common.DEVICE)
        ds_sample = sample["data_samples"]

        x_adv = iterative_linf_attack_reg(
            surrogate, x_clean, ds_sample, epsilon=EPSILON, num_iter=NUM_ITER, decay=DECAY,
            objective=OBJECTIVE, reg_stages=setting["reg_stages"])

        for name, model in models.items():
            results[name]["clean"].extend(common.predict_coco_format(model, x_clean, ds_sample, cat_ids))
            results[name]["adv"].extend(common.predict_coco_format(model, x_adv, ds_sample, cat_ids))

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp3a][{setting['key']}] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút)")

    summary = {}
    for name, (_, _, family) in MODELS.items():
        clean_map = common.compute_map(coco_gt, results[name]["clean"], img_ids)
        adv_map = common.compute_map(coco_gt, results[name]["adv"], img_ids)
        clean_by_img = common.group_by_image(results[name]["clean"])
        adv_by_img = common.group_by_image(results[name]["adv"])
        asr = common.compute_asr(gt_by_image, clean_by_img, adv_by_img)
        summary[name] = common.summarize_model_result(family, clean_map, adv_map, asr)
        print(f"[exp3a][{setting['key']}] {name} ({family}): clean AP50={clean_map['AP50']:.4f} -> "
              f"adv AP50={adv_map['AP50']:.4f} | ASR={asr['ASR']:.4f} "
              f"({asr['evaded_objects']}/{asr['clean_correct_objects']})")

    return summary


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[exp3a] loading models...")
    models = {name: common.load_model(cfg, ckpt) for name, (cfg, ckpt, _f) in MODELS.items()}

    print(f"[exp3a] building dataset (limit={limit})...")
    ds, n = common.build_dataset(MODELS["surrogate_r50"][0], limit=limit)
    cat_ids = ds.cat_ids
    print(f"[exp3a] {n} images (dataset total: {len(ds)})")

    coco_gt = COCO(common.SUBSET_ANNO)
    img_ids = [ds[i]["data_samples"].metainfo["img_id"] for i in range(n)]
    gt_by_image = common.load_gt_by_image(coco_gt, img_ids)

    all_results = {}
    for setting in SETTINGS:
        all_results[setting["key"]] = {
            "desc": setting["desc"], "reg_stages": sorted(setting["reg_stages"]),
            "summary": run_one(setting, ds, n, cat_ids, coco_gt, img_ids, gt_by_image, models),
        }
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump({"epsilon": EPSILON, "num_iter": NUM_ITER, "decay": DECAY, "objective": OBJECTIVE,
                       "n_images": n, "settings": all_results}, f, indent=2)

    print("\n[exp3a] Gain (ASR_reg - ASR_baseline) theo target, so S1-S2 vs S3-S4:")
    base = all_results["baseline"]["summary"]
    for key in ["reg_s1s2", "reg_s3s4", "reg_all"]:
        print(f"  --- {key} ---")
        for name in MODELS:
            gain = all_results[key]["summary"][name]["ASR"] - base[name]["ASR"]
            print(f"    {name}: gain={gain:+.4f}")

    print(f"\n[exp3a] Xong. Kết quả: {OUT_DIR}/results.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
