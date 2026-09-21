"""Thí nghiệm 4A — OSFD matched-budget (chốt 2026-09-21, xem docs/progress_log.md): so sánh method
v0.1 với OSFD (Wu et al., AAAI 2024) — attack chuyên biệt cho object detection transferability,
comparator quan trọng nhất cho claim SOTA (research_plan.md §11). Cùng threat model TUYỆT ĐỐI với
mọi thí nghiệm khác trong dự án: epsilon=8.0, num_iter=10, cùng 300 ảnh, cùng 4 model (surrogate
R50 + target R101/ConvNeXt-Tiny/Swin-Tiny), cùng eval/ASR pipeline (experiments/common.py).

Port OSFD: attacks/osfd_attack.py (đọc trực tiếp official code /workspace/OSFD, không đoán từ
paper). Đây là "OSFD-matched" — primary comparison. "OSFD-extended" (budget lớn hơn, sanity check
phụ, KHÔNG dùng để claim thắng/thua) làm riêng ở experiment4b.py.

Output: outputs/experiment4a/results.json + raw_predictions.json (giống format experiment1.py).
"""
import json
import os
import sys
import time

from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attacks.osfd_attack import iterative_osfd_attack  # noqa: E402
from experiments import common  # noqa: E402
from experiments.experiment1 import MODELS  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402
register_all_modules(init_default_scope=True)

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment4a")

EPSILON = 8.0
NUM_ITER = 10
DECAY = 1.0


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[exp4a] loading models...")
    models = {name: common.load_model(cfg, ckpt) for name, (cfg, ckpt, _f) in MODELS.items()}
    surrogate = models["surrogate_r50"]

    print(f"[exp4a] building dataset (limit={limit})...")
    ds, n = common.build_dataset(MODELS["surrogate_r50"][0], limit=limit)
    cat_ids = ds.cat_ids
    print(f"[exp4a] {n} images (dataset total: {len(ds)})")

    coco_gt = COCO(common.SUBSET_ANNO)
    img_ids = [ds[i]["data_samples"].metainfo["img_id"] for i in range(n)]
    gt_by_image = common.load_gt_by_image(coco_gt, img_ids)

    results = {name: {"clean": [], "adv": []} for name in MODELS}
    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        x_clean = sample["inputs"].float().to(common.DEVICE)
        ds_sample = sample["data_samples"]

        x_adv = iterative_osfd_attack(surrogate, x_clean, ds_sample, epsilon=EPSILON,
                                       num_iter=NUM_ITER, decay=DECAY)

        for name, model in models.items():
            results[name]["clean"].extend(common.predict_coco_format(model, x_clean, ds_sample, cat_ids))
            results[name]["adv"].extend(common.predict_coco_format(model, x_adv, ds_sample, cat_ids))

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp4a] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút)")

    with open(os.path.join(OUT_DIR, "raw_predictions.json"), "w") as f:
        json.dump(results, f)

    summary = {}
    for name, (_, _, family) in MODELS.items():
        clean_map = common.compute_map(coco_gt, results[name]["clean"], img_ids)
        adv_map = common.compute_map(coco_gt, results[name]["adv"], img_ids)
        clean_by_img = common.group_by_image(results[name]["clean"])
        adv_by_img = common.group_by_image(results[name]["adv"])
        asr = common.compute_asr(gt_by_image, clean_by_img, adv_by_img)
        summary[name] = common.summarize_model_result(family, clean_map, adv_map, asr)
        print(f"[exp4a] {name} ({family}): clean AP50={clean_map['AP50']:.4f} -> "
              f"adv AP50={adv_map['AP50']:.4f} | ASR={asr['ASR']:.4f} "
              f"({asr['evaded_objects']}/{asr['clean_correct_objects']})")

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"attack": "OSFD-matched", "epsilon": EPSILON, "num_iter": NUM_ITER, "decay": DECAY,
                    "osfd_k": 3.0, "n_images": n, "summary": summary}, f, indent=2)

    cross_avg = (summary["target_convnext_t"]["ASR"] + summary["target_swin_t"]["ASR"]) / 2
    gap = summary["target_r101"]["ASR"] - cross_avg
    print(f"\n[exp4a] CrossAvg(ConvNeXt,Swin)={cross_avg:.4f} | R101={summary['target_r101']['ASR']:.4f} "
          f"| TransferGap={gap:+.4f}")
    print(f"[exp4a] Xong. Kết quả: {OUT_DIR}/results.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
