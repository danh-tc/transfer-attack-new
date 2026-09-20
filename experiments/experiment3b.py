"""Thí nghiệm 3B (chốt 2026-09-20, sau literature search overlap — xem docs/progress_log.md):
3A dùng 1 kiểu regularize DUY NHẤT ở Stage 3-4 — clip variance ĐỀU trên toàn bộ feature map (lẫn
cả background, không chỉ vùng object). Câu hỏi 3B:

> Object-aware weighting có giúp backward regularization tập trung vào transferable
> object-sensitive directions thay vì regularize toàn feature map như nhau hay không?

4 setting, cùng 4 model + cùng config attack (epsilon=8.0, num_iter=10, decay=1.0, objective=cls —
y hệt Thí nghiệm 1/3A):
  - baseline: không regularize.
  - reg_s3s4: clip variance đều tại Stage 3-4 (y hệt 3A, làm mốc tham chiếu).
  - object_weight_only: CHỈ nhân gradient với mask object (1.0 trong vùng GT box, 0.3 ngoài) tại
    Stage 3-4, KHÔNG clip — cô lập tác dụng của riêng "tập trung vào object".
  - reg_s3s4_object_weight: clip NHƯNG chỉ áp trong vùng object (blend theo mask) — kết hợp cả 2.

Nếu setting cuối vượt rõ `reg_s3s4` (đặc biệt trên ConvNeXt/Swin) → object-conditioned weighting
có giúp ích thật, đủ khác biệt so với literature (TGR/MIG/PAS/GRA đều không có bước "đo backward
sensitivity để chọn stage" + "object-conditioned" — xem docs/progress_log.md) để phát triển thành
method candidate.

LƯU Ý (phát hiện 2026-09-20 khi verify trước khi chạy): setting có backward hook cho kết quả dao
động nhẹ giữa các lần chạy CÙNG 1 code (baseline luôn tái lập y hệt, không có hook nên không bị ảnh
hưởng) — nondeterminism của GPU/cudnn khi có `tensor.register_hook`, không phải bug logic. Ở n=300
(hàng nghìn object) hiệu ứng này nên bị trung bình hóa nhỏ đi nhiều so với n=5 lúc sanity-test.
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

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment3b")

EPSILON = 8.0
NUM_ITER = 10
DECAY = 1.0
OBJECTIVE = "cls"

SETTINGS = [
    {"key": "baseline", "stage_modes": {}, "desc": "Baseline (không regularize, y hệt Thí nghiệm 1/3A)"},
    {"key": "reg_s3s4", "stage_modes": {3: "clip", 4: "clip"},
     "desc": "Clip variance ĐỀU tại Stage 3-4 (y hệt 3A, mốc tham chiếu)"},
    {"key": "object_weight_only", "stage_modes": {3: "weight", 4: "weight"},
     "desc": "CHỈ weight theo object mask tại Stage 3-4, KHÔNG clip"},
    {"key": "reg_s3s4_object_weight", "stage_modes": {3: "clip_weight", 4: "clip_weight"},
     "desc": "Clip + object mask kết hợp tại Stage 3-4"},
]


def run_one(setting, ds, n, cat_ids, coco_gt, img_ids, gt_by_image, models):
    print(f"\n[exp3b] === {setting['key']}: {setting['desc']} ===")
    surrogate = models["surrogate_r50"]
    results = {name: {"clean": [], "adv": []} for name in MODELS}
    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        x_clean = sample["inputs"].float().to(common.DEVICE)
        ds_sample = sample["data_samples"]

        x_adv = iterative_linf_attack_reg(
            surrogate, x_clean, ds_sample, epsilon=EPSILON, num_iter=NUM_ITER, decay=DECAY,
            objective=OBJECTIVE, stage_modes=setting["stage_modes"])

        for name, model in models.items():
            results[name]["clean"].extend(common.predict_coco_format(model, x_clean, ds_sample, cat_ids))
            results[name]["adv"].extend(common.predict_coco_format(model, x_adv, ds_sample, cat_ids))

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp3b][{setting['key']}] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút)")

    summary = {}
    for name, (_, _, family) in MODELS.items():
        clean_map = common.compute_map(coco_gt, results[name]["clean"], img_ids)
        adv_map = common.compute_map(coco_gt, results[name]["adv"], img_ids)
        clean_by_img = common.group_by_image(results[name]["clean"])
        adv_by_img = common.group_by_image(results[name]["adv"])
        asr = common.compute_asr(gt_by_image, clean_by_img, adv_by_img)
        summary[name] = common.summarize_model_result(family, clean_map, adv_map, asr)
        print(f"[exp3b][{setting['key']}] {name} ({family}): clean AP50={clean_map['AP50']:.4f} -> "
              f"adv AP50={adv_map['AP50']:.4f} | ASR={asr['ASR']:.4f} "
              f"({asr['evaded_objects']}/{asr['clean_correct_objects']})")

    return summary


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[exp3b] loading models...")
    models = {name: common.load_model(cfg, ckpt) for name, (cfg, ckpt, _f) in MODELS.items()}

    print(f"[exp3b] building dataset (limit={limit})...")
    ds, n = common.build_dataset(MODELS["surrogate_r50"][0], limit=limit)
    cat_ids = ds.cat_ids
    print(f"[exp3b] {n} images (dataset total: {len(ds)})")

    coco_gt = COCO(common.SUBSET_ANNO)
    img_ids = [ds[i]["data_samples"].metainfo["img_id"] for i in range(n)]
    gt_by_image = common.load_gt_by_image(coco_gt, img_ids)

    all_results = {}
    for setting in SETTINGS:
        all_results[setting["key"]] = {
            "desc": setting["desc"], "stage_modes": setting["stage_modes"],
            "summary": run_one(setting, ds, n, cat_ids, coco_gt, img_ids, gt_by_image, models),
        }
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump({"epsilon": EPSILON, "num_iter": NUM_ITER, "decay": DECAY, "objective": OBJECTIVE,
                       "n_images": n, "settings": all_results}, f, indent=2)

    print("\n[exp3b] Gain (ASR_setting - ASR_baseline) theo target:")
    base = all_results["baseline"]["summary"]
    for key in ["reg_s3s4", "object_weight_only", "reg_s3s4_object_weight"]:
        print(f"  --- {key} ---")
        for name in MODELS:
            gain = all_results[key]["summary"][name]["ASR"] - base[name]["ASR"]
            print(f"    {name}: gain={gain:+.4f}")

    print(f"\n[exp3b] Xong. Kết quả: {OUT_DIR}/results.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
