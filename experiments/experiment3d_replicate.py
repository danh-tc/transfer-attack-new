"""Replicate check cho Ablation B (chốt 2026-09-20, xem docs/progress_log.md entry Ablation A về lý
do cần replicate trước khi tin 1 điểm dữ liệu duy nhất — noise floor GPU/cudnn quan sát được
~0.001-0.005 ở CrossAvg). Chạy lại ĐỘC LẬP lambda=0 (control — pure clipping, không object
weighting) và lambda=0.5 (candidate tốt nhất trong sweep Ablation B) để xem chênh lệch
CrossAvg(lambda=0.5) - CrossAvg(lambda=0) ≈ 0.008 quan sát được ở lần chạy đầu có lặp lại được
không, trước khi chốt entry Ablation B vào progress_log.md (append-only — tránh phải viết entry
"tạm thời" rồi đính chính).

lambda=2.333 KHÔNG cần replicate ở đây — đã có 2 điểm gần tương đương từ Ablation A (k=3 gốc=0.4357,
k=3 lặp lại=0.4349, chênh 0.0008), coi như đã biết noise floor tại vùng lambda cao.

Tái sử dụng run_one() từ experiments/experiment3d.py (không copy lại logic)."""
import json
import os
import sys

from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from experiments import common  # noqa: E402
from experiments.experiment1 import MODELS  # noqa: E402
from experiments.experiment3d import run_one  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402
register_all_modules(init_default_scope=True)

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment3d_replicate")

SETTINGS = [
    {"key": "lam0_replicate", "lam": 0.0, "desc": "lambda=0 REPLICATE (control — pure clipping, không object weighting)"},
    {"key": "lam0.5_replicate", "lam": 0.5, "desc": "lambda=0.5 REPLICATE (candidate tốt nhất ở lần chạy đầu)"},
]


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[exp3d_replicate] loading models...")
    models = {name: common.load_model(cfg, ckpt) for name, (cfg, ckpt, _f) in MODELS.items()}

    print(f"[exp3d_replicate] building dataset (limit={limit})...")
    ds, n = common.build_dataset(MODELS["surrogate_r50"][0], limit=limit)
    cat_ids = ds.cat_ids
    print(f"[exp3d_replicate] {n} images (dataset total: {len(ds)})")

    coco_gt = COCO(common.SUBSET_ANNO)
    img_ids = [ds[i]["data_samples"].metainfo["img_id"] for i in range(n)]
    gt_by_image = common.load_gt_by_image(coco_gt, img_ids)

    all_results = {}
    for setting in SETTINGS:
        all_results[setting["key"]] = run_one(setting, ds, n, cat_ids, coco_gt, img_ids, gt_by_image, models)
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump({"n_images": n, "settings": all_results}, f, indent=2)

    # So với lần chạy đầu (đã ghi trong outputs/experiment3d/results.json)
    with open(os.path.join(REPO_ROOT, "outputs", "experiment3d", "results.json")) as f:
        first_run = json.load(f)["settings"]

    print("\n[exp3d_replicate] So sánh lần 1 vs lần lặp lại:")
    pairs = [("lam0_eq_reg_s3s4", "lam0_replicate"), ("lam0.5", "lam0.5_replicate")]
    for first_key, rep_key in pairs:
        s1, s2 = first_run[first_key], all_results[rep_key]
        cross1 = (s1["target_convnext_t"]["ASR"] + s1["target_swin_t"]["ASR"]) / 2
        cross2 = (s2["target_convnext_t"]["ASR"] + s2["target_swin_t"]["ASR"]) / 2
        print(f"  {first_key} (lần 1): CrossAvg={cross1:.4f} | {rep_key} (lặp lại): CrossAvg={cross2:.4f} "
              f"| diff={cross2 - cross1:+.4f}")

    print(f"\n[exp3d_replicate] Xong. Kết quả: {OUT_DIR}/results.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
