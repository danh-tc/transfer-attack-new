"""Component Ablation (chốt 2026-09-20, sau khi Ablation A+B đủ để dừng hyperparameter tuning — xem
docs/progress_log.md). Câu hỏi: improvement thật sự đến từ Stage 3, Stage 4, hay cần CẢ HAI? Stage
nào đóng góp nhiều hơn?

5 điểm, cùng model/attack config (epsilon=8.0, num_iter=10, decay=1.0, objective=cls, k=3):
  - baseline: không regularize.
  - clip_s3: chỉ clip Stage 3 (mode="clip", KHÔNG object weighting).
  - clip_s4: chỉ clip Stage 4 (mode="clip", KHÔNG object weighting).
  - clip_s3s4: clip cả Stage 3+4 (mode="clip") — = `reg_s3s4` đã có ở Thí nghiệm 3A/3B.
  - clip_s3s4_objw: Stage 3+4, mode="clip_weight", lambda=0.5 — candidate tốt nhất từ Ablation B.

CHỈ CHẠY MỚI clip_s3, clip_s4 (2 điểm). 3 điểm còn lại TÁI SỬ DỤNG (không chạy lại — tránh lặp lại
sự cố ghi đè đã gặp, và tiết kiệm compute):
  - baseline, clip_s3s4 (=reg_s3s4): lấy từ outputs/experiment3b/results.json (đã khôi phục đúng
    n=300 sau sự cố ghi đè — 2 điểm này tính CÙNG 1 lần chạy nên nhất quán nội bộ với nhau).
  - clip_s3s4_objw (=lambda=0.5): lấy từ outputs/experiment3d/results.json (lần chạy đầu của
    Ablation B, đã có replicate xác nhận thứ hạng ở entry riêng).
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

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment3e")

EPSILON = 8.0
NUM_ITER = 10
DECAY = 1.0
OBJECTIVE = "cls"
K = 3

SETTINGS = [
    {"key": "clip_s3", "stage_modes": {3: "clip"}, "desc": "Chỉ clip Stage 3 (shallow-mid của vùng đã chọn)"},
    {"key": "clip_s4", "stage_modes": {4: "clip"}, "desc": "Chỉ clip Stage 4 (sâu nhất)"},
]


def run_one(setting, ds, n, cat_ids, coco_gt, img_ids, gt_by_image, models):
    print(f"\n[exp3e] === {setting['key']}: {setting['desc']} ===")
    surrogate = models["surrogate_r50"]
    results = {name: {"clean": [], "adv": []} for name in MODELS}
    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        x_clean = sample["inputs"].float().to(common.DEVICE)
        ds_sample = sample["data_samples"]

        x_adv = iterative_linf_attack_reg(
            surrogate, x_clean, ds_sample, epsilon=EPSILON, num_iter=NUM_ITER, decay=DECAY,
            objective=OBJECTIVE, stage_modes=setting["stage_modes"], k=K)

        for name, model in models.items():
            results[name]["clean"].extend(common.predict_coco_format(model, x_clean, ds_sample, cat_ids))
            results[name]["adv"].extend(common.predict_coco_format(model, x_adv, ds_sample, cat_ids))

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp3e][{setting['key']}] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút)")

    summary = {}
    for name, (_, _, family) in MODELS.items():
        clean_map = common.compute_map(coco_gt, results[name]["clean"], img_ids)
        adv_map = common.compute_map(coco_gt, results[name]["adv"], img_ids)
        clean_by_img = common.group_by_image(results[name]["clean"])
        adv_by_img = common.group_by_image(results[name]["adv"])
        asr = common.compute_asr(gt_by_image, clean_by_img, adv_by_img)
        summary[name] = common.summarize_model_result(family, clean_map, adv_map, asr)
        print(f"[exp3e][{setting['key']}] {name} ({family}): clean AP50={clean_map['AP50']:.4f} -> "
              f"adv AP50={adv_map['AP50']:.4f} | ASR={asr['ASR']:.4f} "
              f"({asr['evaded_objects']}/{asr['clean_correct_objects']})")

    return summary


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(REPO_ROOT, "outputs", "experiment3b", "results.json")) as f:
        exp3b = json.load(f)
    with open(os.path.join(REPO_ROOT, "outputs", "experiment3d", "results.json")) as f:
        exp3d = json.load(f)
    reused = {
        "baseline": exp3b["settings"]["baseline"]["summary"],
        "clip_s3s4": exp3b["settings"]["reg_s3s4"]["summary"],
        "clip_s3s4_objw": exp3d["settings"]["lam0.5"],
    }
    print("[exp3e] tái sử dụng baseline + clip_s3s4 từ experiment3b, clip_s3s4_objw từ experiment3d")

    print("[exp3e] loading models...")
    models = {name: common.load_model(cfg, ckpt) for name, (cfg, ckpt, _f) in MODELS.items()}

    print(f"[exp3e] building dataset (limit={limit})...")
    ds, n = common.build_dataset(MODELS["surrogate_r50"][0], limit=limit)
    cat_ids = ds.cat_ids
    print(f"[exp3e] {n} images (dataset total: {len(ds)})")

    coco_gt = COCO(common.SUBSET_ANNO)
    img_ids = [ds[i]["data_samples"].metainfo["img_id"] for i in range(n)]
    gt_by_image = common.load_gt_by_image(coco_gt, img_ids)

    all_results = dict(reused)
    for setting in SETTINGS:
        all_results[setting["key"]] = run_one(setting, ds, n, cat_ids, coco_gt, img_ids, gt_by_image, models)
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump({
                "epsilon": EPSILON, "num_iter": NUM_ITER, "decay": DECAY, "objective": OBJECTIVE,
                "k": K, "n_images": n,
                "reused_from": ["outputs/experiment3b/results.json", "outputs/experiment3d/results.json"],
                "settings": all_results,
            }, f, indent=2)

    print("\n[exp3e] Bảng theo component:")
    order = ["baseline", "clip_s3", "clip_s4", "clip_s3s4", "clip_s3s4_objw"]
    for key in order:
        s = all_results[key]
        cross_avg = (s["target_convnext_t"]["ASR"] + s["target_swin_t"]["ASR"]) / 2
        gap = s["target_r101"]["ASR"] - cross_avg
        print(f"  {key:16s} ConvNeXt={s['target_convnext_t']['ASR']:.4f} Swin={s['target_swin_t']['ASR']:.4f} "
              f"CrossAvg={cross_avg:.4f} | R101={s['target_r101']['ASR']:.4f} | "
              f"WhiteBox={s['surrogate_r50']['ASR']:.4f} | TransferGap={gap:+.4f}")

    print(f"\n[exp3e] Xong. Kết quả: {OUT_DIR}/results.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
