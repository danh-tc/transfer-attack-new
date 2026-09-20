"""Ablation B (research_plan.md §9.1, chốt 2026-09-20 sau Ablation A — xem docs/progress_log.md):
sweep object-weight strength lambda, giữ cố định k=3 (đã xác nhận tối ưu ở Ablation A), Stage {3,4},
mode="clip_weight", epsilon=8.0, num_iter=10, decay=1.0, objective=cls, surrogate=r50, n=300.

Ablation A đã xác nhận: clipping (khối 2) là thành phần chính tạo gain, có mức tối ưu ở k=3. Câu hỏi
còn lại: object-region weighting (khối 3) có thực sự CẦN THIẾT không (so với chỉ clip đều, không
phân biệt object/background — lambda=0), và strength bao nhiêu là hợp lý?

lambda -> beta(lambda) = 1/(1+lambda) (research_plan.md §9.1):
  - lambda=0  -> beta=1 -> mask đồng nhất -> suy biến CHÍNH XÁC về "clip" thuần = reg_s3s4 (đã có
    từ 3A/3B, TÁI SỬ DỤNG, không chạy lại — verify unit test 2026-09-20: max diff=0.0 tại lambda=0).
  - lambda~2.333 (beta=0.3) = config đã chạy ở 3B/Ablation A (k3_original), GẦN nhưng KHÔNG trùng
    lambda=2 trong grid dưới — vẫn giữ làm điểm tham chiếu nội suy, không thay thế lambda=2.

Grid theo đúng đề xuất: {0, 0.25, 0.5, 1, 2}. Chỉ chạy MỚI 0.25/0.5/1/2 (4 điểm), lambda=0 tái sử
dụng từ reg_s3s4 (3B, đã verify n=300 đúng trong docs/progress_log.md — KHÔNG đọc từ
outputs/experiment3b/results.json vì file đó từng bị ghi đè nhầm, xem entry Ablation A).
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

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment3d")

EPSILON = 8.0
NUM_ITER = 10
DECAY = 1.0
OBJECTIVE = "cls"
K = 3
STAGE_MODES = {3: "clip_weight", 4: "clip_weight"}

# Số liệu ĐÃ VERIFY đúng ở n=300 (docs/progress_log.md, entry Thí nghiệm 3B + Ablation A) — không
# đọc từ outputs/experiment3b/results.json (từng bị ghi đè nhầm bởi 1 lần sanity-test n=5).
REUSED = {
    "lam0_eq_reg_s3s4": {
        "surrogate_r50": {"ASR": 0.9610}, "target_r101": {"ASR": 0.7167},
        "target_convnext_t": {"ASR": 0.4649}, "target_swin_t": {"ASR": 0.3965},
    },
    "lam2.333_ref_k3_original": {
        "surrogate_r50": {"ASR": 0.9644}, "target_r101": {"ASR": 0.7128},
        "target_convnext_t": {"ASR": 0.4702}, "target_swin_t": {"ASR": 0.4013},
    },
}

SETTINGS = [
    {"key": "lam0.25", "lam": 0.25, "desc": "lambda=0.25 (beta=0.8, gần reg_s3s4)"},
    {"key": "lam0.5", "lam": 0.5, "desc": "lambda=0.5 (beta=0.667)"},
    {"key": "lam1", "lam": 1.0, "desc": "lambda=1 (beta=0.5)"},
    {"key": "lam2", "lam": 2.0, "desc": "lambda=2 (beta=0.333, gần lambda~2.333 đã có)"},
]


def run_one(setting, ds, n, cat_ids, coco_gt, img_ids, gt_by_image, models):
    print(f"\n[exp3d] === {setting['key']}: {setting['desc']} ===")
    surrogate = models["surrogate_r50"]
    results = {name: {"clean": [], "adv": []} for name in MODELS}
    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        x_clean = sample["inputs"].float().to(common.DEVICE)
        ds_sample = sample["data_samples"]

        x_adv = iterative_linf_attack_reg(
            surrogate, x_clean, ds_sample, epsilon=EPSILON, num_iter=NUM_ITER, decay=DECAY,
            objective=OBJECTIVE, stage_modes=STAGE_MODES, k=K, lam=setting["lam"])

        for name, model in models.items():
            results[name]["clean"].extend(common.predict_coco_format(model, x_clean, ds_sample, cat_ids))
            results[name]["adv"].extend(common.predict_coco_format(model, x_adv, ds_sample, cat_ids))

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp3d][{setting['key']}] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút)")

    summary = {}
    for name, (_, _, family) in MODELS.items():
        clean_map = common.compute_map(coco_gt, results[name]["clean"], img_ids)
        adv_map = common.compute_map(coco_gt, results[name]["adv"], img_ids)
        clean_by_img = common.group_by_image(results[name]["clean"])
        adv_by_img = common.group_by_image(results[name]["adv"])
        asr = common.compute_asr(gt_by_image, clean_by_img, adv_by_img)
        summary[name] = common.summarize_model_result(family, clean_map, adv_map, asr)
        print(f"[exp3d][{setting['key']}] {name} ({family}): clean AP50={clean_map['AP50']:.4f} -> "
              f"adv AP50={adv_map['AP50']:.4f} | ASR={asr['ASR']:.4f} "
              f"({asr['evaded_objects']}/{asr['clean_correct_objects']})")

    return summary


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[exp3d] loading models...")
    models = {name: common.load_model(cfg, ckpt) for name, (cfg, ckpt, _f) in MODELS.items()}

    print(f"[exp3d] building dataset (limit={limit})...")
    ds, n = common.build_dataset(MODELS["surrogate_r50"][0], limit=limit)
    cat_ids = ds.cat_ids
    print(f"[exp3d] {n} images (dataset total: {len(ds)})")

    coco_gt = COCO(common.SUBSET_ANNO)
    img_ids = [ds[i]["data_samples"].metainfo["img_id"] for i in range(n)]
    gt_by_image = common.load_gt_by_image(coco_gt, img_ids)

    all_results = dict(REUSED)
    for setting in SETTINGS:
        all_results[setting["key"]] = run_one(setting, ds, n, cat_ids, coco_gt, img_ids, gt_by_image, models)
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump({
                "epsilon": EPSILON, "num_iter": NUM_ITER, "decay": DECAY, "objective": OBJECTIVE,
                "k": K, "stage_modes": STAGE_MODES, "n_images": n,
                "reused_note": "lam0_eq_reg_s3s4 và lam2.333_ref_k3_original lấy từ docs/progress_log.md "
                                "(verify n=300 đúng), không đọc từ outputs/experiment3b/results.json",
                "settings": all_results,
            }, f, indent=2)

    print("\n[exp3d] Bảng ASR theo lambda:")
    order = ["lam0_eq_reg_s3s4", "lam0.25", "lam0.5", "lam1", "lam2", "lam2.333_ref_k3_original"]
    for key in order:
        s = all_results[key]
        cross_avg = (s["target_convnext_t"]["ASR"] + s["target_swin_t"]["ASR"]) / 2
        gap = s["target_r101"]["ASR"] - cross_avg
        print(f"  {key:26s} ConvNeXt={s['target_convnext_t']['ASR']:.4f} Swin={s['target_swin_t']['ASR']:.4f} "
              f"CrossAvg={cross_avg:.4f} | R101={s['target_r101']['ASR']:.4f} | "
              f"WhiteBox={s['surrogate_r50']['ASR']:.4f} | TransferGap={gap:+.4f}")

    print(f"\n[exp3d] Xong. Kết quả: {OUT_DIR}/results.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
