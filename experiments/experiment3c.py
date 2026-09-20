"""Ablation A (research_plan.md §9.1, chốt 2026-09-20 sau khi formalize Method v0.1 — xem
docs/progress_log.md): sweep clip bound k, giữ cố định mọi thứ khác — lam≈2.333 (beta=0.3, đúng
config Thí nghiệm 3B), Stage {3,4}, mode="clip_weight", epsilon=8.0, num_iter=10, decay=1.0,
objective=cls, surrogate=r50, subset n=300.

Hypothesis: non-monotonic — k quá nhỏ (clip quá mạnh) mất signal hữu ích; k quá lớn/no-clip (gần
gradient gốc) cross-family giảm; vùng k trung gian cho cross-family ASR tốt nhất mà không hy sinh
quá nhiều white-box/same-family ASR.

TÁI SỬ DỤNG (không chạy lại) 3 điểm đã có từ Thí nghiệm 3B (n=300, cùng subset, cùng config):
  - baseline (không regularize)
  - no-clip = "object_weight_only" của 3B (mask/weighting nhưng KHÔNG clip — mode="weight", không
    phải "clip_weight" với clip=identity, vì 2 thứ đó cho kết quả khác nhau về mặt toán học: xem
    docs/research_plan.md §9.1, "weight" = mask⊙g trực tiếp, còn "clip_weight" với clip→identity
    suy biến về g không đổi, không phải 1 control có ý nghĩa)
  - k=3 (gốc) = "reg_s3s4_object_weight" của 3B

Script này CHỈ chạy mới: k=1, k=2, k=4, và 1 bản lặp lại k=3 (noise floor — so với k=3 gốc để biết
sai lệch giữa k=1/2/4 và k=3 có lớn hơn nhiễu run-to-run thật hay không, do đã phát hiện GPU/cudnn
gây dao động nhẹ ngay cả ở baseline không hook — docs/progress_log.md 2026-09-20 entry Thí nghiệm 3B).

QUAN TRỌNG: không đổi bất kỳ config cudnn/determinism nào giữa các lần chạy (kể cả so với lúc chạy
3B) để giữ đúng tinh thần "so sánh công bằng" — không set torch.backends.cudnn.deterministic hay
tương tự ở đây hay ở 3B, đúng theo yêu cầu user.
"""
import json
import os
import sys
import time

from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attacks.backward_reg_attack import iterative_linf_attack_reg, DEFAULT_LAM  # noqa: E402
from experiments import common  # noqa: E402
from experiments.experiment1 import MODELS  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402
register_all_modules(init_default_scope=True)

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment3c")
EXP3B_RESULTS = os.path.join(REPO_ROOT, "outputs", "experiment3b", "results.json")

EPSILON = 8.0
NUM_ITER = 10
DECAY = 1.0
OBJECTIVE = "cls"
STAGE_MODES = {3: "clip_weight", 4: "clip_weight"}

SETTINGS = [
    {"key": "k1", "k": 1, "desc": "Clip bound k=1 (clipping mạnh)"},
    {"key": "k2", "k": 2, "desc": "Clip bound k=2 (vừa-mạnh)"},
    {"key": "k4", "k": 4, "desc": "Clip bound k=4 (clipping nhẹ)"},
    {"key": "k3_replicate", "k": 3, "desc": "Clip bound k=3, LẶP LẠI 3B để đo noise floor"},
]


def run_one(setting, ds, n, cat_ids, coco_gt, img_ids, gt_by_image, models):
    print(f"\n[exp3c] === {setting['key']}: {setting['desc']} ===")
    surrogate = models["surrogate_r50"]
    results = {name: {"clean": [], "adv": []} for name in MODELS}
    t0 = time.time()
    for i in range(n):
        sample = ds[i]
        x_clean = sample["inputs"].float().to(common.DEVICE)
        ds_sample = sample["data_samples"]

        x_adv = iterative_linf_attack_reg(
            surrogate, x_clean, ds_sample, epsilon=EPSILON, num_iter=NUM_ITER, decay=DECAY,
            objective=OBJECTIVE, stage_modes=STAGE_MODES, k=setting["k"], lam=DEFAULT_LAM)

        for name, model in models.items():
            results[name]["clean"].extend(common.predict_coco_format(model, x_clean, ds_sample, cat_ids))
            results[name]["adv"].extend(common.predict_coco_format(model, x_adv, ds_sample, cat_ids))

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp3c][{setting['key']}] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút)")

    summary = {}
    for name, (_, _, family) in MODELS.items():
        clean_map = common.compute_map(coco_gt, results[name]["clean"], img_ids)
        adv_map = common.compute_map(coco_gt, results[name]["adv"], img_ids)
        clean_by_img = common.group_by_image(results[name]["clean"])
        adv_by_img = common.group_by_image(results[name]["adv"])
        asr = common.compute_asr(gt_by_image, clean_by_img, adv_by_img)
        summary[name] = common.summarize_model_result(family, clean_map, adv_map, asr)
        print(f"[exp3c][{setting['key']}] {name} ({family}): clean AP50={clean_map['AP50']:.4f} -> "
              f"adv AP50={adv_map['AP50']:.4f} | ASR={asr['ASR']:.4f} "
              f"({asr['evaded_objects']}/{asr['clean_correct_objects']})")

    return summary


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(EXP3B_RESULTS) as f:
        exp3b = json.load(f)
    reused = {
        "baseline": exp3b["settings"]["baseline"]["summary"],
        "no-clip": exp3b["settings"]["object_weight_only"]["summary"],
        "k3_original": exp3b["settings"]["reg_s3s4_object_weight"]["summary"],
    }
    print(f"[exp3c] tái sử dụng 3 điểm từ {EXP3B_RESULTS}: baseline, no-clip (=object_weight_only), "
          f"k3_original (=reg_s3s4_object_weight)")

    print("[exp3c] loading models...")
    models = {name: common.load_model(cfg, ckpt) for name, (cfg, ckpt, _f) in MODELS.items()}

    print(f"[exp3c] building dataset (limit={limit})...")
    ds, n = common.build_dataset(MODELS["surrogate_r50"][0], limit=limit)
    cat_ids = ds.cat_ids
    print(f"[exp3c] {n} images (dataset total: {len(ds)})")

    coco_gt = COCO(common.SUBSET_ANNO)
    img_ids = [ds[i]["data_samples"].metainfo["img_id"] for i in range(n)]
    gt_by_image = common.load_gt_by_image(coco_gt, img_ids)

    all_results = dict(reused)
    for setting in SETTINGS:
        all_results[setting["key"]] = run_one(setting, ds, n, cat_ids, coco_gt, img_ids, gt_by_image, models)
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
            json.dump({
                "epsilon": EPSILON, "num_iter": NUM_ITER, "decay": DECAY, "objective": OBJECTIVE,
                "lam": DEFAULT_LAM, "stage_modes": STAGE_MODES, "n_images": n,
                "reused_from": "outputs/experiment3b/results.json",
                "settings": all_results,
            }, f, indent=2)

    print("\n[exp3c] Bảng ASR theo k (đọc theo đúng thứ tự đã thống nhất: cross-family -> same-family "
          "-> white-box -> TransferGap):")
    order = ["baseline", "no-clip", "k1", "k2", "k3_original", "k3_replicate", "k4"]
    for key in order:
        s = all_results[key]
        cross_avg = (s["target_convnext_t"]["ASR"] + s["target_swin_t"]["ASR"]) / 2
        gap = s["target_r101"]["ASR"] - cross_avg
        print(f"  {key:14s} ConvNeXt={s['target_convnext_t']['ASR']:.4f} Swin={s['target_swin_t']['ASR']:.4f} "
              f"CrossAvg={cross_avg:.4f} | R101={s['target_r101']['ASR']:.4f} | "
              f"WhiteBox={s['surrogate_r50']['ASR']:.4f} | TransferGap={gap:+.4f}")

    print(f"\n[exp3c] Xong. Kết quả: {OUT_DIR}/results.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
