"""Thí nghiệm 3I (chốt 2026-09-21, sau khi 3F negative + 3H yếu — xem docs/progress_log.md): thu
hẹp câu hỏi mechanism lại 1 mức nữa. Không hỏi "regularization tăng cosine alignment với target"
(3F, bác bỏ) hay "regularization làm trajectory ổn định hơn nói chung" (3H, effect yếu) — hỏi trực
tiếp: **per-image reduction in gradient concentration (ΔC, đã có từ 3G) có tương quan với việc 1
object cụ thể "giành lại" được transfer (baseline fail -> reg success) hay không?**

Tái sử dụng tối đa dữ liệu đã có, chỉ chạy MỚI phần bắt buộc:
  - evaded_baseline per (image, target, object): lấy thẳng từ outputs/experiment2a/records.jsonl
    (baseline MI-FGSM, epsilon=8/num_iter=10/decay=1/objective=cls — đúng "baseline" ở đây, không
    chạy lại attack này).
  - ΔC(image, stage, metric) = raw - reg_clip_weight: lấy thẳng từ outputs/experiment3g/records.jsonl
    (không cần GPU lại — 3G đã tính sẵn trên đúng surrogate, đúng config k=3/lam=0.5/clip_weight).
  - MỚI: chạy attack REG (Stage 3-4, k=3, lam=0.5, mode clip_weight — dùng thẳng
    attacks.backward_reg_attack.iterative_linf_attack_reg có sẵn, không viết lại) trên surrogate cho
    đúng 300 ảnh trong population 2A, rồi predict trên 3 target để lấy evaded_reg per object (so
    khớp object_idx với population đã có ở 2A — KHÔNG cần predict lại ảnh sạch, population 2A đã tự
    đảm bảo object đó clean-correct).

Đơn vị phân tích: 1 (ẢNH, TARGET) = 1 ROW (không phải per-object — object cùng ảnh chia sẻ chung 1
ΔC(image), không độc lập với nhau; nhưng các ẢNH khác nhau thì độc lập). Với mỗi (ảnh, target) có
>=1 object baseline-fail (evaded_baseline=False, tức "còn cơ hội để gained"):
  gained_rate = n_gained / n_baseline_fail
  trong đó gained = object có evaded_baseline=False VÀ evaded_reg=True.

Test: Mann-Whitney U so ΔC(image) giữa nhóm ảnh có gained_rate>0 vs nhóm gained_rate==0 (độc lập
hoàn toàn giữa các ảnh — không cần cluster-correction) + Spearman correlation(ΔC, gained_rate) làm
kiểm tra continuous. Chạy riêng cho từng target (ConvNeXt, Swin — trọng tâm cross-family; R101 làm
đối chứng same-family, kỳ vọng tín hiệu yếu hơn hoặc không có nếu hiệu ứng đặc thù cross-family).

Output: outputs/experiment3i/records.jsonl (1 dòng/ảnh/target) + summary.json.
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from attacks.backward_reg_attack import iterative_linf_attack_reg  # noqa: E402
from experiments import common  # noqa: E402
from experiments.experiment1 import MODELS  # noqa: E402
from experiments.experiment2a import OUT_DIR as EXP2A_DIR  # noqa: E402
from experiments.experiment2a import _gt_list_orig  # noqa: E402
from experiments.experiment3g import OUT_DIR as EXP3G_DIR  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402
register_all_modules(init_default_scope=True)

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment3i")

EPSILON = 8.0
NUM_ITER = 10
DECAY = 1.0
OBJECTIVE = "cls"
K = 3
LAM = 0.5
REG_STAGE_MODES = {3: "clip_weight", 4: "clip_weight"}
DC_METRIC = "top1pct_energy_frac"  # metric chính dùng cho ΔC — xem docstring 3G để biết vì sao


def _load_2a_baseline():
    """Trả về: by_image[img_id][target] = {object_idx: evaded_baseline}."""
    by_image = defaultdict(lambda: defaultdict(dict))
    with open(os.path.join(EXP2A_DIR, "records.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            by_image[r["image_id"]][r["target"]][r["object_idx"]] = r["evaded"]
    return by_image


def _load_3g_delta_c():
    """Trả về: delta_c[img_id] = mean qua Stage 3+4 của (raw - reg_clip_weight) cho DC_METRIC."""
    per_image_stage = {}
    with open(os.path.join(EXP3G_DIR, "records.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            raw = r[f"raw__{DC_METRIC}"]
            reg = r[f"reg_clip_weight__{DC_METRIC}"]
            per_image_stage.setdefault(r["image_id"], []).append(raw - reg)
    return {img_id: float(np.mean(vals)) for img_id, vals in per_image_stage.items()}


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[exp3i] đọc population baseline (2A) + ΔC (3G)...")
    baseline_by_image = _load_2a_baseline()
    delta_c_by_image = _load_3g_delta_c()
    common_images = set(baseline_by_image.keys()) & set(delta_c_by_image.keys())
    print(f"[exp3i] {len(baseline_by_image)} ảnh trong 2A, {len(delta_c_by_image)} ảnh trong 3G, "
          f"{len(common_images)} ảnh chung dùng được")

    print("[exp3i] loading models...")
    models = {name: common.load_model(cfg, ckpt) for name, (cfg, ckpt, _f) in MODELS.items()}
    surrogate = models["surrogate_r50"]
    targets = {name: m for name, m in models.items() if name != "surrogate_r50"}

    print(f"[exp3i] building dataset (limit={limit})...")
    ds, n = common.build_dataset(MODELS["surrogate_r50"][0], limit=limit)
    cat_ids = ds.cat_ids
    print(f"[exp3i] {n} images (dataset total: {len(ds)})")

    records = []
    t0 = time.time()
    n_used = 0
    for i in range(n):
        sample = ds[i]
        ds_sample = sample["data_samples"]
        img_id = int(ds_sample.metainfo["img_id"])
        if img_id not in common_images:
            continue
        needed_by_target = baseline_by_image[img_id]
        if not any(needed_by_target.values()):
            continue
        n_used += 1

        x_clean = sample["inputs"].float().to(common.DEVICE)
        gt_list_orig = _gt_list_orig(ds_sample, cat_ids)
        delta_c = delta_c_by_image[img_id]

        x_adv_reg = iterative_linf_attack_reg(
            surrogate, x_clean, ds_sample, epsilon=EPSILON, num_iter=NUM_ITER, decay=DECAY,
            objective=OBJECTIVE, stage_modes=REG_STAGE_MODES, k=K, lam=LAM)

        for t_name, t_model in targets.items():
            obj_baseline = needed_by_target.get(t_name, {})
            if not obj_baseline:
                continue
            adv_reg_preds = common.predict_coco_format(t_model, x_adv_reg, ds_sample, cat_ids)
            adv_reg_matched = common.match_greedy(gt_list_orig, adv_reg_preds, common.IOU_THR, common.SCORE_THR)

            n_baseline_fail = 0
            n_gained = 0
            n_baseline_evaded = 0
            n_lost = 0
            for j, evaded_baseline in obj_baseline.items():
                evaded_reg = j not in adv_reg_matched
                if evaded_baseline:
                    n_baseline_evaded += 1
                    if not evaded_reg:
                        n_lost += 1
                else:
                    n_baseline_fail += 1
                    if evaded_reg:
                        n_gained += 1

            if n_baseline_fail == 0:
                continue
            records.append({
                "image_id": img_id, "target": t_name, "delta_c": delta_c,
                "n_baseline_fail": n_baseline_fail, "n_gained": n_gained,
                "gained_rate": n_gained / n_baseline_fail,
                "n_baseline_evaded": n_baseline_evaded, "n_lost": n_lost,
            })

        if (i + 1) % 20 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp3i] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút, "
                  f"{n_used} ảnh dùng được, {len(records)} record)")

    with open(os.path.join(OUT_DIR, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print("[exp3i] tổng hợp theo target (Mann-Whitney + Spearman)...")
    summary = {}
    for t_name in targets:
        rs = [r for r in records if r["target"] == t_name]
        if not rs:
            continue
        delta_c_arr = np.array([r["delta_c"] for r in rs])
        gained_rate_arr = np.array([r["gained_rate"] for r in rs])
        dc_gain = delta_c_arr[gained_rate_arr > 0]
        dc_nogain = delta_c_arr[gained_rate_arr == 0]

        entry = {
            "family": MODELS[t_name][2], "n_rows": len(rs),
            "n_images_with_gain": int((gained_rate_arr > 0).sum()),
            "n_images_no_gain": int((gained_rate_arr == 0).sum()),
            "mean_delta_c_gain": float(dc_gain.mean()) if len(dc_gain) else float("nan"),
            "mean_delta_c_nogain": float(dc_nogain.mean()) if len(dc_nogain) else float("nan"),
        }
        if len(dc_gain) > 1 and len(dc_nogain) > 1:
            u_stat, p_value = mannwhitneyu(dc_gain, dc_nogain, alternative="greater")
            entry["mannwhitney_u"] = float(u_stat)
            entry["mannwhitney_p_onesided"] = float(p_value)
        if len(rs) > 2 and delta_c_arr.std() > 0 and gained_rate_arr.std() > 0:
            rho, p_spearman = spearmanr(delta_c_arr, gained_rate_arr)
            entry["spearman_rho"] = float(rho)
            entry["spearman_p"] = float(p_spearman)

        summary[t_name] = entry
        extra = ""
        if "mannwhitney_p_onesided" in entry:
            extra += f" | MWU p={entry['mannwhitney_p_onesided']:.2e}"
        if "spearman_rho" in entry:
            extra += f" | spearman rho={entry['spearman_rho']:.4f} p={entry['spearman_p']:.2e}"
        print(f"[exp3i] {t_name}: n={len(rs)} ΔC(gain)={entry['mean_delta_c_gain']:.4g} "
              f"ΔC(no-gain)={entry['mean_delta_c_nogain']:.4g}{extra}")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"epsilon": EPSILON, "num_iter": NUM_ITER, "decay": DECAY, "objective": OBJECTIVE,
                    "k": K, "lam": LAM, "reg_stage_modes": REG_STAGE_MODES, "dc_metric": DC_METRIC,
                    "n_images": n, "n_images_used": n_used, "summary": summary}, f, indent=2)
    print(f"[exp3i] Xong. Kết quả: {OUT_DIR}/records.jsonl, {OUT_DIR}/summary.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
