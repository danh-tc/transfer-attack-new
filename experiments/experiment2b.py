"""Thí nghiệm 2B (research_plan.md §7.B, chốt 2026-09-20): layer-wise feature alignment.

Câu hỏi: cross-family divergence xuất hiện mạnh nhất ở tầng feature nào, và mức divergence đó có
liên quan tới transferability (ASR/gradient alignment đã đo ở Thí nghiệm 2A) không?

Thiết kế: TÁI SỬ DỤNG đúng population object đã dùng ở Thí nghiệm 2A (đọc từ
outputs/experiment2a/records.jsonl — cùng 300 ảnh, cùng tập object clean-correct theo từng
target) để so sánh trực tiếp được với cos_sim/evaded đã có, KHÔNG generate lại attack (2B chỉ cần
ảnh sạch — so sánh representation của backbone, không liên quan tới adversarial example).

Với mỗi stage backbone (S1..S4, tương ứng C2-C5/ResNet hay Stage1-4/ConvNeXt,Swin — đã verify
2026-09-20: cả 4 model đều có đúng 4 stage, cùng stride [4,8,16,32] dù channel dim khác nhau,
256/512/1024/2048 cho ResNet vs 96/192/384/768 cho ConvNeXt/Swin — đây chính là lý do dùng linear
CKA thay vì cosine similarity trực tiếp như 2A, vì CKA không yêu cầu cùng chiều feature):

  1. Dùng RoIAlign (spatial_scale = 1/stride_stage) trên feature map của stage đó để lấy
     object-conditioned feature (pool 7x7 rồi global-average) cho từng object trong population.
  2. Với mỗi (stage, target), gom feature surrogate (X) và feature target (Y) của CÙNG population
     object thành 2 ma trận (N, C_s) và (N, C_t), tính linear CKA(X, Y).

Hypothesis: CKA_{R50,R101}^(l) > CKA_{R50,ConvNeXt}^(l) > CKA_{R50,Swin}^(l) ở mọi stage, và tìm
vùng stage cụ thể nơi divergence tăng mạnh nhất theo family (thường kỳ vọng: shallow gần nhau, deep
tách biệt theo kiến trúc — theo FIA/SMP-Attack literature, xem docs/progress_log.md).

Output: outputs/experiment2b/summary.json (bảng CKA theo stage x target).
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from mmcv.ops import roi_align
from mmdet.structures.bbox import bbox2roi

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from experiments import common  # noqa: E402
from experiments.experiment1 import MODELS  # noqa: E402
from experiments.experiment2a import OUT_DIR as EXP2A_DIR  # noqa: E402
from experiments.experiment2a import _gt_bboxes_net  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402
register_all_modules(init_default_scope=True)

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment2b")

STAGE_STRIDES = [4, 8, 16, 32]  # verify 2026-09-20: giống hệt cho cả 4 model, xem progress_log.md
ROI_OUTPUT_SIZE = 7


def _load_2a_population():
    """Đọc outputs/experiment2a/records.jsonl -> {img_id: {target_name: {object_idx: evaded}}}.
    Giữ luôn nhãn evaded (không chỉ set object_idx) để 2B.1 dùng lại được mà không phải join lại
    với records.jsonl lần nữa."""
    records_path = os.path.join(EXP2A_DIR, "records.jsonl")
    by_image = defaultdict(lambda: defaultdict(dict))
    with open(records_path) as f:
        for line in f:
            r = json.loads(line)
            by_image[r["image_id"]][r["target"]][r["object_idx"]] = r["evaded"]
    print(f"[exp2b] đọc {len(by_image)} ảnh có object từ {records_path}")
    return by_image


@torch.no_grad()
def _forward_stages(model, x, ds_sample):
    data = dict(inputs=[x], data_samples=[ds_sample])
    batch = model.data_preprocessor(data, training=False)
    return model.backbone(batch["inputs"])  # tuple 4 stage feature map


@torch.no_grad()
def _pool_objects(feat, rois, stride):
    """RoIAlign object-conditioned feature tại 1 stage, pool 7x7 -> global average -> (k, C)."""
    # torch.autograd.Function.apply (bản torch đang dùng) không nhận keyword arguments — phải
    # truyền hết theo đúng thứ tự positional của RoIAlignFunction.forward.
    pooled = roi_align(feat, rois, (ROI_OUTPUT_SIZE, ROI_OUTPUT_SIZE), 1.0 / stride, 0, "avg", True)
    return pooled.mean(dim=(2, 3))


def linear_cka(X, Y):
    """Linear CKA (Kornblith et al. 2019), công thức hiệu quả qua cross-covariance — không cần
    cùng chiều feature giữa X, Y (đúng lý do dùng CKA thay vì cosine trực tiếp ở đây)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro")
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro")
    return float(hsic_xy / (hsic_xx * hsic_yy + 1e-12))


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    by_image = _load_2a_population()

    print("[exp2b] loading models...")
    models = {name: common.load_model(cfg, ckpt) for name, (cfg, ckpt, _f) in MODELS.items()}
    surrogate = models["surrogate_r50"]
    targets = {name: m for name, m in models.items() if name != "surrogate_r50"}
    for m in models.values():
        m.eval()

    print(f"[exp2b] building dataset (limit={limit})...")
    ds, n = common.build_dataset(MODELS["surrogate_r50"][0], limit=limit)
    print(f"[exp2b] {n} images (dataset total: {len(ds)})")

    n_stages = len(STAGE_STRIDES)
    # X[stage][target] / Y[stage][target]: list các vector numpy, tích lũy qua toàn bộ ảnh.
    # evaded[stage][target]: list bool cùng thứ tự — giữ lại để 2B.1 tách nhóm evaded/not-evaded mà
    # không cần forward lại (research_plan.md §7.A/2B.1, docs/progress_log.md 2026-09-20).
    X = {s: {t: [] for t in targets} for s in range(n_stages)}
    Y = {s: {t: [] for t in targets} for s in range(n_stages)}
    evaded = {s: {t: [] for t in targets} for s in range(n_stages)}

    t0 = time.time()
    n_images_used = 0
    for i in range(n):
        sample = ds[i]
        ds_sample = sample["data_samples"]
        img_id = int(ds_sample.metainfo["img_id"])
        needed_by_target = by_image.get(img_id)
        if not needed_by_target:
            continue

        union_needed = sorted(set().union(*[set(d.keys()) for d in needed_by_target.values()]))
        if not union_needed:
            continue
        n_images_used += 1

        x_clean = sample["inputs"].float().to(common.DEVICE)
        gt_bboxes_net = _gt_bboxes_net(ds_sample, common.DEVICE)
        rois_all = bbox2roi([gt_bboxes_net])

        union_idx_tensor = torch.tensor(union_needed, device=common.DEVICE, dtype=torch.long)
        union_rois = rois_all[union_idx_tensor]
        feats_s = _forward_stages(surrogate, x_clean, ds_sample)
        surrogate_vecs = {}  # stage -> {object_idx: vector}
        for s in range(n_stages):
            vecs = _pool_objects(feats_s[s], union_rois, STAGE_STRIDES[s])
            surrogate_vecs[s] = {obj_idx: vecs[k] for k, obj_idx in enumerate(union_needed)}
        del feats_s

        for t_name, t_model in targets.items():
            t_obj_evaded = needed_by_target.get(t_name, {})
            t_needed = sorted(t_obj_evaded.keys())
            if not t_needed:
                continue
            t_idx_tensor = torch.tensor(t_needed, device=common.DEVICE, dtype=torch.long)
            t_rois = rois_all[t_idx_tensor]
            feats_t = _forward_stages(t_model, x_clean, ds_sample)
            for s in range(n_stages):
                vecs_t = _pool_objects(feats_t[s], t_rois, STAGE_STRIDES[s])
                for k, obj_idx in enumerate(t_needed):
                    X[s][t_name].append(surrogate_vecs[s][obj_idx].cpu().numpy())
                    Y[s][t_name].append(vecs_t[k].cpu().numpy())
                    evaded[s][t_name].append(bool(t_obj_evaded[obj_idx]))
            del feats_t

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"[exp2b] {i + 1}/{n} ảnh xong ({rate:.2f} ảnh/s, ETA {eta / 60:.1f} phút, "
                  f"{n_images_used} ảnh có object dùng được)")

    print("[exp2b] tính linear CKA theo stage x target + lưu raw feature cho 2B.1...")
    features_dir = os.path.join(OUT_DIR, "features")
    os.makedirs(features_dir, exist_ok=True)
    cka_table = {}
    for s in range(n_stages):
        cka_table[f"stage{s + 1}"] = {}
        for t_name in targets:
            if not X[s][t_name]:
                continue
            Xs = np.stack(X[s][t_name]).astype(np.float32)
            Ys = np.stack(Y[s][t_name]).astype(np.float32)
            evaded_arr = np.array(evaded[s][t_name], dtype=bool)
            cka = linear_cka(Xs, Ys)
            cka_table[f"stage{s + 1}"][t_name] = {"cka": cka, "n_objects": len(X[s][t_name])}
            print(f"[exp2b] stage{s + 1} {t_name}: CKA={cka:.4f} (n={len(X[s][t_name])})")
            np.savez(os.path.join(features_dir, f"{t_name}_stage{s + 1}.npz"),
                     X=Xs, Y=Ys, evaded=evaded_arr)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "n_images": n, "n_images_with_objects": n_images_used,
            "stage_strides": STAGE_STRIDES, "roi_output_size": ROI_OUTPUT_SIZE,
            "cka_table": cka_table,
        }, f, indent=2)
    print(f"[exp2b] Xong. Kết quả: {OUT_DIR}/summary.json, raw feature: {features_dir}/")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
