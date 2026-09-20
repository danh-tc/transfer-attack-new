"""Thí nghiệm 2B.1 (chốt 2026-09-20, sau khi 2B cho negative finding — xem docs/progress_log.md):
2B cho thấy global CKA(surrogate, target) KHÔNG dự báo được thứ tự ASR (R101 > Swin > ConvNeXt ở
CKA, nhưng R101 > ConvNeXt > Swin ở ASR/gradient alignment). Câu hỏi hẹp hơn ở đây, TRONG TỪNG
TARGET riêng biệt (không so target với nhau nữa):

> Trong cùng 1 target, object transfer THÀNH CÔNG (evaded) có feature representation align với
> surrogate cao hơn object transfer THẤT BẠI (not evaded) không?

Nếu CKA^evaded > CKA^not ở 1 vùng stage cụ thể → feature similarity vẫn có vai trò cục bộ (theo
từng target), chỉ là không dùng được để so sánh GIỮA các target khác nhau (global CKA 2B). Nếu
không thấy pattern này → raw forward feature similarity không phải cơ chế giải thích transfer,
nên chuyển hẳn sang backward information (gradient/feature-gradient/Jacobian alignment).

Dùng lại raw feature đã lưu ở outputs/experiment2b/features/{target}_stage{s}.npz (X=surrogate
feature, Y=target feature, evaded=nhãn bool) — KHÔNG forward lại GPU, chỉ tính toán CPU/numpy.

So sánh công bằng bằng bootstrap: n_sample = min(n_evaded, n_not_evaded) (2 nhóm luôn khác size,
đặc biệt r101 có ASR cao nên n_evaded gần bằng hoặc hơn n_not_evaded, còn convnext/swin thì ngược
lại) — resample CÓ HOÀN LẠI n_sample từ mỗi nhóm, B lần, tính CKA mỗi lần, so phân phối 2 nhóm thay
vì so 1 con số CKA tính trên toàn bộ nhóm (vốn bị lệch cỡ mẫu).

Output: outputs/experiment2b1/summary.json.
"""
import json
import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(REPO_ROOT, "outputs", "experiment2b", "features")
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "experiment2b1")

N_BOOT = 200
SEED = 42


def linear_cka(X, Y):
    """Linear CKA qua Gram matrix (N x N) thay vì cross-covariance (D x D) — với D lên tới 2048
    (stage sâu của ResNet) và N (số object) chỉ ~500-800, dạng D x D đắt hơn nhiều (đo thực nghiệm
    2026-09-20: ~1.5s/lần gọi ở D=2048,N=700 dùng công thức D x D, blowup tổng runtime bootstrap
    lên ~70 phút — quá chậm cho 1 kiểm tra "rẻ"). 2 công thức toán học tương đương cho linear
    kernel (Kornblith et al. 2019 Lemma), chỉ khác complexity: O(N^2 D) so với O(N D^2)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    hsic_xy = float(np.sum(K * L))
    hsic_xx = float(np.sum(K * K))
    hsic_yy = float(np.sum(L * L))
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12)


def bootstrap_cka(X, Y, n_sample, rng, n_boot=N_BOOT):
    """Resample CÓ HOÀN LẠI n_sample dòng từ (X,Y) (giữ khớp index — cùng object), tính CKA mỗi
    lần. Trả về mảng (n_boot,) giá trị CKA."""
    n = X.shape[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n_sample)
        vals[b] = linear_cka(X[idx], Y[idx])
    return vals


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    files = sorted(os.listdir(FEATURES_DIR))
    targets_stages = sorted({tuple(f.replace(".npz", "").rsplit("_stage", 1)) for f in files})

    results = {}
    for t_name, stage_str in targets_stages:
        stage = f"stage{stage_str}"
        data = np.load(os.path.join(FEATURES_DIR, f"{t_name}_stage{stage_str}.npz"))
        X, Y, evaded = data["X"], data["Y"], data["evaded"]

        X_ev, Y_ev = X[evaded], Y[evaded]
        X_not, Y_not = X[~evaded], Y[~evaded]
        n_ev, n_not = len(X_ev), len(X_not)
        if n_ev < 5 or n_not < 5:
            print(f"[exp2b1] {t_name} {stage}: bỏ qua (n_evaded={n_ev}, n_not_evaded={n_not} quá ít)")
            continue
        n_sample = min(n_ev, n_not)

        boot_ev = bootstrap_cka(X_ev, Y_ev, n_sample, rng)
        boot_not = bootstrap_cka(X_not, Y_not, n_sample, rng)

        mean_ev, mean_not = float(boot_ev.mean()), float(boot_not.mean())
        ci_ev = [float(np.percentile(boot_ev, 2.5)), float(np.percentile(boot_ev, 97.5))]
        ci_not = [float(np.percentile(boot_not, 2.5)), float(np.percentile(boot_not, 97.5))]
        # non-overlap của 2 CI 95% là 1 chỉ báo thô về ý nghĩa thống kê (bảo thủ hơn 1 test chính
        # thức, nhưng đủ dùng ở đây vì 2 bootstrap distribution độc lập, không paired được).
        non_overlap = ci_ev[0] > ci_not[1] or ci_not[0] > ci_ev[1]

        results.setdefault(t_name, {})[stage] = {
            "n_evaded_total": int(n_ev), "n_not_evaded_total": int(n_not), "n_sample_matched": int(n_sample),
            "cka_evaded_mean": mean_ev, "cka_evaded_ci95": ci_ev,
            "cka_not_evaded_mean": mean_not, "cka_not_evaded_ci95": ci_not,
            "diff_evaded_minus_not": mean_ev - mean_not,
            "ci95_non_overlap": bool(non_overlap),
        }
        flag = "***" if non_overlap else ""
        print(f"[exp2b1] {t_name} {stage}: CKA_evaded={mean_ev:.4f} {ci_ev} vs "
              f"CKA_not_evaded={mean_not:.4f} {ci_not} | diff={mean_ev - mean_not:+.4f} {flag}")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"n_boot": N_BOOT, "seed": SEED, "results": results}, f, indent=2)
    print(f"[exp2b1] Xong. Kết quả: {OUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
