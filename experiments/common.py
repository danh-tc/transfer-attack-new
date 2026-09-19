"""Hạ tầng dùng chung cho experiments/experiment1.py và experiment1b.py: load model/dataset,
predict ra COCO format, tính mAP (pycocotools) và ASR (greedy IoU match tự viết).
"""
import os

import torch
from pycocotools.cocoeval import COCOeval

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MMDET_DIR = os.path.join(REPO_ROOT, "third_party", "mmdetection")
DATA_ROOT = os.path.join(REPO_ROOT, "data", "coco")
SUBSET_ANNO = os.path.join(DATA_ROOT, "annotations", "instances_val2017_subset1000.json")

DEVICE = "cuda:0"

# Ngưỡng đánh giá — không được research_plan.md ấn định cứng, chọn giá trị chuẩn COCO/mmdet
# thường dùng (xem docs/progress_log.md 2026-09-19 entry Thí nghiệm 1 để biết lý do).
SCORE_THR = 0.3
IOU_THR = 0.5


def build_dataset(surrogate_cfg_path, limit=None):
    """Trả về (dataset, n) — n = min(limit, len(dataset)) nếu limit khác None.

    LƯU Ý: KHÔNG được giới hạn bằng cách gán đè ds.data_list — mmengine BaseDataset serialize
    data_list thành byte buffer ngay trong __init__ (serialize_data=True mặc định), gán đè sau đó
    không có tác dụng (đã tự kiểm chứng bằng thực nghiệm, xem progress_log.md 2026-09-19 "Bug 2").
    Giới hạn đúng cách: chỉ lặp range(n) nhỏ hơn ở nơi gọi.
    """
    from mmdet.registry import DATASETS
    from mmengine.config import Config

    cfg = Config.fromfile(surrogate_cfg_path)
    test_pipeline = cfg.test_dataloader.dataset.pipeline
    ds = DATASETS.build(dict(
        type="CocoDataset",
        data_root=DATA_ROOT + "/",
        ann_file="annotations/instances_val2017_subset1000.json",
        data_prefix=dict(img="val2017/"),
        test_mode=True,
        pipeline=test_pipeline,
    ))
    n = len(ds) if limit is None else min(limit, len(ds))
    return ds, n


def load_model(cfg_path, ckpt_path):
    from mmdet.apis import init_detector
    m = init_detector(cfg_path, ckpt_path, device=DEVICE)
    m.eval()
    return m


@torch.no_grad()
def predict_coco_format(model, x, ds_sample, cat_ids):
    """Chạy 1 ảnh qua model, trả về list dict COCO-format (đã rescale về tọa độ ảnh gốc)."""
    data = dict(inputs=[x], data_samples=[ds_sample])
    batch = model.data_preprocessor(data, training=False)
    results = model.predict(batch["inputs"], batch["data_samples"], rescale=True)
    r = results[0].pred_instances
    img_id = int(ds_sample.metainfo["img_id"])
    out = []
    bboxes = r.bboxes.cpu().numpy()
    scores = r.scores.cpu().numpy()
    labels = r.labels.cpu().numpy()
    for box, score, label in zip(bboxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        out.append({
            "image_id": img_id,
            "category_id": int(cat_ids[int(label)]),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score),
        })
    return out


def compute_map(coco_gt, coco_results, img_ids):
    if len(coco_results) == 0:
        return {"AP": 0.0, "AP50": 0.0}
    coco_dt = coco_gt.loadRes(coco_results)
    E = COCOeval(coco_gt, coco_dt, iouType="bbox")
    E.params.imgIds = img_ids
    E.evaluate()
    E.accumulate()
    E.summarize()
    return {"AP": float(E.stats[0]), "AP50": float(E.stats[1])}


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_greedy(gt_list, pred_list, iou_thr, score_thr):
    """COCO-style greedy match: pred sort giảm dần score, mỗi pred bắt cặp GT cùng category_id
    IoU cao nhất còn trống, IoU>=iou_thr. Trả về set các gt index đã được match."""
    preds = [p for p in pred_list if p["score"] >= score_thr]
    preds.sort(key=lambda p: -p["score"])
    matched_gt = set()
    for p in preds:
        best_iou, best_j = iou_thr, -1
        for j, g in enumerate(gt_list):
            if j in matched_gt or g["category_id"] != p["category_id"]:
                continue
            iou = box_iou(xywh_to_xyxy(g["bbox"]), xywh_to_xyxy(p["bbox"]))
            if iou >= best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            matched_gt.add(best_j)
    return matched_gt


def group_by_image(results):
    out = {}
    for r in results:
        out.setdefault(r["image_id"], []).append(r)
    return out


def compute_asr(gt_by_image, clean_results_by_image, adv_results_by_image, iou_thr=IOU_THR, score_thr=SCORE_THR):
    n_clean_correct, n_evaded = 0, 0
    for img_id, gts in gt_by_image.items():
        if not gts:
            continue
        clean_preds = clean_results_by_image.get(img_id, [])
        adv_preds = adv_results_by_image.get(img_id, [])
        clean_matched = match_greedy(gts, clean_preds, iou_thr, score_thr)
        adv_matched = match_greedy(gts, adv_preds, iou_thr, score_thr)
        n_clean_correct += len(clean_matched)
        n_evaded += len(clean_matched - adv_matched)
    asr = n_evaded / n_clean_correct if n_clean_correct > 0 else float("nan")
    return {"clean_correct_objects": n_clean_correct, "evaded_objects": n_evaded, "ASR": asr}


def load_gt_by_image(coco_gt, img_ids):
    gt_by_image = {}
    for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_ids)):
        gt_by_image.setdefault(ann["image_id"], []).append(
            {"category_id": ann["category_id"], "bbox": ann["bbox"]})
    for img_id in img_ids:
        gt_by_image.setdefault(img_id, [])
    return gt_by_image
