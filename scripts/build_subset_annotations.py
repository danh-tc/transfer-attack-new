"""Lọc instances_val2017.json về đúng 1000 ảnh trong configs/coco_val2017_subset_1000.json.

Sinh ra data/coco/annotations/instances_val2017_subset1000.json (không commit — dữ liệu dẫn
xuất, deterministic từ 2 file input). CocoDataset/CocoMetric của mmdet dùng file này làm
ann_file cho cả việc build data sample lẫn tính mAP, để mAP tính đúng trên đúng subset.
"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET_FILE = os.path.join(REPO_ROOT, "configs", "coco_val2017_subset_1000.json")
FULL_ANNO = os.path.join(REPO_ROOT, "data", "coco", "annotations", "instances_val2017.json")
OUT_ANNO = os.path.join(REPO_ROOT, "data", "coco", "annotations", "instances_val2017_subset1000.json")


def main():
    with open(SUBSET_FILE) as f:
        subset = json.load(f)
    subset_ids = set(subset["image_ids"])

    with open(FULL_ANNO) as f:
        coco = json.load(f)

    images = [img for img in coco["images"] if img["id"] in subset_ids]
    assert len(images) == len(subset_ids), f"{len(images)} != {len(subset_ids)}"
    annotations = [a for a in coco["annotations"] if a["image_id"] in subset_ids]

    out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": images,
        "annotations": annotations,
    }
    with open(OUT_ANNO, "w") as f:
        json.dump(out, f)
    print(f"images={len(images)} annotations={len(annotations)} -> {OUT_ANNO}")


if __name__ == "__main__":
    main()
