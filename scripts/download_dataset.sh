#!/usr/bin/env bash
# Tải subset COCO val2017 cho Thí nghiệm 1 (research_plan.md §4, §13).
# Chạy: bash scripts/download_dataset.sh   (cần venv đã dựng qua scripts/setup_env.sh)
# Idempotent: bỏ qua annotation/subset/ảnh đã có sẵn — chạy lại an toàn nếu bị đứt giữa chừng.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="$REPO_ROOT/data/coco"
ANNO_ZIP_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNO_ZIP="$DATA_DIR/annotations_trainval2017.zip"
ANNO_JSON="$DATA_DIR/annotations/instances_val2017.json"
IMAGES_DIR="$DATA_DIR/val2017"
SUBSET_FILE="$REPO_ROOT/configs/coco_val2017_subset_1000.json"
SUBSET_SIZE=1000   # research_plan.md §4: subset 500-1000 ảnh cho thực nghiệm nhanh ở Thí nghiệm 1
SUBSET_SEED=42     # cố định để mọi máy GPU thuê (fresh) ra đúng cùng 1 subset — xem research_plan.md §6.5

log() { echo -e "\n[dataset] $*"; }

if [ ! -f "$REPO_ROOT/.venv/bin/activate" ]; then
  echo "[dataset][LỖI] Không tìm thấy .venv — chạy scripts/setup_env.sh trước." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

log "Kiểm tra apt dependency (unzip)..."
if ! command -v unzip >/dev/null 2>&1; then
  SUDO=""
  if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq unzip
fi

mkdir -p "$DATA_DIR" "$(dirname "$SUBSET_FILE")"

log "Tải annotation COCO val2017 (chỉ lấy instances_val2017.json từ zip, ~240MB)..."
if [ ! -f "$ANNO_JSON" ]; then
  curl -fL --retry 3 -o "$ANNO_ZIP" "$ANNO_ZIP_URL"
  unzip -o -q "$ANNO_ZIP" annotations/instances_val2017.json -d "$DATA_DIR"
  rm -f "$ANNO_ZIP"
else
  log "Đã có $ANNO_JSON, bỏ qua tải lại."
fi

log "Chọn subset $SUBSET_SIZE ảnh (seed=$SUBSET_SEED, chỉ ảnh có >=1 annotation)..."
python3 - "$ANNO_JSON" "$SUBSET_FILE" "$SUBSET_SIZE" "$SUBSET_SEED" <<'PYEOF'
import json, os, random, sys

anno_json, subset_file, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

if os.path.exists(subset_file):
    print(f"[dataset] {subset_file} đã tồn tại, bỏ qua (xoá file này nếu muốn tạo lại subset).")
    sys.exit(0)

with open(anno_json) as f:
    coco = json.load(f)

img_ids_with_anno = sorted({a["image_id"] for a in coco["annotations"]})
if len(img_ids_with_anno) < n:
    raise SystemExit(f"[dataset][LỖI] Chỉ có {len(img_ids_with_anno)} ảnh có annotation, không đủ {n}.")

# random.Random (không phải numpy) — thuật toán Mersenne Twister ổn định giữa các version Python,
# đảm bảo cùng seed ra cùng subset dù chạy trên máy/thời điểm nào.
rng = random.Random(seed)
subset_ids = sorted(rng.sample(img_ids_with_anno, n))

id_to_img = {img["id"]: img for img in coco["images"]}
subset = {
    "source_annotation_file": os.path.basename(anno_json),
    "seed": seed,
    "count": n,
    "selection_rule": "random.Random(seed).sample trên tập ảnh val2017 có >=1 annotation",
    "image_ids": subset_ids,
    "images": [
        {"id": i, "file_name": id_to_img[i]["file_name"], "coco_url": id_to_img[i]["coco_url"]}
        for i in subset_ids
    ],
}
with open(subset_file, "w") as f:
    json.dump(subset, f, indent=2)
print(f"[dataset] Đã ghi {n} ảnh vào {subset_file}")
PYEOF

log "Tải các ảnh trong subset vào $IMAGES_DIR (bỏ qua ảnh đã có, retry 3 lần mỗi ảnh)..."
mkdir -p "$IMAGES_DIR"
python3 - "$SUBSET_FILE" "$IMAGES_DIR" <<'PYEOF'
import json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

subset_file, images_dir = sys.argv[1], sys.argv[2]
with open(subset_file) as f:
    subset = json.load(f)

def fetch(img):
    dst = os.path.join(images_dir, img["file_name"])
    if os.path.exists(dst):
        return True
    tmp = dst + ".part"
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(img["coco_url"], tmp)
            os.rename(tmp, dst)
            return True
        except Exception as e:
            last_err = e
    print(f"[dataset][LỖI] {img['file_name']}: {last_err}")
    return False

todo = subset["images"]
done, failed = 0, []
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(fetch, img): img for img in todo}
    for fut in as_completed(futures):
        ok = fut.result()
        if not ok:
            failed.append(futures[fut]["file_name"])
        done += 1
        if done % 200 == 0:
            print(f"[dataset]   {done}/{len(todo)} ảnh đã xử lý...")

if failed:
    print(f"[dataset][LỖI] {len(failed)}/{len(todo)} ảnh tải thất bại sau 3 lần thử: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    print("[dataset][LỖI] Chạy lại script để thử tải tiếp các ảnh còn thiếu (idempotent).")
    sys.exit(1)
print(f"[dataset] Xong: {len(todo)}/{len(todo)} ảnh trong subset đã sẵn sàng tại {images_dir}")
PYEOF

log "Xong. Subset config (đã commit git): $SUBSET_FILE"
log "Annotation đầy đủ (không commit, xem .gitignore): $ANNO_JSON"
log "Ảnh subset (không commit): $IMAGES_DIR"
