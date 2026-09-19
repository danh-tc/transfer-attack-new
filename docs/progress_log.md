# Nhật ký tiến độ (Progress Log)

Chỉ thêm mới (append-only). Sau mỗi phiên làm việc có kết quả hoặc quyết định, thêm một entry mới có ngày ở cuối file. Không sửa lại entry cũ, trừ khi sửa lỗi sai sự kiện — nếu phân vân, gạch ngang phần cũ thay vì xóa.

---

## 2026-09-19 — Khởi tạo dự án

- Đã viết kế hoạch nghiên cứu (v1): tấn công né tránh có khả năng chuyển giao xuyên họ feature-extractor, dùng single-surrogate, cho object detection. Xem [research_plan.md](research_plan.md).
- Tổ chức lại docs: `CLAUDE.md` (root, pointer cho mỗi session) + `docs/research_plan.md` + `docs/progress_log.md` (file này) + `docs/model_registry.md`.
- Toàn bộ docs chuyển sang tiếng Việt.
- Chưa có code, chưa có môi trường (venv), chưa verify model nào.
- Bước tiếp theo: Thí nghiệm 1, danh sách việc ở research_plan.md §13 — bắt đầu từ kiểm tra repo/model registry (bước 1-3).

## 2026-09-19 — Quyết định môi trường (venv)

- Bối cảnh: chạy trên GPU thuê (RTX 3090, Ubuntu), mỗi lần thuê là máy hoàn toàn mới, không giữ lại gì giữa các phiên.
- Quyết định: dùng `venv` chuẩn (không conda) + version pin cứng, dựng qua 1 script duy nhất `scripts/setup_env.sh`. Lý do và version cụ thể: [environment_setup.md](environment_setup.md).
- Stack đã chốt: torch 2.1.2 + cu118, mmcv 2.1.0 qua `mim`, mmpretrain ≥1.2.0 (bắt buộc cho config ConvNeXt/Swin), mmdetection v3.3.0 clone editable vào `third_party/`, numpy pin `<2`.
- Thêm `.gitignore` cho `.venv/`, `third_party/`, checkpoint, dataset, output — các thứ này không commit, dựng lại mỗi lần qua script.
- Thêm cơ chế bootstrap trong CLAUDE.md: mỗi session mới phải đọc progress_log.md (entry mới nhất) + model_registry.md trước khi làm việc, vì memory riêng của Claude không tồn tại trên máy GPU mới.
- Chưa chạy thử script này trên máy GPU thật — cần verify ở lần thuê máy đầu tiên.
- Sửa: pin cứng Python 3.10 (trước đó script chỉ dùng `python3` mặc định hệ thống, không nhất quán với triết lý pin version). Script tự cài qua deadsnakes PPA nếu máy thuê không có sẵn 3.10.

## 2026-09-19 — Chạy thử `setup_env.sh` lần đầu trên máy GPU thật, sửa 4 bug

- Bối cảnh: lần đầu chạy `scripts/setup_env.sh` trên máy thuê thật (RTX 3090, driver 570.124.04). Máy có `nvidia-smi` sẵn, không cần cài driver.
- 4 bug phát hiện và sửa (script cũ chạy tới đâu lỗi tới đó, phải chạy lại nhiều lần):
  1. **`pip install -e third_party/mmdetection` lỗi `ModuleNotFoundError: No module named 'torch'`** — `setup.py` của mmdetection `import torch` ở build-time (`torch.utils.cpp_extension`), nhưng build isolation mặc định của pip cô lập khỏi site-packages nên không thấy torch đã cài trong venv. Sửa: thêm `--no-build-isolation` vào lệnh cài.
  2. **Sau khi sửa (1), lỗi mới: "missing the 'build_editable' hook"** — `openxlab` (dependency gián tiếp của `openmim`, chắc do `mim install` kéo theo) ghim `setuptools~=60.2.0`, đè lên bản upgrade ở đầu script; 60.2.0 không có PEP 660 `build_editable`. Thử upgrade lên bản mới nhất (84.0.0) thì lỗi tiếp `ModuleNotFoundError: No module named 'pkg_resources'` vì setuptools ≥81 đã bỏ hẳn `pkg_resources` mà `torch.utils.cpp_extension` cần. Sửa: ghim cứng `setuptools==69.5.1` (đủ mới cho PEP 660, chưa bỏ `pkg_resources`) ngay trước bước cài mmdetection.
  3. **numpy bị đẩy về `>=2` nhiều lần giữa script** — mmcv/mmengine (qua `mim install`) và `opencv-python-headless` bản mới nhất đều có dependency `opencv-python`/`numpy` không pin, kéo numpy lên 2.x, phá ABI của torch/mmcv (đã build cho numpy 1.x). Import vẫn "thành công" nhưng kèm `UserWarning: Failed to initialize NumPy: _ARRAY_API not found` — nguy hiểm vì đây là dạng lỗi câm lặng, không crash ngay nhưng có thể cho kết quả số sai lệch. Sửa: ghim `numpy==1.26.4` lại nhiều điểm trong script (trước mim install, sau mim install, và pin cuối cùng bằng `--force-reinstall --no-deps` ngay trước verify), đồng thời ghim `opencv-python-headless==4.10.0.84` (bản 4.x cuối cùng tương thích numpy<2) thay vì "mới nhất".
  4. **Cài `opencv-python-headless` xong rồi mới `pip uninstall opencv-python`  → hỏng `cv2`** (`AttributeError: module 'cv2' has no attribute 'COLOR_BGR2RGB'`) — `opencv-python` (bản GUI, bị mmcv/mmengine kéo theo không pin) và `opencv-python-headless` cùng ghi file vào chung thư mục `cv2/` trong site-packages; uninstall sau khi install đã xóa nhầm file mà bản headless vừa ghi đè lên (pip uninstall xóa theo RECORD của package bị gỡ, không biết file đã bị ghi đè). Sửa: đảo thứ tự — uninstall `opencv-python` trước, install `opencv-python-headless` sau.
- Đã tăng cường bước verify cuối script: không chỉ import mà chạy thật 1 op CUDA compiled của mmcv (`nms` trên tensor `.cuda()`), assert numpy version bắt đầu bằng `"1."`, và bật `warnings.simplefilter("error")` quanh `import torch` để bất kỳ lần tái phát lỗi ABI numpy nào cũng làm script fail rõ ràng thay vì trôi qua âm thầm.
- Kết quả cuối: script chạy sạch từ đầu đến cuối, verify pass — `torch 2.1.2+cu118 cuda_available=True` (RTX 3090), `numpy 1.26.4`, `mmengine 0.10.7`, `mmcv 2.1.0`, `mmdet 3.3.0`, `cv2 4.10.0`, mmcv CUDA op chạy thật OK.
- Cập nhật bảng version trong [environment_setup.md](environment_setup.md) khớp với các pin mới (setuptools, opencv-python-headless).
- Chưa có bước tải dataset (COCO) trong `scripts/` — mới chỉ cài `pycocotools` (thư viện eval), chưa tải ảnh/annotation thật. Việc này chưa được script hóa, để dành cho lúc chuẩn bị Thí nghiệm 1.
- Bước tiếp theo: research_plan.md §13 — kiểm tra MMDetection để xác minh model nào dùng backbone ResNet/ConvNeXt/Swin, cập nhật model_registry.md.

## 2026-09-19 — Thêm `scripts/download_dataset.sh`, tải subset COCO val2017

- Quyết định (đã hỏi user, không tự đoán): script tải dataset tách riêng khỏi `setup_env.sh` (lý do: setup env vs tải data là 2 việc khác nhau về bản chất — env nhanh/ổn định, data chậm/phụ thuộc mạng ngoài, không nên để 1 lỗi mạng COCO làm fail luôn bước dựng venv). Subset chốt: **1000 ảnh**, seed cố định `42`, danh sách image ID **commit vào git** tại `configs/coco_val2017_subset_1000.json` — để mọi máy GPU thuê mới (fresh) luôn ra đúng cùng 1 subset, không phụ thuộc random state có thể khác nhau giữa các lần/version Python.
- Thiết kế: chỉ tải `instances_val2017.json` từ zip annotation gốc (không cần captions/keypoints), dùng nó để lọc ra các ảnh có ≥1 annotation rồi `random.Random(seed).sample` (không dùng `numpy.random` — thuật toán Mersenne Twister của `random` chuẩn Python ổn định hơn giữa các version). Sau đó chỉ tải đúng 1000 ảnh trong subset qua `coco_url` của từng ảnh (không tải nguyên zip val2017 ~1GB/5000 ảnh) — nhanh hơn nhiều cho việc setup lại trên máy thuê mới mỗi lần. Script idempotent, retry 3 lần/ảnh, báo lỗi rõ và thoát non-zero nếu còn ảnh thiếu sau retry (không âm thầm chạy tiếp với subset thiếu).
- Đã chạy thử thật trên máy GPU thuê: tải annotation (241MB) + 1000 ảnh (~177MB) thành công 100%, verify lại bằng `pycocotools` — subset load đúng 1000 ảnh, 7496 annotation, không thiếu file. Chạy lại lần 2 để kiểm tra idempotent: bỏ qua toàn bộ, xong trong 0.13s.
- Cập nhật `CLAUDE.md` mục "Chạy môi trường" để trỏ tới script mới.
- Bước tiếp theo: vẫn là research_plan.md §13 — verify model registry (ResNet/ConvNeXt/Swin) trong MMDetection. Dataset giờ đã sẵn sàng để dùng khi tới bước eval.

## 2026-09-19 — Verify model registry, pivot surrogate/target sang Mask R-CNN

- Verify trực tiếp (đọc config thật, không đoán từ tên file) trong `third_party/mmdetection` (v3.3.0): `configs/faster_rcnn/` chỉ có backbone ResNet (r50/r101/x101), **không có** Faster R-CNN + ConvNeXt hay Faster R-CNN + Swin. `configs/convnext/` chỉ có Mask R-CNN / Cascade Mask R-CNN. `configs/swin/` chỉ có Mask R-CNN / RetinaNet. Kết luận: plan gốc (research_plan.md §6.1–§6.2, toàn bộ Faster R-CNN) không khả thi với MMDetection có sẵn.
- Đã hỏi user, chọn phương án: **đổi cả surrogate lẫn target sang Mask R-CNN** (thay vì giữ Faster R-CNN cho surrogate+R101 rồi lẫn Mask R-CNN cho ConvNeXt/Swin). Lý do chọn: phương án trộn 2 kiến trúc detector sẽ tạo confound đúng ngay lúc đổi backbone family (R101→ConvNeXt/Swin), vi phạm nguyên tắc "chỉ đổi feature extractor" mà §6.2 đề ra cho *toàn bộ* target, không chỉ 1/3. Đã so sánh trực tiếp `configs/_base_/models/faster-rcnn_r50_fpn.py` vs `mask-rcnn_r50_fpn.py`: nhánh box (rpn_head, bbox_head, loss) giống hệt nhau, Mask R-CNN chỉ thêm mask_head song song không tham gia box prediction — nên dùng box output của Mask R-CNN coi như tương đương Faster R-CNN cho mục đích thí nghiệm này là hợp lý.
- Đã cập nhật research_plan.md §6.1, §6.2 phản ánh pivot này (kèm lý do ngay trong văn bản, theo đúng quy ước "chỉ sửa khi có pivot có chủ đích").
- Bộ model kiểm soát tối thiểu cho Thí nghiệm 1, đã điền đầy đủ vào [model_registry.md](model_registry.md) (config path, checkpoint URL, box AP công bố, ngày verify):
  - Surrogate: Mask R-CNN + ResNet-50 (`mask-rcnn_r50_fpn_1x_coco.py`, box AP 38.2)
  - Target same-family: Mask R-CNN + ResNet-101 (`mask-rcnn_r101_fpn_1x_coco.py`, box AP 40.0)
  - Target cross-CNN-family: Mask R-CNN + ConvNeXt-Tiny (`mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py`, box AP 46.2)
  - Target CNN→Transformer: Mask R-CNN + Swin-Tiny (`mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py`, box AP 42.7)
- Vấn đề mở quan trọng (ghi trong model_registry.md): checkpoint ConvNeXt-Tiny chính thức duy nhất dùng schedule 3x+AMP+ms-crop, trong khi R50/R101/Swin đều 1x — chênh lệch training recipe nằm ngoài kiểm soát thí nghiệm (không có bản 1x thay thế trong MMDetection), cần nhớ khi diễn giải kết quả nếu ConvNeXt transfer tốt bất thường.
- Chưa tải checkpoint nào về máy, chưa chạy inference/eval thật — mới dừng ở xác minh + chọn bộ model trên giấy (đúng phạm vi research_plan.md §13 bước 1-4).
- Bước tiếp theo: research_plan.md §13 bước 5 trở đi — tìm/tái sử dụng cài đặt tấn công chuyển giao đã có (baseline), tải 4 checkpoint trên, viết eval pipeline (clean mAP / adversarial mAP / ASR) trên subset COCO đã tải.

## 2026-09-19 — Cài đặt attack + eval pipeline Thí nghiệm 1, phát hiện & sửa 2 bug quan trọng

- Tải xong 4 checkpoint đã chọn trong model_registry.md vào `checkpoints/` (không commit). Verify cả 4 load đúng qua `init_detector` (backbone type khớp: ResNet/ResNet/ConvNeXt/SwinTransformer).
- Viết `scripts/build_subset_annotations.py`: lọc `instances_val2017.json` về đúng 1000 ảnh trong subset (sinh `data/coco/annotations/instances_val2017_subset1000.json`, không commit — dẫn xuất deterministic từ 2 file đã có).
- Baseline attack: **MI-FGSM** (Dong et al. 2018 — momentum, L_inf, đã kiểm chứng rộng rãi cho transferability, đúng tinh thần research_plan.md §6.3/§8 "cài đặt test đơn giản nhất"). Code ở `attacks/mi_fgsm.py`.
- **Bug 1 (quan trọng, phát hiện qua thực nghiệm chứ không phải đọc code)**: bản đầu tiên của attack tối đa hóa TOÀN BỘ loss huấn luyện của surrogate (loss_rpn_cls/bbox + loss_cls/bbox của RCNN). Kết quả thực tế: KHÔNG tạo ra evasion — thay vào đó tạo hàng loạt false-positive confidence ~1.0 (ảnh test: 8 detection sạch → 54 detection sau tấn công). Lý do: phần lớn khối lượng loss đến từ anchor/proposal nền (background), gradient ascent "ăn gian" bằng cách biến nền thành vật thể giả (dễ tăng loss hơn nhiều so với việc thật sự làm vật thể biến mất, vì background chiếm số lượng áp đảo). Đây là hiện tượng gần với "TOG-fabrication" (Chow et al. 2020), không phải "né tránh" như research_plan.md yêu cầu (đúng nghĩa "evasion" trong literature là vật thể biến mất, không phải sinh vật thể giả).
  - **Fix**: đổi objective sang kiểu DAG (Xie et al. 2017) — chỉ tính cross-entropy trên đúng RoI của GT box (dùng thẳng `bbox_roi_extractor`+`bbox_head` có sẵn của mmdet, bỏ qua RPN/sampler), maximize CE của đúng nhãn GT tại đúng vị trí GT. Không có RoI nền nào tham gia nên không còn đường "ăn gian". Verify lại: số detection giảm đúng như kỳ vọng evasion (vd ảnh test: 9→4 detection), không còn flood false-positive.
  - Bài học: **tăng loss huấn luyện tổng ≠ evasion** cho detector 2-stage — cần nhắm đúng vào vị trí object thật. Ghi lại để không lặp lại sai lầm này nếu sau này đổi/mở rộng attack.
- **Bug 2 (quan trọng, làm dry-run "15 ảnh" thực chất chạy nhầm 1000 ảnh)**: `experiments/experiment1.py` giới hạn số ảnh bằng cách gán đè `ds.data_list = ds.data_list[:limit]` sau khi dataset đã build — KHÔNG có tác dụng, vì mmengine `BaseDataset` mặc định `serialize_data=True`, serialize toàn bộ `data_list` thành byte buffer ngay trong `__init__`/`full_init()`; gán đè attribute sau đó không ảnh hưởng gì tới `__getitem__`/`__len__` thật. Hậu quả: chạy "dry-run 15 ảnh" tưởng bị treo/chậm bất thường (>10 phút chưa xong), phải dừng lại dùng `nvidia-smi pmon`/profile từng phần riêng lẻ (model load, predict, attack, dataset `__getitem__`) mới lần ra — từng phần đều nhanh (model load <1s, predict ~0.05-0.4s, attack 10 iter ~0.5s, dataset access ~0.02s/ảnh), tổng không khớp với >10 phút quan sát được → nghi ngờ và verify lại số ảnh thực tế đang chạy, phát hiện log in ra "1000 images" dù truyền limit=15.
  - **Fix**: bỏ cách gán đè dataset, giới hạn bằng `range(min(limit, len(ds)))` ở vòng lặp gọi, không đụng vào dataset object.
  - Sau khi fix: dry-run 15 ảnh thật chạy ~1.24 ảnh/s, đúng như kỳ vọng.
- Dry-run 15 ảnh (mẫu rất nhỏ, chỉ sanity-check pipeline, KHÔNG phải kết quả chính thức) đã cho đúng xu hướng research_plan.md §6.6 dự đoán: ASR surrogate=0.343 > target_r101 (same-family)=0.132 > target_convnext_t (cross-CNN)=0.099 > target_swin_t (CNN→Transformer)=0.079.
- Full run 1000 ảnh × 4 model hoàn tất, ~13 phút (1.3 ảnh/s). Kết quả đầy đủ: `outputs/experiment1/results.json` (không commit).

## 2026-09-19 — Kết quả Thí nghiệm 1 (full 1000 ảnh): transfer gap rõ ràng, đúng giả thuyết

Config: epsilon=8.0 (thang pixel 0-255, tương đương 8/255), MI-FGSM num_iter=10, decay=1.0, score_thr=0.3, iou_thr=0.5 (xem attacks/mi_fgsm.py, experiments/experiment1.py). 1000 ảnh subset cố định (configs/coco_val2017_subset_1000.json).

| Model | Họ backbone | Clean AP50 | Adv AP50 | ΔAP50 | Clean-correct objects | Evaded | ASR |
|---|---|---|---|---|---|---|---|
| Mask R-CNN + ResNet-50 (surrogate) | ResNet | 0.5843 | 0.3382 | −0.2460 | 4994 | 1647 | 0.3298 |
| Mask R-CNN + ResNet-101 (same-family) | ResNet | 0.6014 | 0.5062 | −0.0952 | 5085 | 811 | 0.1595 |
| Mask R-CNN + ConvNeXt-Tiny (cross-CNN) | ConvNeXt | 0.6864 | 0.6364 | −0.0500 | 5623 | 511 | 0.0909 |
| Mask R-CNN + Swin-Tiny (CNN→Transformer) | Swin | 0.6616 | 0.6157 | −0.0459 | 5418 | 415 | 0.0766 |

(clean_AP — COCO AP chuẩn IoU 0.50:0.95 — và full breakdown theo area cũng có trong results.json, không chỉ AP50 ở trên.)

**Quan sát chính**:
- ASR giảm đơn điệu đúng thứ tự research_plan.md §6.6 dự đoán: same-family (0.1595) > cross-CNN-family (0.0909) > CNN→Transformer (0.0766). ΔAP50 cũng giảm đơn điệu cùng thứ tự (−0.0952 > −0.0500 > −0.0459).
- TransferGap (định nghĩa §5, ASR_same-family − ASR_cross-family): so ConvNeXt = 0.1595−0.0909 = **0.0686**; so Swin = 0.1595−0.0766 = **0.0829**. Cả hai dương và tăng theo khoảng cách kiến trúc — CNN→Transformer mất khả năng chuyển giao nhiều hơn CNN→CNN-khác-họ.
- Mẫu đủ lớn để tin cậy: 1000 ảnh, 4994–5623 object detect đúng ở điều kiện sạch mỗi model (không phải vài chục như dry-run) — pattern giữ nguyên và rõ hơn so với dry-run 15 ảnh (dry-run: 0.343/0.132/0.099/0.079 — cùng thứ tự, khớp hướng).

**Kết luận (research_plan.md §13 mục 10 + §6.7)**: quan sát được transfer gap xuyên họ feature-extractor **rõ ràng và có hệ thống**, không phải nhiễu ngẫu nhiên — tấn công single-surrogate chuyển giao yếu dần khi backbone target càng khác surrogate về bản chất kiến trúc, suy giảm mạnh nhất khi CNN→Transformer. Theo quy tắc quyết định §6.7 ("nếu transfer gap rõ ràng → tiến hành Thí nghiệm 2"), tiền đề cốt lõi của nghiên cứu đã được xác lập trên thiết lập single-surrogate có kiểm soát.

**Giới hạn cần lưu ý khi diễn giải**: (1) caveat training-recipe của ConvNeXt-Tiny đã ghi trong model_registry.md (checkpoint 3x+AMP+ms-crop, mạnh hơn 3 model còn lại dùng 1x) — có thể một phần lý do ConvNeXt có ASR thấp hơn Swin là do model này vốn "khỏe" hơn (clean AP50 cao nhất, 0.686) chứ không chỉ do họ backbone; (2) attack chỉ tấn công classification tại RoI của GT (không tấn công localization/bbox regression), nên ASR đo ở đây là "mất đúng nhãn tại đúng vị trí", chưa phản ánh đầy đủ mọi kiểu evasion có thể có; (3) baseline mới chỉ có 1 attack (MI-FGSM/DAG-style) — chưa biết mức độ transfer gap này có giữ nguyên với attack khác hay không.
- Bước tiếp theo: theo research_plan.md §7 (Thí nghiệm 2 — tìm cơ chế: gradient alignment, feature similarity, saliency, transformation consistency) hoặc trước tiên có thể muốn kiểm tra lại giới hạn (1) và (2) ở trên để chắc kết quả không bị confound bởi training recipe/attack objective trước khi đầu tư vào Thí nghiệm 2.

## 2026-09-19 — HANDOFF: Thí nghiệm 1B CHƯA CHẠY (bị dừng chủ động), mai phải chạy lại từ đầu

**Đọc entry này đầu tiên nếu bắt đầu phiên mới.**

### Trạng thái thật: KHÔNG có kết quả nào để đọc

`python3 experiments/experiment1b.py` đã được chạy nền rồi **bị dừng chủ động (kill) theo yêu cầu user** khi mới tới `runB_mi_fgsm_zoo` 100/1000 ảnh (~9% tổng tiến độ 4 run) — không phải crash, không phải chạy xong. `outputs/experiment1b/results.json` **không tồn tại** (chưa kịp ghi lần nào — script chỉ ghi file này sau khi 1 run trong 4 run hoàn tất, chưa run nào xong). `outputs/experiment1b/run_full.log` có nhưng chỉ là log dở dang, không dùng được.

**Việc cần làm khi resume: chạy lại từ đầu, không có gì để "tiếp tục" hay resume dở dang cả** (script không có checkpoint/cache giữa chừng):
```bash
cd /workspace/transfer-attack-new
source .venv/bin/activate
rm -f outputs/experiment1b/run_full.log outputs/experiment1b/results.json
nohup python3 -u experiments/experiment1b.py > outputs/experiment1b/run_full.log 2>&1 &
```
Ước tính ~60-70 phút cho cả 4 run (đo từ 1 lần chạy thật trước đó: runB ~1.08 ảnh/s ở 100 ảnh đầu, dự kiến chậm hơn Thí nghiệm 1 gốc vì runB có 6 model thay vì 4).

- Nếu máy GPU thuê đã đổi sang máy mới hoàn toàn (theo CLAUDE.md — luôn giả định vậy trừ khi chắc chắn là cùng máy): phải chạy lại từ đầu `bash scripts/setup_env.sh` → `bash scripts/download_dataset.sh` → tải lại checkpoint (xem lệnh `curl` trong entry "Kết quả Thí nghiệm 1" và "Cài đặt attack pipeline" phía trên, hoặc list trong `docs/model_registry.md` + 2 checkpoint mới ResNeXt-101/R50-3x) → `python3 scripts/build_subset_annotations.py` → rồi mới chạy experiment1b.

### Vì sao có Thí nghiệm 1B

User review kết quả Thí nghiệm 1 (transfer gap rõ ràng, xem entry phía trên), chỉ ra claim hiện tại chỉ an toàn ở mức "cross-family transfer gap tồn tại trong 1 thiết lập cụ thể", CHƯA đủ để nói "khoảng cách kiến trúc LÀ NGUYÊN NHÂN" — còn 3 confound (đúng 3 điều tôi tự flag ở cuối entry Thí nghiệm 1). User đề xuất, đã thống nhất scope qua AskUserQuestion (chọn "Recommended" cho model zoo — thêm 2-3 model cùng họ ResNet, khác độ mạnh):

- **Run B** (model-strength confound): model zoo mở rộng — thêm Mask R-CNN + ResNeXt-101-32x4d (box AP 41.9) và Mask R-CNN + ResNet-50 train 3x+ms (box AP 40.9, CÙNG backbone surrogate, chỉ train mạnh hơn) — checkpoint đã tải vào `checkpoints/`, verify load OK. Cùng attack MI-FGSM cls-only như Thí nghiệm 1, chạy trên 6 model.
- **Run C** (attack robustness): BIM/I-FGSM (bỏ momentum, `decay=0.0`) thay MI-FGSM, trên 4 model gốc.
- **Run D** (objective robustness): thêm loss hồi quy bbox vào objective attack (không chỉ classification) — `objective="cls_bbox"`.
- **Run E** (augmentation-based transfer, baseline "mạnh hơn" theo research_plan.md §11): DI-FGSM (input diversity — resize+pad ngẫu nhiên mỗi iter) — `input_diversity=True`.

Code: `attacks/detection_attacks.py` (tổng quát hóa từ `attacks/mi_fgsm.py`, giữ nguyên file cũ vì `experiment1.py` vẫn dùng làm baseline chính — không xóa để tránh phải re-run Thí nghiệm 1). `experiments/common.py` (hạ tầng dùng chung, tách từ `experiment1.py`). `experiments/experiment1b.py` (orchestrator 4 run).

Đã verify bằng dry-run 5 ảnh trước khi chạy full — cả 4 run chạy hết không lỗi, pattern sơ bộ (5 ảnh, KHÔNG đủ tin cậy, chỉ sanity-check): runB target_r50_3x (cùng backbone surrogate, train mạnh hơn) có ASR=0.250 — thấp hơn target_r101 (0.154)? gần bằng nhau trong mẫu bé — **chưa kết luận được gì, chờ full 1000 ảnh**.

### Nguồn gốc động lực Exp1B (bằng chứng độc lập)

User trích 1 paper: arXiv 2602.16494 "Benchmarking Adversarial Robustness and Adversarial Training Strategies for Object Detection" (Winter et al., nộp 18/2/2026) — đã tự verify bằng WebFetch, **citation thật**, báo cáo đúng hiện tượng "modern adversarial attacks... significant lack of transferability to transformer-based architectures", khớp trực tiếp với phát hiện của Thí nghiệm 1. Nên trích dẫn vào research_plan.md khi viết lại phần liên quan (§9-§11) — CHƯA làm, còn để ngỏ.

### Trạng thái git — CHƯA COMMIT GÌ trong toàn bộ session này

`git status --short` tại thời điểm viết entry này:
```
M  CLAUDE.md
A  attacks/detection_attacks.py
A  attacks/mi_fgsm.py
A  configs/coco_val2017_subset_1000.json
M  docs/environment_setup.md
M  docs/model_registry.md
M  docs/progress_log.md
M  docs/research_plan.md
A  experiments/common.py
A  experiments/experiment1.py
A  experiments/experiment1b.py
A  scripts/build_subset_annotations.py
A  scripts/download_dataset.sh
M  scripts/setup_env.sh
```
User chưa yêu cầu commit lần nào trong session — theo quy tắc "chỉ commit khi được yêu cầu rõ ràng", nên toàn bộ vẫn nằm ở working tree. **Nếu bắt đầu phiên mới trên đúng máy này (working tree còn nguyên) thì không cần làm lại gì, chỉ cần check kết quả Exp1B.** Nếu máy mới hoàn toàn (mất working tree, chỉ còn nhánh `main` trên remote) thì TOÀN BỘ code phía trên (attacks/, experiments/, scripts/download_dataset.sh, scripts/build_subset_annotations.py) đã MẤT — phải hỏi user xem có bản backup/patch nào không trước khi viết lại từ đầu, đừng giả định có thể tái tạo y hệt.

### Việc tiếp theo sau khi Exp1B có kết quả

1. Đọc `outputs/experiment1b/results.json`, so sánh 4 run với baseline Thí nghiệm 1 (`outputs/experiment1/results.json`):
   - Run B: vẽ/so sánh clean AP vs ASR trong nhóm same-family (surrogate R50, R101, ResNeXt101, R50-3x) — xem có tương quan rõ không.
   - Run C/D/E: xem thứ tự same-family > cross-CNN > CNN→Transformer có giữ nguyên không.
2. Viết entry progress_log.md mới tổng kết Exp1B (theo đúng format entry "Kết quả Thí nghiệm 1" phía trên).
3. Cập nhật research_plan.md nếu kết luận Exp1B củng cố hoặc làm yếu claim "khoảng cách kiến trúc là nguyên nhân" — và trích arXiv 2602.16494 ở đây.
4. Hỏi user có commit toàn bộ session không trước khi coi như xong.
5. Sau đó mới sang Thí nghiệm 2 (research_plan.md §7).
