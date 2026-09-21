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

> **⚠️ SUPERSEDED (2026-09-20)**: toàn bộ số liệu ASR/AP dưới đây tính từ attack có bug tọa độ RoI (GT bbox chưa scale theo `scale_factor`, xem entry "Phát hiện + fix bug tọa độ RoI" bên dưới). Attack thật yếu hơn nhiều so với thiết kế (whitebox ASR ~33% thay vì ~95% sau fix). Xu hướng định tính (same-family > cross-CNN > CNN→Transformer) vẫn đúng, nhưng KHÔNG dùng số liệu cụ thể ở entry này để trích dẫn — lấy ở entry sau ngày fix.

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

## 2026-09-20 — Resume trên máy GPU thuê mới (RTX 4000 Ada), rerun Thí nghiệm 1B, 2 quyết định quy trình

- Máy GPU thuê lần này khác máy cũ (RTX 4000 Ada Generation, driver 565.57, thay vì RTX 3090 driver 570.124.04 ở entry trước) — đúng như CLAUDE.md giả định, "máy mới hoàn toàn" mỗi lần thuê. `.venv` và `third_party/mmdetection` đã có sẵn (build từ trước trong phiên này), nhưng `data/coco/` và `checkpoints/` (gitignore, không commit) đã mất — tải lại dataset (`scripts/download_dataset.sh` — user đã tự chạy), tải lại 6 checkpoint (4 gốc + 2 extra ResNeXt-101/R50-3x, URL verify lại qua `third_party/mmdetection/configs/mask_rcnn/metafile.yml`, khớp box AP công bố), build lại `instances_val2017_subset1000.json` (ra đúng 1000 ảnh/7496 annotation, khớp lần trước — xác nhận subset cố định hoạt động đúng như thiết kế). Verify cả 6 checkpoint load đúng backbone qua `init_detector` trước khi chạy.
- Đã khởi động lại `experiments/experiment1b.py` full (theo đúng lệnh trong entry HANDOFF phía trên) — đang chạy nền lúc viết entry này, Run B + Run C đã xong, Run D đang chạy.
- **Quyết định 1 — thêm delta % vào summary**: `clean_AP`/`clean_AP50` chênh lệch khá lớn giữa các model (vd ConvNeXt 0.686 vs surrogate 0.584), nên delta tuyệt đối (`delta_AP50` cũ) có thể đánh giá sai mức "tàn phá tương đối" khi so các model có baseline khác nhau. Thêm `common.summarize_model_result()` (gộp logic build summary, dùng chung cho experiment1.py/experiment1b.py, trước đây lặp code ở 2 nơi) — tính thêm `delta_AP`, `delta_AP_pct`, `delta_AP50_pct` (= delta / clean * 100). Lưu ý quan trọng: sửa code này **sau khi** experiment1b.py đã chạy (process giữ nguyên code cũ trong RAM từ lúc import) — nên toàn bộ 4 run của lần chạy này sẽ thiếu 2 field `_pct` dù code đã có, phải patch tay `outputs/experiment1b/results.json` sau khi script chạy xong hẳn (đã patch tạm cho Run B/C ngay sau khi mỗi run xong, nhưng sẽ bị ghi đè mất khi Run D/E hoàn tất và script tự dump lại toàn bộ `all_results` từ RAM — cần patch lại lần cuối, đầy đủ 4 run, sau khi process kết thúc).
- **Quyết định 2 — quy ước cỡ mẫu 50/300/1000** (đã ghi vào CLAUDE.md mục Quy ước): 50 ảnh = quick test (sanity-check code, không có ý nghĩa thống kê, không ghi log như kết luận); 300 ảnh = confirm (300 ảnh **đầu** của subset 1000 cố định, tập con lồng nhau để so sánh trực tiếp được — được phép dùng làm milestone thật, kể cả ra quyết định theo §6.7, nhưng phải ghi rõ n=300 kèm caveat "sơ bộ, cần đối chiếu ở n=1000"); 1000 ảnh = final, dùng cho số liệu chính thức/paper.
- **Quyết định 3 — tmux thay nohup+disown cho background run dài**: user lo ngại pod GPU rental bị ngắt kết nối giữa chừng (khác vụ Exp1B bị kill chủ động trước đây). Đã xác nhận: cả tmux lẫn nohup+disown đều chặn được `SIGHUP` khi mất SSH/VSCode remote như nhau — không cái nào chống được việc provider tắt hẳn VM/container. Máy này chạy Docker (`docker-init` PID 1, không có systemd) nên không có rủi ro kiểu systemd-logind kill hết process khi user logout — không phải lo thêm. Chọn tmux làm mặc định cho các lần sau vì tiện attach lại xem tiến độ/tương tác trực tiếp, dù không giảm rủi ro pod chết. Đã thêm `tmux` vào `scripts/setup_env.sh` (apt install) + bảng version `environment_setup.md`, cài luôn trên máy hiện tại. Run experiment1b.py hiện tại vẫn để nguyên nohup (không đáng để kill giữa chừng chỉ để đổi cách chạy).
- Bước tiếp theo: chờ Run D/E xong, patch lại `results.json` đầy đủ (kèm field `_pct`), rồi viết entry tổng kết Thí nghiệm 1B đầy đủ theo kế hoạch đã ghi ở entry HANDOFF phía trên.

## 2026-09-20 — Kết quả Thí nghiệm 1B (full 1000 ảnh, cả 4 run): transfer gap đứng vững qua toàn bộ confound-check

> **⚠️ SUPERSEDED (cùng ngày, phát hiện sau)**: toàn bộ số liệu ASR/AP/TransferGap dưới đây tính từ attack có bug tọa độ RoI, xem entry "Phát hiện + fix bug tọa độ RoI" bên dưới. Kết luận định tính (transfer gap đứng vững qua 4 confound) vẫn đúng và còn rõ hơn sau fix, nhưng KHÔNG dùng số liệu cụ thể ở entry này để trích dẫn.

Cả 4 run đã chạy xong full 1000 ảnh (subset cố định, `configs/coco_val2017_subset_1000.json`), `outputs/experiment1b/results.json` đã patch đầy đủ field `delta_AP`, `delta_AP_pct`, `delta_AP50_pct` (xem entry phía trên về quyết định thêm field %). Config chung: epsilon=8.0 (pixel scale), num_iter=10 — giống hệt Thí nghiệm 1. Code: `attacks/detection_attacks.py` (`iterative_linf_attack`), `experiments/experiment1b.py`.

### Run B — model-strength confound (model zoo 6 model, MI-FGSM cls-only, decay=1.0)

| Model | Family | Clean AP50 | Adv AP50 | ΔAP50% | Clean mAP | Adv mAP | ΔmAP% | ASR |
|---|---|---|---|---|---|---|---|---|
| surrogate_r50 | surrogate | 0.5838 | 0.3399 | −41.8% | 0.3906 | 0.2176 | −44.3% | 0.3258 |
| target_r101 | same-family | 0.6013 | 0.5068 | −15.7% | 0.4077 | 0.3324 | −18.5% | 0.1606 |
| target_x101 (ResNeXt) | same-family | 0.6291 | 0.5326 | −15.3% | 0.4345 | 0.3582 | −17.6% | 0.1522 |
| target_r50_3x | same-family | 0.6174 | 0.4261 | −31.0% | 0.4198 | 0.2792 | −33.5% | 0.2620 |
| target_convnext_t | cross-cnn-family | 0.6864 | 0.6316 | −8.0% | 0.4757 | 0.4298 | −9.7% | 0.0911 |
| target_swin_t | cnn-to-transformer | 0.6616 | 0.6109 | −7.7% | 0.4429 | 0.4030 | −9.0% | 0.0795 |

Quan sát mấu chốt: trong cùng nhóm same-family, `target_r50_3x` (CÙNG backbone R50 với surrogate, chỉ train recipe mạnh hơn — 3x+ms) có ASR=0.2620, **cao hơn hẳn** r101 (0.1606) và x101 (0.1522), dù clean AP50 của nó (0.6174) nằm giữa 2 model kia. Model-strength (đo bằng clean AP) **không** đơn điệu với ASR trong cùng 1 họ backbone → loại được giả thuyết "chỉ cần model mạnh hơn là transfer kém hơn" như nguyên nhân chính; ủng hộ backbone family là yếu tố chi phối.

### Run C — attack robustness (BIM/I-FGSM, không momentum, decay=0.0, 4 model gốc)

| Model | Family | Clean AP50 | Adv AP50 | ΔAP50% | ASR |
|---|---|---|---|---|---|
| surrogate_r50 | surrogate | 0.5838 | 0.3509 | −39.9% | 0.3230 |
| target_r101 | same-family | 0.6013 | 0.5421 | −9.8% | 0.1123 |
| target_convnext_t | cross-cnn-family | 0.6864 | 0.6677 | −2.7% | 0.0446 |
| target_swin_t | cnn-to-transformer | 0.6616 | 0.6400 | −3.3% | 0.0436 |

Bỏ momentum làm ASR tụt ở mọi model (đúng kỳ vọng — momentum giúp transfer tốt hơn), nhưng thứ tự same (0.112) > cross-CNN (0.045) ≈ CNN→Transformer (0.044) vẫn giữ. Đáng chú ý: ConvNeXt và Swin gần bằng nhau ở BIM (khác Run B/MI-FGSM nơi Swin thấp hơn ConvNeXt rõ) — dấu hiệu cho thấy khoảng cách ConvNeXt-vs-Swin (trong cùng nhóm cross-family) nhạy với attack hơn là khoảng cách same-vs-cross, cái sau ổn định hơn nhiều.

### Run D — objective robustness (MI-FGSM, cls+bbox, 4 model gốc)

| Model | Family | Clean AP50 | Adv AP50 | ΔAP50% | ASR |
|---|---|---|---|---|---|
| surrogate_r50 | surrogate | 0.5838 | 0.3378 | −42.1% | 0.3288 |
| target_r101 | same-family | 0.6013 | 0.5035 | −16.3% | 0.1583 |
| target_convnext_t | cross-cnn-family | 0.6864 | 0.6326 | −7.8% | 0.0873 |
| target_swin_t | cnn-to-transformer | 0.6616 | 0.6166 | −6.8% | 0.0814 |

Gần như trùng khớp Run B (cls-only): thêm loss hồi quy bbox vào objective không đổi pattern hay đổi đáng kể độ lớn ASR. Kết luận: transfer gap không phải hệ quả của việc attack "chỉ" nhắm classification — vẫn xuất hiện khi tấn công cả localization.

### Run E — augmentation-based transfer (DI-FGSM, input diversity, 4 model gốc)

| Model | Family | Clean AP50 | Adv AP50 | ΔAP50% | ASR |
|---|---|---|---|---|---|
| surrogate_r50 | surrogate | 0.5838 | 0.3796 | −35.0% | 0.2841 |
| target_r101 | same-family | 0.6013 | 0.4860 | −19.2% | 0.1878 |
| target_convnext_t | cross-cnn-family | 0.6864 | 0.6017 | −12.3% | 0.1313 |
| target_swin_t | cnn-to-transformer | 0.6616 | 0.5970 | −9.8% | 0.1080 |

Baseline "mạnh hơn" theo research_plan.md §11 (input diversity): ASR tăng đáng kể ở **mọi** target so với 3 run kia (kể cả so ASR surrogate — DI-FGSM đánh đổi 1 phần sức tấn công trên chính surrogate để lấy transfer tốt hơn, đúng lý thuyết gốc của Xie et al. 2019), nhưng thứ tự same (0.188) > cross-CNN (0.131) > CNN→Transformer (0.108) **vẫn giữ nguyên**. Đây là run có TransferGap tuyệt đối nhỏ nhất tính theo ASR (0.057 so ConvNeXt) dù ASR tuyệt đối cao nhất — tăng cường transferability nói chung (qua augmentation) có vẻ thu hẹp phần nào khoảng cách xuyên họ, nhưng không xóa bỏ nó.

### TransferGap (định nghĩa research_plan.md §5, ASR_same-family[r101] − ASR_cross-family) qua 4 run

| Run | Gap vs ConvNeXt | Gap vs Swin |
|---|---|---|
| B (MI-FGSM, baseline lặp lại) | 0.0695 | 0.0811 |
| C (BIM) | 0.0677 | 0.0687 |
| D (cls+bbox) | 0.0710 | 0.0769 |
| E (DI-FGSM) | 0.0565 | 0.0798 |
| *(đối chiếu Thí nghiệm 1 gốc)* | *0.0686* | *0.0829* |

Run B (cùng model, cùng attack config với Thí nghiệm 1 gốc, chỉ khác implementation — dùng `attacks/detection_attacks.py` tổng quát thay vì `attacks/mi_fgsm.py` cũ) cho số gần như trùng khớp Thí nghiệm 1 (0.0695/0.0811 vs 0.0686/0.0829) — xác nhận 2 cài đặt tương đương, không có bug lệch giữa bản cũ và bản tổng quát hóa.

**TransferGap dương và có ý nghĩa ở cả 4 run, không phụ thuộc vào việc đổi model-zoo/attack/objective/augmentation.**

### Kết luận tổng thể

Cả 3 confound mà Thí nghiệm 1 tự flag (model-strength, attack choice, attack objective) — cộng thêm 1 kiểm tra bổ sung (augmentation-based attack mạnh hơn, Run E) — đều **không xóa bỏ hoặc đảo ngược** được transfer gap xuyên họ backbone quan sát ở Thí nghiệm 1. Claim "khoảng cách chuyển giao xuyên feature-extractor gắn với kiến trúc backbone, không phải do model mạnh/yếu hay lựa chọn attack cụ thể" giờ có cơ sở vững hơn nhiều so với chỉ dựa vào Thí nghiệm 1 gốc.

**Giới hạn còn lại (chưa giải quyết)**:
- Vẫn còn confound training-recipe của ConvNeXt-Tiny (schedule 3x+AMP+ms-crop, mạnh hơn 1x của các model khác) đã ghi trong model_registry.md — Run B cho thấy model-strength không đơn điệu trong nhóm ResNet, nhưng KHÔNG trực tiếp loại được confound này cho riêng ConvNeXt vì không có bản 1x để so sánh.
- Cả 4 run đều dùng đúng 1 subset 1000 ảnh, 1 seed — chưa có nhiều seed/nhiều subset để báo cáo mean/std (research_plan.md §6.5 khuyến nghị nếu attack có tính ngẫu nhiên; DI-FGSM ở Run E có random resize/pad nên về nguyên tắc nên chạy lại với seed khác để chắc chắn, chưa làm).
- arXiv 2602.16494 (Winter et al., trích ở entry HANDOFF phía trên) vẫn CHƯA được trích dẫn chính thức vào research_plan.md §9-§11 — còn nợ.

### Việc tiếp theo

1. ~~Cập nhật `research_plan.md` phản ánh kết luận Exp1B (transfer gap đứng vững qua 4 confound-check) + trích arXiv 2602.16494.~~ Đã làm — xem entry dưới.
2. Hỏi user có commit toàn bộ session (Exp1 + Exp1B + code + docs) không — user tự làm.
3. Sau đó chuyển sang Thí nghiệm 2 (research_plan.md §7 — tìm cơ chế: gradient alignment, feature similarity, saliency, transformation consistency).

## 2026-09-20 — Khóa RQ1, chốt thiết kế Thí nghiệm 2A (gradient alignment)

- Đã cập nhật `research_plan.md`:
  - **RQ1 khóa** (§12): "Cross-family transfer gap là hiện tượng ổn định trong setting hiện tại, không phụ thuộc riêng vào attack, objective hay clean model strength." Diễn đạt cố ý theo hướng "strong evidence that backbone architecture contributes to the transfer gap" — KHÔNG claim quan hệ nhân quả tuyệt đối ("is the cause"), vì confound training-recipe ConvNeXt-Tiny chưa cô lập và chưa test nhiều seed.
  - **§6.8 mới**: tóm tắt Thí nghiệm 1B (4 run, kết luận, giới hạn), trích arXiv:2602.16494 (Winter et al.) làm bằng chứng độc lập.
  - **§7.A mở rộng**: chốt thiết kế Thí nghiệm 2A — gradient alignment làm bước đầu tiên (rẻ nhất). Điểm quan trọng: đo cosine similarity(g_s, g_t) **per-image/per-object**, so distribution giữa nhóm evaded vs not-evaded — KHÔNG chỉ tương quan ở mức 3 điểm (3 target model), vì quá ít điểm để kết luận thống kê được. Rẽ nhánh: nếu gradient alignment giải thích được gap → xuống feature-level (layer nào gây divergence); nếu yếu → nhảy sang object evidence/saliency/transformation consistency, không cố bám gradient alignment nếu dữ liệu không ủng hộ.
- Bước tiếp theo: cài đặt Thí nghiệm 2A — cần tính gradient của surrogate và từng target tại cùng 1 ảnh/object (có thể tái dùng phần lớn hạ tầng `experiments/common.py` để load model/dataset, viết thêm hàm tính gradient theo đúng loss tại RoI của GT giống `attacks/detection_attacks.py._bbox_cls_loss` nhưng KHÔNG chạy attack — chỉ lấy `torch.autograd.grad` 1 lần trên ảnh sạch cho mỗi model), rồi so cosine similarity với nhãn evaded/not-evaded đã có sẵn từ Thí nghiệm 1B (`outputs/experiment1b/results.json`, hoặc tính lại từ `common.compute_asr`/`match_greedy` cho từng object cụ thể thay vì chỉ đếm tổng).

## 2026-09-20 — Phát hiện + fix bug tọa độ RoI (bug #3), re-run Exp1 + Exp1B ở n=300 (SUPERSEDE toàn bộ số liệu Exp1/1B trước đó)

**Bối cảnh phát hiện**: chuẩn bị viết code Thí nghiệm 2A (cần tính RoI thủ công tương tự attack), kiểm tra lại kỹ `data_sample.gt_instances.bboxes` trước khi tái sử dụng logic RoI của `attacks/detection_attacks.py`.

### Bug là gì

`gt_instances.bboxes` (dùng trực tiếp bởi `attacks/mi_fgsm.py` và `attacks/detection_attacks.py` để build RoI qua `bbox2roi`) hóa ra đang ở **tọa độ ẢNH GỐC** (`ori_shape`), KHÔNG phải tọa độ ảnh đã resize đưa vào mạng (`img_shape`, dùng để tính `feats`). Verify bằng cách so trực tiếp với annotation gốc trong file COCO — khớp tuyệt đối từng số thập phân, xác nhận GT chưa hề qua resize.

**Nguyên nhân gốc**: test pipeline của MMDetection có thứ tự `Resize` → `LoadAnnotations` (ngược train pipeline). Lúc `Resize` chạy, `gt_bboxes` chưa tồn tại trong data dict nên không bị resize theo; `LoadAnnotations` load thẳng từ file gốc sau đó, không có bước bù resize (vì mmdet coi GT test-time chỉ để evaluation ở tọa độ gốc).

**Hậu quả**: `bbox2roi([gt_bboxes])` đưa tọa độ ảnh gốc vào RoIAlign trên `feats` tính từ ảnh đã resize (`scale_factor` ~1.87x quan sát được trên subset này) — lệch tỷ lệ có hệ thống, RoIAlign trích sai vùng feature (nhỏ hơn, lệch về góc trên-trái so với vị trí object thật trong không gian mạng). Bug tồn tại từ đầu Thí nghiệm 1 (`attacks/mi_fgsm.py`), kế thừa sang `attacks/detection_attacks.py` — ảnh hưởng **toàn bộ** số liệu Thí nghiệm 1 + 1B đã ghi ở 2 entry phía trên (nay đánh dấu SUPERSEDED).

### Fix

Nhân `gt_bboxes` với `scale_factor` (lấy từ `data_sample.metainfo["scale_factor"]`) trước khi `bbox2roi`, trong cả `attacks/mi_fgsm.py` và `attacks/detection_attacks.py`.

### Verify fix — quick-test 50 ảnh (`experiment1.py 50`)

| Model | ASR trước fix (n=1000) | ASR sau fix (n=50, quick-test) |
|---|---|---|
| surrogate_r50 | 0.3298 | 0.9489 |
| target_r101 | 0.1595 | 0.6933 |
| target_convnext_t | 0.0909 | 0.4118 |
| target_swin_t | 0.0766 | 0.3373 |

Whitebox ASR nhảy từ 33%→95% — đúng tầm 1 attack kiểu DAG target chuẩn. Thứ tự same>cross-CNN>CNN→Transformer giữ nguyên, gap lớn hơn hẳn.

### Re-run n=1000 (Exp1 only) rồi đổi sang n=300 (cả Exp1+1B, theo quyết định của user để 2 thí nghiệm dùng cùng cỡ mẫu, dễ so sánh — n=1000 để dành lúc viết paper)

Đã chạy full Exp1 n=1000 với fix (kết quả lưu riêng, không bị ghi đè, tại `outputs/experiment1/results_n1000_postfix_verify.json`, để tham khảo/đối chiếu sau — KHÔNG phải kết quả final chính thức vì lúc chạy Exp1B vẫn ở n=1000, sau đó quyết định đổi cỡ mẫu giữa chừng):

| Model | Family | Clean AP50 | Adv AP50 | ASR |
|---|---|---|---|---|
| surrogate_r50 | surrogate | 0.5838 | 0.0035 | 0.9531 |
| target_r101 | same-family | 0.6013 | 0.0807 | 0.7255 |
| target_convnext_t | cross-cnn-family | 0.6864 | 0.2845 | 0.4614 |
| target_swin_t | cnn-to-transformer | 0.6616 | 0.3452 | 0.3725 |

Sau đó kill, chạy lại cả Exp1 + Exp1B ở **n=300** (tier "confirm" theo quy ước 50/300/1000 trong CLAUDE.md) — đây là số liệu chính thức dùng làm milestone hiện tại, thay thế hoàn toàn 2 entry cũ:

**Exp1 (n=300):**

| Model | Family | Clean AP50 | Adv AP50 | ASR |
|---|---|---|---|---|
| surrogate_r50 | surrogate | 0.5815 | 0.0048 | 0.9556 |
| target_r101 | same-family | 0.6099 | 0.0945 | 0.7121 |
| target_convnext_t | cross-cnn-family | 0.7086 | 0.3165 | 0.4519 |
| target_swin_t | cnn-to-transformer | 0.6811 | 0.3607 | 0.3763 |

Khớp gần như y hệt bản n=1000 ngay phía trên (chênh lệch <2 điểm % ở mọi model) — xác nhận n=300 đã đủ ổn định để làm tier "confirm".

**Exp1B (n=300), cả 4 run:**

| Model / Family | Run B (MI-FGSM, model zoo) | Run C (BIM) | Run D (cls+bbox) | Run E (DI-FGSM) |
|---|---|---|---|---|
| surrogate | 0.9563 | 0.9650 | 0.9462 | 0.8998 |
| same-family (r101) | 0.7082 | 0.5843 | 0.6892 | 0.7731 |
| same-family (x101) | 0.6823 | *(chỉ Run B)* | *(chỉ Run B)* | *(chỉ Run B)* |
| same-family (r50_3x) | **0.9130** | *(chỉ Run B)* | *(chỉ Run B)* | *(chỉ Run B)* |
| cross-CNN (ConvNeXt) | 0.4530 | 0.2575 | 0.4513 | 0.6273 |
| CNN→Transformer (Swin) | 0.3751 | 0.2181 | 0.3702 | 0.5308 |

**TransferGap (ASR_r101 − ASR_target), so với số liệu cũ (bug) trong ngoặc:**

| Run | Gap vs ConvNeXt | Gap vs Swin |
|---|---|---|
| B | 0.2552 (cũ: 0.0695) | 0.3331 (cũ: 0.0811) |
| C | 0.3268 (cũ: 0.0677) | 0.3662 (cũ: 0.0687) |
| D | 0.2379 (cũ: 0.0710) | 0.3190 (cũ: 0.0769) |
| E | 0.1458 (cũ: 0.0565) | 0.2423 (cũ: 0.0798) |

### Kết luận

- Transfer gap **lớn hơn 3-5 lần** số liệu cũ ở mọi run, sau khi attack được target đúng chỗ. Thứ tự same-family > cross-CNN > CNN→Transformer giữ nguyên tuyệt đối qua cả 4 confound-check — **RQ1 (đã khóa ở entry trước) không cần mở khóa lại, bằng chứng còn mạnh hơn**.
- Phát hiện mới đáng chú ý (Run B): `target_r50_3x` (cùng backbone R50 **y hệt** surrogate, chỉ khác recipe train) đạt ASR=0.9130 — gần bằng whitebox (0.9563). Sau khi attack đúng chỗ, cùng-kiến-trúc-chính-xác gần như transfer hoàn hảo, còn lệch kiến trúc (dù cùng họ CNN hay xa hơn) giảm rõ rệt và đơn điệu theo khoảng cách kiến trúc — tín hiệu mạnh hơn nhiều so với trước, ủng hộ hướng Thí nghiệm 2A (gradient alignment) rất tốt.
- `research_plan.md` §6.8 và RQ1 (§12) đã được đánh dấu supersede tương ứng, trỏ về entry này.
- **Việc chưa làm / cần cẩn thận**: `outputs/experiment1/raw_predictions.json` và các file kết quả n=1000 cũ (trước fix) đã bị ghi đè khi re-run — chỉ còn `results_n1000_postfix_verify.json` (n=1000, ĐÃ fix) làm tham khảo, không còn giữ bản n=1000 pre-fix nào (không cần thiết, đã biết là sai). `raw_predictions.json` hiện tại tương ứng với lần chạy n=300 mới nhất, không phải n=1000.
- Bước tiếp theo: viết code Thí nghiệm 2A (gradient alignment, xem entry phía trên) — dùng attack ĐÃ FIX, nhãn evaded/not-evaded nên tính lại trực tiếp trong code Thí nghiệm 2A (không tái sử dụng `outputs/experiment1b/results.json` vì file đó không lưu per-object raw predictions).

## 2026-09-20 — Thí nghiệm 2A: cài đặt + chạy (n=300, confirm), gradient alignment giải thích được transfer gap

### Đề xuất gốc của user (ghi nguyên văn, làm cơ sở đối chiếu đã cover được bao nhiêu)

> Bước tiếp theo nên vào Experiment 2 – tìm cơ chế. Mình đề xuất bắt đầu bằng thứ tự rẻ → sâu:
>
> 1. Gradient alignment trước. Với cùng clean image/object, đo cosine similarity: cos(g_s,g_t) = g_s^T g_t / (||g_s|| ||g_t||) giữa surrogate và từng target. Hypothesis: R50-R101 > R50-ConvNeXt > R50-Swin
> 2. Kiểm tra: GradientSimilarity ↑ ⟺ TransferSuccess ↑. Không chỉ correlate theo 3 target models, vì quá ít điểm. Hãy tính per-image hoặc per-object, rồi so distribution giữa evaded và not evaded.
> 3. Nếu gradient alignment giải thích được gap → tiếp tục xuống feature-level alignment để tìm layer nào gây ra divergence.
> 4. Nếu gradient alignment yếu → chuyển sớm sang object evidence / saliency / transformation consistency.
>
> Một Experiment 2A rất sạch sẽ là: Architecture family → gradient alignment → transfer success
>
> Nếu kết quả ra kiểu (R50→R101 high/high, R50→ConvNeXt medium/medium, R50→Swin low/low) thì bạn bắt đầu có mechanistic explanation, chứ không còn chỉ có empirical transfer gap.
>
> Sau đó mới đặt câu hỏi quan trọng nhất cho method: Làm sao từ một surrogate duy nhất tạo gradient/feature direction ít architecture-specific hơn? Đây sẽ là điểm xuất phát trực tiếp để search idea và tạo method mới.

### Phần đã cover trong lần chạy này (so với đề xuất trên)

- **Mục 1 (gradient alignment, cos_sim giữa surrogate và từng target)** — ĐÃ LÀM ĐỦ. Code `experiments/experiment2a.py`, verify hypothesis thứ tự R50-R101 > R50-ConvNeXt > R50-Swin bằng `mean_cos_sim_overall` từng target — **khớp đúng thứ tự**.
- **Mục 2 (kiểm tra GradientSimilarity ↑ ⟺ TransferSuccess ↑, per-object không phải chỉ 3 điểm, so distribution evaded vs not-evaded)** — ĐÃ LÀM PHẦN LÕI, còn thiếu 1 phần nhỏ: đã tính per-object thật (không phải per-image), đã so `mean_cos_sim_evaded` vs `mean_cos_sim_not_evaded` + Pearson/point-biserial correlation trên hàng nghìn record — nhưng **chưa chạy 1 test thống kê chính thức** (vd Mann-Whitney U hoặc t-test 2 mẫu độc lập) để khẳng định khác biệt evaded/not-evaded có ý nghĩa thống kê hay chỉ dừng ở so mean/std/correlation mô tả. Nên làm nếu cần con số p-value cho paper.
- **Mục "Experiment 2A sạch sẽ: Architecture family → gradient alignment → transfer success"** — ĐÃ CONFIRM đầy đủ cả chuỗi, không chỉ từng khúc rời: family family ordering (r101 same > ConvNeXt cross-CNN > Swin cross-family) → cos_sim ordering (khớp) → ASR ordering đã biết từ Exp1/1B (khớp) — đúng y hệt pattern ví dụ user đưa ra (high/high, medium/medium, low/low).
- **Mục 3 (nếu giải thích được → xuống feature-level alignment)** — CHƯA LÀM, đây là bước kế tiếp (§7.B research_plan.md), quyết định rẽ nhánh đã đúng hướng vì mục 1+2 cho tín hiệu mạnh (không rẽ sang mục 4).
- **Mục 4 (object evidence/saliency/transformation consistency)** — không áp dụng, không đi nhánh này vì gradient alignment đã giải thích được gap.
- **Câu hỏi cuối cho method ("làm sao tạo gradient/feature direction ít architecture-specific hơn")** — CHƯA bắt đầu, đây là việc của giai đoạn thiết kế method (research_plan.md §8-§9), sau khi xong cả gradient-level lẫn feature-level.

**Tóm lại: đã cover trọn phần "gradient alignment" (mục 1+2 và toàn bộ chuỗi suy luận), chưa đụng tới feature-level (mục 3) và câu hỏi method (đoạn cuối) — đúng như thiết kế rẻ→sâu, dừng đúng chỗ để chờ quyết định có xuống sâu hơn không.**

### Kết quả (n=300, tier confirm — code: `experiments/experiment2a.py`, output: `outputs/experiment2a/records.jsonl` + `summary.json`)

Config attack y hệt Exp1/1B (đã fix bug tọa độ RoI): epsilon=8.0, num_iter=10, decay=1.0, objective="cls". Với mỗi object clean-correct (theo từng target), tính cos(g_s, g_t) trên ảnh sạch (không chạy attack lúc tính gradient), so với nhãn evaded/not-evaded (object đó có bị attack — crafted trên surrogate — làm target né tránh hay không).

| Target | Family | n objects | mean cos_sim (evaded) | mean cos_sim (not evaded) | correlation (point-biserial) |
|---|---|---|---|---|---|
| target_r101 | same-family | 1525 | 0.1397 | 0.0878 | +0.3170 |
| target_convnext_t | cross-cnn-family | 1693 | 0.0905 | 0.0524 | +0.3515 |
| target_swin_t | cnn-to-transformer | 1637 | 0.0491 | 0.0282 | +0.3272 |

Đối chiếu quick-test n=50 trước đó (r101: 0.1332/0.0981/corr=0.2398; convnext: 0.0919/0.0508/corr=0.3560; swin: 0.0519/0.0297/corr=0.3167) — **sai lệch <0.02 ở mọi ô, pattern hoàn toàn ổn định**, không phải nhiễu mẫu nhỏ.

**Kết luận**: gradient alignment giải thích được transfer gap ở cả 2 mức — (1) per-object: cos_sim cao hơn ở nhóm evaded so với not-evaded, correlation dương ổn định ~0.32-0.35 ở cả 3 target; (2) per-target: độ lớn cos_sim tổng thể giảm đơn điệu đúng thứ tự kiến trúc, khớp hoàn toàn thứ tự ASR đã biết. Chuỗi `Architecture family → gradient alignment → transfer success` được xác nhận thực nghiệm — chuyển từ "empirical transfer gap" (Exp1/1B) sang **mechanistic explanation** đầu tiên của dự án.

### Bước tiếp theo

1. ~~(Tùy chọn, nếu cần rigor cho paper) Chạy test thống kê chính thức (Mann-Whitney U) trên distribution cos_sim evaded vs not-evaded thay vì chỉ so mean/std.~~ Đã làm — xem entry dưới.
2. ~~Theo quy tắc rẽ nhánh đã chốt: xuống **feature-level alignment** (research_plan.md §7.B) — tìm layer nào trong backbone gây divergence nhiều nhất giữa surrogate và từng target.~~ Đã làm (Thí nghiệm 2B + 2B.1) — xem entry dưới, kết quả là **negative finding**.
3. Cân nhắc chạy Thí nghiệm 2A ở n=1000 khi cần số liệu final cho paper (hiện tại n=300 đã đủ làm milestone/confirm).

## 2026-09-20 — Thí nghiệm 2A bổ sung (Mann-Whitney U), Thí nghiệm 2B + 2B.1: feature-level alignment KHÔNG giải thích được transfer success (negative finding)

### Bổ sung Thí nghiệm 2A: Mann-Whitney U + effect size

Thêm vào `experiments/experiment2a.py` (hàm `compute_summary`, tái sử dụng qua `--reanalyze` không cần chạy lại GPU): Mann-Whitney U một phía (H1: cos_sim nhóm evaded > not-evaded) + 2 effect size (rank-biserial — chuẩn cho Mann-Whitney; Cohen's d — tham khảo). Bug nhỏ tự bắt trước khi báo cáo: công thức rank-biserial `1 - 2U/(n1n2)` cho dấu NGƯỢC (verify bằng ví dụ tổng hợp x>>y phải cho r=+1 nhưng công thức đó ra -1) — sửa thành `2U/(n1n2) - 1`.

Kết quả (n=300, dùng lại `records.jsonl` đã có, không cần GPU):

| Target | MWU p-value (1 phía) | rank-biserial | Cohen's d |
|---|---|---|---|
| target_r101 | 5.62e-43 | +0.4470 | 0.7372 |
| target_convnext_t | 8.34e-53 | +0.4299 | 0.7536 |
| target_swin_t | 8.30e-42 | +0.3984 | 0.7158 |

p cực nhỏ, effect size trung bình-lớn (rank-biserial ~0.40-0.45, Cohen's d ~0.72-0.75) ở cả 3 target — chốt chính thức được thống kê cho kết luận 2A (gradient alignment cao hơn ở nhóm evaded).

### Thí nghiệm 2B: global CKA theo stage — negative finding so với hypothesis

Thiết kế (research_plan.md §7.B): với mỗi stage backbone (S1-S4, verify cùng stride [4,8,16,32] ở cả 4 model dù channel dim khác — 256/512/1024/2048 ResNet vs 96/192/384/768 ConvNeXt/Swin, đây là lý do dùng **linear CKA** thay vì cosine trực tiếp), pool object-conditioned feature (RoIAlign 7x7 + global average) trên đúng population object đã dùng ở 2A (đọc từ `records.jsonl`, không generate lại attack — 2B chỉ cần ảnh sạch). Code: `experiments/experiment2b.py`.

Bug kỹ thuật gặp và fix: `mmcv.ops.roi_align` (bản mmcv/torch đang dùng) không nhận keyword arguments qua `torch.autograd.Function.apply` — phải truyền toàn bộ theo thứ tự positional của `RoIAlignFunction.forward`.

**Kết quả (n=300, khớp population 2A)**:

| Stage | R50→R101 | R50→ConvNeXt | R50→Swin |
|---|---|---|---|
| Stage 1 | 0.9798 | 0.7241 | 0.8459 |
| Stage 2 | 0.9668 | 0.7482 | 0.9299 |
| Stage 3 | 0.9134 | 0.7548 | 0.8454 |
| Stage 4 | 0.9387 | 0.7810 | 0.8137 |

**Negative finding**: thứ tự thực tế là **R101 > Swin > ConvNeXt** ở CẢ 4 stage — không phải R101 > ConvNeXt > Swin như hypothesis (và như ASR/gradient alignment ở Exp1/1B/2A đã cho). Đã tự verify không phải bug: `linear_cka` cho self-CKA=1.0, scale-invariant, random-baseline ~0.15 (kiểm tra bằng dữ liệu tổng hợp).

> **Global forward-feature similarity (CKA) không đi cùng ASR**: ASR/gradient alignment cho R101 > ConvNeXt > Swin, còn CKA cho R101 > Swin > ConvNeXt.

### Thí nghiệm 2B.1: trong-từng-target, evaded vs not-evaded — vẫn negative finding, nhưng theo hướng ngược lại thú vị

Câu hỏi hẹp hơn: TRONG CÙNG 1 target (không so target với nhau), object evaded có CKA với surrogate cao hơn object not-evaded không? Tái sử dụng raw feature đã lưu ở `outputs/experiment2b/features/*.npz` (đã sửa `experiment2b.py` lưu thêm nhãn evaded per-object, không cần forward lại). So sánh bằng bootstrap (n_sample = min(n_evaded, n_not_evaded), resample có hoàn lại, B=200 — ban đầu thử B=1000 với công thức CKA kiểu cross-covariance (D×D), ước tính chạy ~70 phút (D=2048 ở stage sâu quá đắt, benchmark thực tế 1.5s/lần gọi) → đã dừng, đổi sang công thức CKA qua Gram matrix (N×N), verify cho ra cùng kết quả nhưng nhanh hơn tới 8.5x ở D=2048, giảm B xuống 200, tổng chạy còn ~13 phút). Code: `experiments/experiment2b1.py`.

**Kết quả đầy đủ (12 tổ hợp target×stage)**:

| Target | Stage | CKA(evaded) | CKA(not evaded) | diff | CI95 non-overlap |
|---|---|---|---|---|---|
| ConvNeXt | 1 | 0.7234 | 0.7437 | −0.0203 | |
| ConvNeXt | 2 | 0.7519 | 0.7625 | −0.0106 | |
| ConvNeXt | 3 | 0.7721 | 0.7988 | −0.0268 | |
| ConvNeXt | 4 | 0.7785 | 0.8242 | **−0.0458** | ★ |
| R101 | 1 | 0.9803 | 0.9802 | +0.0002 | |
| R101 | 2 | 0.9681 | 0.9656 | +0.0025 | |
| R101 | 3 | 0.9217 | 0.9401 | −0.0184 | |
| R101 | 4 | 0.9376 | 0.9626 | **−0.0250** | ★ |
| Swin | 1 | 0.8571 | 0.8439 | +0.0132 | |
| Swin | 2 | 0.9216 | 0.9387 | **−0.0171** | ★ |
| Swin | 3 | 0.8564 | 0.8661 | −0.0098 | |
| Swin | 4 | 0.8133 | 0.8476 | **−0.0343** | ★ |

**Phát hiện**: ở stage 4 (sâu nhất), **cả 3 target** đều có `CKA_evaded < CKA_not_evaded` với CI95 không chồng lấp — NGƯỢC hoàn toàn với hướng hypothesis đặt ra (`CKA_evaded > CKA_not_evaded`).

### Interpretation đã chốt (user, 2026-09-20)

- **Bác bỏ hypothesis "forward feature similarity cao → transfer dễ hơn"** — dữ liệu nói điều ngược lại ở deep stage, nhất quán qua cả 3 target.
- Deep CKA có vẻ phản ánh **object representation stability/canonicality** (object nào 2 model "nhìn" giống nhau ở mức biểu diễn cao thường là object rõ ràng/tự tin, khó bị đánh bật bởi 1 nhiễu bounded-epsilon) hơn là **compatibility của adversarial direction** giữa 2 model.
- Tính đến giờ, **gradient alignment (2A) là tín hiệu giải thích transfer success tốt hơn hẳn raw feature alignment (2B/2B.1)** — 2A đi đúng chiều hypothesis (cos_sim cao hơn ở nhóm evaded, p cực nhỏ, effect size trung bình-lớn), còn feature-level alignment thì không, thậm chí đi ngược ở deep layer.
- **Lưu ý phương pháp luận cho paper sau này**: "CI95 không chồng lấp" ở 2B.1 là bằng chứng đủ mạnh để BÁO CÁO Ở ĐÂY (progress_log, giai đoạn khám phá), nhưng KHÔNG nên gọi là "formal significance test" khi viết paper — nên dùng trực tiếp bootstrap CI của hiệu số `CKA_evaded - CKA_not` (paired trong cùng resample, không phải 2 CI độc lập rồi so sánh chồng lấp), hoặc permutation test, mới đúng chuẩn thống kê.

### Bước tiếp theo: chuyển sang backward information (feature-gradient / Jacobian alignment)

Không cố ép "forward feature similarity" vào chuỗi cơ chế nữa. Câu hỏi mới cho Thí nghiệm tiếp theo (tạm gọi 2C):

> Ở stage nào, backward sensitivity (Jacobian của feature đối với input, hoặc feature-gradient) của surrogate và target bắt đầu diverge theo đúng thứ tự R101 > ConvNeXt > Swin (khớp ASR/gradient alignment đã biết)?

Nếu tìm được, chuỗi cơ chế trở thành:

```
Architecture → Backward sensitivity divergence → Gradient misalignment → Transfer failure
```

thay vì cố dùng forward feature similarity (đã bị bác bỏ ở 2B/2B.1) làm mắt xích giữa kiến trúc và gradient misalignment.

## 2026-09-20 — Thí nghiệm 2C: backward sensitivity (feature-gradient) theo stage — finding mạnh nhất của Exp2, khớp đúng thứ tự ASR

### Thiết kế

Câu hỏi: khác biệt giữa forward feature similarity (2B, sai thứ tự) và gradient alignment tại RoI cuối (2A, đúng thứ tự) bắt đầu từ đâu trong mạng? Với mỗi object, tại mỗi stage backbone (S1-S4), pool feature (RoIAlign 7x7 + global average, giống 2B), lấy scalar = ||f^l(x)||₂² (tổng bình phương feature — lựa chọn tự nhiên, không cần projection ngẫu nhiên tùy tiện, không cần nhãn), backprop về ảnh SẠCH (không chạy attack, giống 2B) ra gradient trong không gian pixel input, so cos similarity giữa surrogate và target — cùng population object đã dùng ở 2A/2B. Code: `experiments/experiment2c.py`. Không gặp bug kỹ thuật mới (tái sử dụng đúng các helper đã verify ở 2A/2B: `_gt_bboxes_net` cho tọa độ RoI, `roi_align` positional args).

### Kết quả (n=300, 19420 record)

| Stage | R50→R101 | R50→ConvNeXt | R50→Swin | Khớp thứ tự R101>ConvNeXt>Swin? |
|---|---|---|---|---|
| Stage 1 (shallow) | 0.3285 | 0.0676 | 0.0713 | ✗ (Swin nhỉnh hơn ConvNeXt, sát nhau) |
| Stage 2 | 0.2926 | −0.0870 | −0.0835 | ✗ (cả 2 âm, Swin nhỉnh hơn nhẹ) |
| **Stage 3** | 0.1330 | **0.0296** | **−0.0145** | **✓ ĐÚNG** |
| **Stage 4 (deep)** | 0.1062 | **0.0438** | **−0.0032** | **✓ ĐÚNG** |

Thêm: within mỗi target, `mean_cos_sim(evaded) > mean_cos_sim(not_evaded)` giữ đúng ở hầu hết mọi stage/target — nhất quán với phát hiện 2A (chi tiết per-object trong `outputs/experiment2c/records.jsonl`).

### Kết luận đã chốt (user, 2026-09-20) — finding mạnh nhất của Exp2 tính đến giờ

> **Experiment 2C confirms that cross-family transferability is better explained by backward sensitivity alignment than by forward feature similarity.** Forward CKA in Exp2B failed to follow the ASR ordering, while feature-gradient alignment in Exp2C recovers the correct `R101 > ConvNeXt > Swin` ordering from Stage 3 onward. This suggests that the critical divergence emerges in mid-to-deep backbone stages and propagates into input-gradient misalignment, which then limits transfer success.

Chuỗi cơ chế nâng cấp từ bản nháp ở entry trước:

$$
\boxed{Architecture \rightarrow \text{mid/deep backward-sensitivity divergence (từ Stage 3)} \rightarrow \text{input-gradient misalignment} \rightarrow \text{transfer failure}}
$$

Giữ nguyên negative finding của 2B/2B.1 làm 1 phần của story (không xóa, làm rõ tương phản):

$$
\text{forward similarity} \not\Rightarrow \text{transferability}, \quad \text{backward alignment} \Rightarrow \text{transferability}
$$

### Đánh giá: đã đủ để chuyển sang giai đoạn method (research_plan.md §8-§9), không cần đào thêm diagnostic

Theo user: kết quả 2A+2B+2B.1+2C đã đủ mạnh và nhất quán (âm ở forward feature, dương và đúng thứ tự ở backward sensitivity từ stage 3) để dừng vòng lặp tìm-cơ-chế (research_plan.md §8, bước 1-6) và bắt đầu thiết kế method (bước 7). Câu hỏi trung tâm cho method, cụ thể hóa từ câu hỏi chung ở research_plan.md §9-§10:

> **Làm sao từ một surrogate duy nhất giảm tính architecture-specific của backward signal tại Stage 3-4?**

Đây là điểm nhắm trực tiếp cho novelty/SOTA claim (research_plan.md §10-§11), thay vì hướng chung chung "architecture-invariant object evidence disruption" đã phác thảo trước đó ở §9.

### Bước tiếp theo

1. ~~Cân nhắc khóa RQ2 (research_plan.md §12) với kết luận trên — chưa làm, để hỏi user riêng (giống cách RQ1 đã khóa ở entry trước).~~ Đã khóa ngay sau đó cùng ngày — xem `research_plan.md` §12: RQ2 answered (backward sensitivity alignment > forward feature similarity, divergence từ Stage 3-4), RQ3 cụ thể hóa thành câu hỏi method duy nhất còn mở.
2. ~~Bắt đầu thiết kế method: tấn công single-surrogate nhắm trực tiếp vào backward sensitivity ở Stage 3-4 (thay vì chỉ RoI classification loss cuối như MI-FGSM hiện tại) — cần literature search cụ thể hơn cho hướng "feature-gradient/Jacobian-based attack" (research_plan.md §8 bước 1).~~ Đã làm bước cầu nối (Thí nghiệm 3A, causal intervention) trước khi thiết kế method hẳn — xem entry dưới.

## 2026-09-20 — Thí nghiệm 3A: causal intervention xác nhận Stage 3-4 là nguồn tác động thật (không chỉ correlation)

### Thiết kế

RQ2 (đã khóa) dựa trên correlation (2A/2C: backward-sensitivity alignment đi cùng ASR, mạnh nhất từ Stage 3-4). 3A kiểm tra xem có phải quan hệ NHÂN QUẢ không: regularize (giảm variance) gradient CHÍNH XÁC tại Stage 3-4 trong lúc craft attack có tăng cross-family ASR nhiều hơn regularize ở Stage 1-2 không?

Literature check trước khi implement (đúng đề xuất — search rất tập trung, không tùy tiện):
- **TGR** (Zhang et al., CVPR 2023, "Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization"): giảm variance gradient lan truyền ngược tại block trung gian bằng cách loại bỏ giá trị cực trị (token-wise, cho ViT Attention/QKV/MLP block).
- **MIG** (Ma et al., ICCV 2023, "Transferable Adversarial Attack for Both Vision Transformers and Convolutional Networks via Momentum Integrated Gradients"): dùng integrated gradients (tích phân dọc đường thẳng từ baseline tới ảnh thật) thay vì gradient tại 1 điểm, vì integrated gradients tương đồng hơn giữa các model.

Baseline regularizer implement — **CỐ Ý đơn giản**, mượn tinh thần TGR (clip gradient theo mean±3·std của chính nó) nhưng áp ở granularity BACKBONE STAGE (feature map) thay vì token/attention-block, để dùng thống nhất được cho cả ResNet/ConvNeXt/Swin — KHÔNG phải reimplement TGR. Cơ chế: `tensor.register_hook()` trực tiếp trên feature map stage cần regularize, đăng ký giữa lúc gọi `model.backbone(x)` và `model.neck(...)` (2 bước tách thủ công từ `extract_feat`, đúng những gì `extract_feat` làm bên trong, không đổi hành vi gì khác ngoài chèn hook). Code: `attacks/backward_reg_attack.py` (file mới, không sửa `attacks/detection_attacks.py`, tái sử dụng `_bbox_cls_loss`/`_bbox_cls_bbox_loss`/`_apply_input_diversity` từ đó).

4 setting, cùng 4 model + cùng config attack (epsilon=8.0, num_iter=10, decay=1.0, objective=cls — y hệt Thí nghiệm 1), chỉ khác `reg_stages`: `baseline` (không regularize), `reg_s1s2` (Stage 1-2, nhóm đối chứng), `reg_s3s4` (Stage 3-4, hypothesis chính), `reg_all` (cả 4 stage, tham khảo). Code: `experiments/experiment3a.py`.

### Kết quả (n=300)

**ASR tuyệt đối:**

| Target | Baseline | reg_s1s2 | reg_s3s4 | reg_all |
|---|---|---|---|---|
| surrogate | 0.9576 | 0.9570 | 0.9597 | 0.9610 |
| target_r101 | 0.7102 | 0.6754 | 0.7161 | 0.6872 |
| target_convnext_t | 0.4501 | 0.4194 | **0.4637** | 0.4312 |
| target_swin_t | 0.3787 | 0.3519 | **0.4020** | 0.3739 |

**Gain (ASR_reg − ASR_baseline):**

| Target | reg_s1s2 | reg_s3s4 | reg_all |
|---|---|---|---|
| surrogate | −0.0007 | +0.0020 | +0.0034 |
| target_r101 | −0.0348 | +0.0059 | −0.0230 |
| target_convnext_t | **−0.0307** | **+0.0136** | −0.0189 |
| target_swin_t | **−0.0269** | **+0.0232** | −0.0049 |

Hypothesis `Gain_{S3-4} > Gain_{S1-2}` confirm rõ ràng ở cả 2 cross-family target (chênh lệch +0.0443 ở ConvNeXt, +0.0501 ở Swin). `reg_s1s2` luôn có hại (mọi target, kể cả same-family); `reg_s3s4` luôn có lợi. `reg_all` tệ hơn `reg_s3s4` riêng lẻ ở mọi target — không phải "càng regularize nhiều càng tốt", mà cụ thể Stage 3-4 mới có tác dụng, pha trộn với Stage 1-2 (có hại) làm loãng lợi ích.

### Kết luận đã chốt (user, 2026-09-20)

> **Experiment 3A provides targeted causal evidence that mid/deep backward signals are the actionable source of cross-family transferability. Regularizing Stage 3–4 consistently improves transfer to both ConvNeXt and Swin, while regularizing Stage 1–2 consistently hurts transfer. Regularizing all stages is also worse than Stage 3–4 alone, showing that the effect is stage-specific rather than a generic benefit of stronger regularization.**

Điểm mạnh nhất, dùng để bác bỏ giải thích "regularization nói chung giúp transfer":

$$
reg_{S3-4} > baseline > reg_{S1-2}
$$

ở cả 2 cross-family target, và:

$$
reg_{all} < reg_{S3-4}
$$

### Trạng thái RQ3: bằng chứng feasibility rất mạnh, CHƯA khóa hoàn toàn

Khác với RQ1/RQ2 (đã khóa), RQ3 chưa khóa vì 3A mới là intervention đơn giản (baseline regularizer mượn tinh thần TGR), chưa phải method novel/SOTA — chỉ đủ để xác nhận feasibility (backward signal ở Stage 3-4 CÓ THỂ can thiệp được để tăng cross-family transfer, và hiệu ứng có tính causal, cụ thể theo stage). Roadmap tiếp theo:

$$
\boxed{\text{Search prior methods} \rightarrow \text{identify overlap} \rightarrow \text{design novel stage-aware backward regularization}}
$$

Định hướng novelty cụ thể: **Stage 3-4 được chọn dựa trên backward-sensitivity evidence** (đo được, không phải chọn heuristic/toàn mạng như TGR) — phân biệt với TGR (không có bước "đo để chọn stage", áp dụng toàn bộ ViT block) và với OSFD (AAAI, phá forward object feature — hướng của dự án này nhắm backward sensitivity geometry, khác hẳn). Có thể mạnh hơn nữa nếu thêm **object-conditioned weighting** vào Stage 3-4 (hợp tự nhiên với object detection, khác classification-only literature hiện có).

### Bước tiếp theo

1. ~~Literature search tập trung hơn nữa cho các method "stage-aware"/"layer-selection" backward regularization đã có (để xác định overlap thật, tránh trùng lặp khi claim novelty).~~ Đã làm — xem entry dưới (TGR, MIG, PAS — Backpropagation Path Search ICCV 2023, GRA — Gradient Relevance Attack ICCV 2023, OSFD).
2. ~~Thiết kế method chính thức: attack single-surrogate kết hợp (a) regularize backward signal tại Stage 3-4 (đã validate ở 3A) + (b) object-conditioned weighting (research_plan.md §9).~~ Đã làm bước cầu nối (Thí nghiệm 3B) — xem entry dưới.
3. Benchmark method mới so với baseline hiện có (MI-FGSM, DI-FGSM đã có ở Thí nghiệm 1B) + OSFD/TGR/MIG nếu tái lập được, theo đúng threat model research_plan.md §11.

## 2026-09-20 — Thí nghiệm 3B: object-conditioned weighting + literature overlap check — candidate method bắt đầu có cấu trúc rõ ràng

### Literature overlap check (trước khi implement 3B)

Search tập trung các method "chỉnh backward path/gradient" gần nhất với hướng Stage 3-4 + object-conditioned:
- **TGR** (CVPR 2023): regularize variance gradient ở intermediate block của ViT (token-wise, Attention/QKV/MLP) — không có bước đo backward sensitivity để CHỌN stage, không phải object detection.
- **MIG** (ICCV 2023): dùng integrated gradients thay gradient tại 1 điểm — không chọn stage, không object-conditioned.
- **PAS — Backpropagation Path Search** (Xu et al., ICCV 2023): search DAG/backprop path để giảm surrogate overfitting — gần nhất về ý tưởng "chỉnh backward path", nhưng classification-centric, không dùng mechanism diagnostic kiểu Stage 3-4 (đo rồi mới chọn).
- **GRA — Gradient Relevance Attack** (Zhu et al., ICCV 2023): chỉnh gradient update dựa trên relevance/neighborhood fluctuation — không stage-aware.
- **OSFD** (AAAI 2024): object-detection-specific, transferable mạnh, nhưng thao tác **forward object feature**, không phải backward sensitivity geometry — hướng khác hẳn.

Kết luận overlap: không method nào có ĐỦ 2 đặc điểm cùng lúc — (1) chọn stage dựa trên đo backward-sensitivity evidence thay vì heuristic/toàn mạng, (2) object-conditioned cho object detection. Đây là khoảng trống thật, nhưng claim novelty phải chính xác ở mức "mechanism-guided stage selection + object-conditioned backward regularization", KHÔNG phải "we regularize intermediate gradients" (đã có TGR/PAS/GRA làm rồi dưới hình thức khác).

### Thiết kế 3B

Câu hỏi: object-aware weighting có giúp backward regularization tập trung vào transferable object-sensitive directions thay vì regularize toàn feature map như nhau không? 4 setting, cùng model/attack config (epsilon=8.0, num_iter=10, decay=1.0, objective=cls) như 3A:
- `baseline`: không regularize.
- `reg_s3s4`: clip variance ĐỀU tại Stage 3-4 (y hệt 3A, mốc tham chiếu).
- `object_weight_only`: CHỈ nhân gradient với mask object (1.0 trong vùng GT box theo stride của stage, 0.3 ngoài — hard box, không làm mượt biên, đủ cho kiểm tra targeted) tại Stage 3-4, KHÔNG clip.
- `reg_s3s4_object_weight`: clip NHƯNG chỉ áp trong vùng object (blend theo mask với gradient gốc bên ngoài).

Code: mở rộng `attacks/backward_reg_attack.py` (thêm `stage_modes` dict tổng quát hơn `reg_stages` cũ của 3A, giữ tương thích ngược — đã verify baseline tái lập y hệt trước/sau refactor) + `experiments/experiment3b.py`.

**Phát hiện kỹ thuật quan trọng khi verify trước khi chạy n=300**: setting có `tensor.register_hook` cho kết quả dao động nhẹ giữa các lần chạy CÙNG 1 code (baseline luôn tái lập y hệt tuyệt đối vì không có hook) — nondeterminism của GPU/cudnn khi có backward hook, không phải bug logic. Ở n=300 hiệu ứng này nhỏ đi nhiều so với n=5 lúc sanity-test (dao động 1-2 object trên 26-30 object ở n=5 là đáng kể, nhưng trên ~1500-1700 object ở n=300 thì không đủ để đảo ngược kết luận). Cần lưu ý khi tái lập chính xác tuyệt đối kết quả (paper sau này nên cân nhắc tắt `cudnn.benchmark`/ép deterministic algorithm nếu cần reproducibility bit-exact).

### Kết quả (n=300)

**ASR tuyệt đối:**

| Target | Baseline | reg_s3s4 | object_weight_only | reg_s3s4_object_weight |
|---|---|---|---|---|
| surrogate | 0.9570 | 0.9610 | 0.9597 | 0.9644 |
| target_r101 | 0.7128 | 0.7167 | 0.7115 | 0.7128 |
| target_convnext_t | 0.4542 | 0.4649 | 0.4460 | **0.4702** |
| target_swin_t | 0.3794 | 0.3965 | 0.3800 | **0.4013** |

**Gain (so với baseline):**

| Target | reg_s3s4 | object_weight_only | reg_s3s4_object_weight |
|---|---|---|---|
| target_r101 | +0.0039 | −0.0013 | +0.0000 |
| target_convnext_t | +0.0106 | −0.0083 | **+0.0159** |
| target_swin_t | +0.0171 | +0.0006 | **+0.0220** |

### Kết luận đã chốt (user, 2026-09-20)

- **`Stage 3-4 regularization` (clip) là thành phần chính tạo gain cross-family** — `object_weight_only` (chỉ mask, không clip) tự nó không có tác dụng, thậm chí hơi có hại ở ConvNeXt (−0.0083).
- **`Object weighting` một mình không đủ, nhưng có interaction DƯƠNG khi kết hợp với Stage 3-4 regularization**: combo (`reg_s3s4_object_weight`) vượt rõ `reg_s3s4` thuần ở cả 2 cross-family target (ConvNeXt +0.0159 vs +0.0106 — tăng thêm ~50%; Swin +0.0220 vs +0.0171 — tăng thêm ~29%).
- **Combo cải thiện ConvNeXt/Swin nhưng gần như không cải thiện R101** (gain=+0.0000, thấp hơn cả `reg_s3s4` riêng +0.0039) → tín hiệu rất tốt rằng method đang nhắm đúng vào **cross-family gap**, không chỉ tăng attack strength chung chung (nếu là hiệu ứng chung, same-family phải tăng theo tỷ lệ tương tự).

### Trạng thái: candidate method đã có cấu trúc rõ ràng — KHÔNG thêm component mới ngay

Công thức method (3 thành phần, đã validate từng phần qua 3A+3B):
1. **Stage-selection rule**: chọn Stage 3-4, dựa trên backward-sensitivity evidence đo được ở Thí nghiệm 2C (không phải heuristic/toàn mạng như literature hiện có).
2. **Backward regularizer**: clip variance (tinh thần TGR, adapt sang backbone-stage granularity).
3. **Object-conditioned weighting**: giới hạn regularizer vào vùng object (tương tác dương với #2, không có tác dụng đứng một mình).

Roadmap tiếp theo:

$$
\boxed{\text{formalize method} \rightarrow \text{ablation/parameter study} \rightarrow \text{strong baselines} \rightarrow \text{full benchmark}}
$$

### Bước tiếp theo

1. ~~Chốt rõ công thức method (đặt tên, viết formal description — 3 thành phần ở trên).~~ Đã làm — xem entry dưới (research_plan.md §9.1).
2. ~~Ablation/parameter study: `OBJECT_MASK_BG_WEIGHT` hiện đang hard-code 0.3 (chưa tune), mask hiện là hard box (chưa thử soft/Gaussian falloff), clip bound hiện 3·std (chưa thử giá trị khác) — cần xem độ nhạy trước khi chốt method cuối.~~ Ablation A (clip bound k) đã làm — xem entry dưới. Ablation B (λ) và C (mask shape) còn lại.
3. Strong baseline: benchmark với TGR/MIG/OSFD/DI-FGSM (đã có DI-FGSM từ Thí nghiệm 1B) để biết candidate này thực sự SOTA hay chỉ hơn baseline nội bộ.
4. Full benchmark theo đúng threat model research_plan.md §11 (cùng dataset, cùng budget, cùng surrogate/target).

## 2026-09-20 — Formalize Method v0.1 (research_plan.md §9.1) + Ablation A (clip bound k)

### Formalize method v0.1

Trước khi ablation, chốt công thức chính xác khớp code (không viết "đẹp hơn" implementation — quy tắc user đặt ra). Full công thức ở `docs/research_plan.md` §9.1. Tóm tắt: với stage \(l \in \mathcal{S}^*=\{3,4\}\),

\[
\hat g_l = M_l \odot \operatorname{clip}(g_l;\ \mu_l-k\sigma_l,\ \mu_l+k\sigma_l) + (1-M_l)\odot g_l
\]

— convex blend per-pixel giữa gradient đã clip và gradient gốc theo mask object \(M_l\) (KHÔNG phải multiplicative `M⊙clip(g)` hay residual `(1+λM)⊙clip(g)` như phác thảo ban đầu của user — đã tự sửa sau khi đối chiếu code thật). Refactor `attacks/backward_reg_attack.py`: tổng quát hóa 2 hằng số hard-code thành tham số `k` (clip bound) và `lam` (λ, object-weight strength qua β(λ)=1/(1+λ)) — verify bằng unit test: λ=0 khớp TUYỆT ĐỐI (max diff=0.0) với `reg_s3s4` thuần. λ mặc định ≈2.333 tái lập đúng β=0.3 đã dùng ở 3B.

### ⚠️ Sự cố: ghi đè nhầm `outputs/experiment3b/results.json` (n=300 → n=5)

Khi chạy regression sanity-test sau refactor (`python3 experiments/experiment3b.py 5`), script ghi đè thẳng `results.json` mà không hỏi/backup — mất bản n=300 gốc của Thí nghiệm 3B (chỉ còn n=5). Phát hiện MUỘN, sau khi `experiments/experiment3c.py` (Ablation A) đã đọc nhầm 3 điểm "reused" (baseline/no-clip/k3_original) từ file đã hỏng, in ra bảng sai (số n=5 lẫn với số n=300 mới của k1/k2/k4). **Bài học lặp lại (đã từng nhắc ở entry Thí nghiệm 1, "backup trước khi ghi đè")**: quy tắc đó áp dụng cho MỌI lần chạy lại 1 script có khả năng ghi đè `results.json`, kể cả khi mục đích chỉ là "sanity-test nhanh sau khi sửa code" — không có ngoại lệ. Cách khắc phục đã dùng: 3 điểm reused có số liệu ĐÚNG đã ghi sẵn ở entry Thí nghiệm 3B phía trên (progress_log.md, không phụ thuộc file kết quả) — dùng lại để patch thủ công `outputs/experiment3c/results.json` mà không cần chạy lại GPU. `outputs/experiment3b/results.json` (file gốc) vẫn còn sai (n=5) — CẦN chạy lại `experiments/experiment3b.py 300` nếu sau này cần dùng file đó trực tiếp (hiện tại không cấp thiết vì số liệu đã bảo toàn trong progress_log.md).

### Ablation A: sweep clip bound k ∈ {1,2,3(gốc+lặp lại),4}, cố định λ≈2.333/β=0.3, Stage {3,4}, mode clip_weight

Tái sử dụng 3 điểm từ Thí nghiệm 3B (baseline, no-clip=`object_weight_only`, k=3 gốc=`reg_s3s4_object_weight`), chỉ chạy mới k=1, k=2, k=4, và 1 bản lặp lại k=3 (đo noise floor). Code: `experiments/experiment3c.py`.

**Kết quả (n=300):**

| Setting | ConvNeXt | Swin | CrossAvg | R101 | WhiteBox | TransferGap |
|---|---|---|---|---|---|---|
| baseline | 0.4542 | 0.3794 | 0.4168 | 0.7128 | 0.9570 | +0.2960 |
| no-clip | 0.4460 | 0.3800 | 0.4130 | 0.7115 | 0.9597 | +0.2985 |
| k=1 | 0.4578 | 0.4032 | 0.4305 | 0.7161 | 0.9657 | +0.2856 |
| k=2 | 0.4589 | 0.4020 | 0.4305 | 0.7102 | 0.9623 | +0.2797 |
| k=3 (gốc) | 0.4702 | 0.4013 | **0.4357** | 0.7128 | 0.9644 | **+0.2771** |
| k=3 (lặp lại) | 0.4690 | 0.4007 | 0.4349 | 0.7108 | 0.9637 | +0.2760 |
| k=4 | 0.4613 | 0.3958 | 0.4286 | 0.7154 | 0.9630 | +0.2868 |

### Kết luận đã chốt (user, 2026-09-20)

1. **k=3 đạt CrossAvg tốt nhất (0.4357/0.4349) và TransferGap thấp nhất (+0.2771/+0.2760)** trong toàn bộ sweep.
2. **Curve theo k có dạng non-monotonic**: sắp theo CrossAvg tăng dần — no-clip (0.4130) < baseline (0.4168) < k=4 (0.4286) < k=1≈k=2 (0.4305) < k=3 (đỉnh). Cả clip mạnh hơn (k=1,2) lẫn yếu hơn (k=4) đều kém hơn k=3.
3. **R101 (0.710-0.716) và white-box (0.957-0.966) gần như phẳng qua mọi k** — cải thiện cross-family ở k=3 KHÔNG đến từ đánh đổi source/same-family.
4. **Replicate k=3 cho sai khác rất nhỏ**: ΔCrossAvg=0.0008, ΔTransferGap=0.0011.

**Diễn đạt cẩn trọng (sửa lại theo yêu cầu user — tránh overclaim thống kê)**: KHÔNG viết "chênh lệch k=1/2/4 so với k=3 là khác biệt thật, không phải nhiễu". Viết đúng hơn:

> Chênh lệch giữa k=3 và các giá trị k lân cận (~0.005-0.007 ở CrossAvg) lớn hơn nhiều so với **observed run-to-run numerical variation** của replicate k=3 (~0.0008-0.0011). Một replicate DUY NHẤT chỉ đo được numerical/run-to-run noise tại đúng 1 điểm (k=3) — CHƯA thay thế được bootstrap CI hay statistical test thật trên n=300 (nếu cần rigor cho paper, nên chạy nhiều seed/replicate ở mọi k, không chỉ k=3).

**Finding riêng đáng chú ý** (mechanism story mạnh hơn):

\[
\text{no-clip} = 0.4130 < \text{baseline} = 0.4168
\]

Object-weighted backward modification MÀ KHÔNG có clipping **không tạo improvement** (thậm chí thấp hơn cả không làm gì). Improvement chỉ xuất hiện khi CÓ clipping, và đạt cực đại ở mức clipping trung gian (k=3). Tóm gọn:

\[
\text{backward modification alone} \not\Rightarrow \text{better transfer}, \qquad \text{properly bounded backward sensitivity} \Rightarrow \text{better cross-family transfer}
\]

### Bước tiếp theo

1. ~~**Ablation B**: sweep λ (object-weight strength), cố định k=3 (đã xác nhận tối ưu ở Ablation A). Trả lời câu hỏi còn lại: object-region weighting có thực sự cần thiết không (so với chỉ clip đều, không weighting — đã có data point này = `reg_s3s4` từ 3A/3B, λ→"đồng nhất"), và strength bao nhiêu là hợp lý?~~ Đã làm — xem entry dưới.
2. Ablation C (mask shape) — **hoãn vô thời hạn** theo quyết định cuối entry dưới (hyperparameter ablation cơ bản coi như đủ, không grid-search sâu thêm chỉ để tối ưu vài phần nghìn ASR).
3. **Nhắc lại quy tắc quan trọng**: mọi lần chạy lại script để sanity-test/regression-check PHẢI backup `results.json` hiện có trước nếu file đó chứa dữ liệu n=300 có giá trị — không có ngoại lệ dù chỉ chạy n nhỏ.

## 2026-09-20 — Ablation B: Object-region weighting strength λ

Giữ cố định Stage 3-4 và clip bound k=3, sweep strength của object-region weighting λ. λ=0 tương ứng pure clipping (`reg_s3s4`), không có object weighting bổ sung. Code: `experiments/experiment3d.py` (4 điểm chạy mới: λ=0.25/0.5/1/2), tái sử dụng λ=0 (=`reg_s3s4`, verify n=300 từ entry 3B) và λ≈2.333 (=`k3_original` của Ablation A) làm điểm tham chiếu, không chạy lại.

**Kết quả n=300:**

| λ | ConvNeXt | Swin | CrossAvg | R101 | WhiteBox | TransferGap |
|---|---|---|---|---|---|---|
| 0 | 0.4649 | 0.3965 | 0.4307 | 0.7167 | 0.9610 | +0.2860 |
| 0.25 | 0.4654 | 0.3971 | 0.4313 | 0.7154 | 0.9610 | +0.2842 |
| **0.5** | **0.4714** | **0.4062** | **0.4388** | 0.7056 | 0.9637 | **+0.2668** |
| 1 | 0.4625 | 0.3995 | 0.4310 | 0.7128 | 0.9610 | +0.2818 |
| 2 | 0.4678 | 0.3983 | 0.4330 | 0.7180 | 0.9603 | +0.2850 |
| 2.333 | 0.4702 | 0.4013 | 0.4357 | 0.7128 | 0.9644 | +0.2771 |

Curve theo λ không đơn điệu và nhìn chung khá phẳng, ngoại trừ vùng λ=0.5 cho CrossAvg cao nhất đồng thời TransferGap thấp nhất. Same-family và white-box không tăng tương ứng; tại λ=0.5, R101 thậm chí thấp hơn (0.7056, thấp nhất sweep), cho thấy cải thiện không đơn giản là do attack mạnh lên trên tất cả model.

### Replicate: chạy độc lập lại λ=0 và λ=0.5 (đo noise floor tại đúng 2 điểm cần confirm)

Code: `experiments/experiment3d_replicate.py` (tái sử dụng `run_one()` từ `experiment3d.py`, không copy logic).

| Setting | Run 1 CrossAvg | Replicate CrossAvg |
|---|---|---|
| λ=0 | 0.4307 | 0.4316 |
| λ=0.5 | 0.4388 | 0.4346 |

λ=0.5 cao hơn λ=0 trong **cả hai lần chạy độc lập**: +0.0081 (lần 1) và +0.0030 (lặp lại). Trung bình hai lần: λ=0.5 ≈ 0.4367, λ=0 ≈ 0.4312 — chênh khoảng +0.0056.

**Nuance quan trọng (thống nhất với user, tránh overclaim)**: replicate của λ=0.5 dao động lớn hơn control (λ=0) — **chưa đủ cơ sở để khẳng định λ=0.5 là global optimum chính xác**. Tuy nhiên, **thứ hạng tương đối λ=0.5 > λ=0 được tái lập độc lập** — đây là claim an toàn, đủ để chốt.

### Kết luận đã chốt (user, 2026-09-20)

> Object-region weighting provides an additional but secondary benefit beyond clipping. Its effect is localized around a moderate weighting strength rather than monotonic, and is weaker than the effect of clip-bound selection.

Từ Ablation A + B: **Stage 3-4 gradient clipping là thành phần chính** (Ablation A: curve rõ ràng, non-monotonic, đơn điệu 2 phía quanh đỉnh k=3), **object-region weighting đóng vai trò refinement bổ sung** (Ablation B: curve phẳng hơn nhiều, chỉ nổi bật cục bộ quanh λ≈0.5, effect yếu hơn hẳn effect của k).

### Quyết định dừng ablation hyperparameter

Sau entry này, **coi hyperparameter ablation cơ bản đã đủ** — không tiếp tục grid-search λ chỉ để tối ưu thêm vài phần nghìn ASR (Ablation C — mask shape — hoãn vô thời hạn cùng lý do). Bước tiếp theo nên kiểm tra **component/mechanism của method** (vd ablation có/không có stage-selection dựa trên evidence so với chọn ngẫu nhiên/heuristic khác — đã làm 1 phần ở Thí nghiệm 3A rồi) hơn là tiếp tục tuning hyperparameter.

## 2026-09-20 — HANDOFF: Component Ablation (3e) + Mechanism Validation (3f) CHƯA CÓ KẾT QUẢ, dừng chủ động giữa chừng

**Đọc entry này đầu tiên nếu bắt đầu phiên mới.**

### Trạng thái thật: KHÔNG có kết quả n=300 nào để đọc cho 3e/3f

User đề xuất roadmap 4 bước sau khi chốt Ablation A+B (xem entry ngay phía trên): (1) Component ablation, (2) Mask/control ablation, (3) Mechanism validation (đo lại cos(g_s,g_t) trước/sau regularization), (4) Full baseline comparison — **ưu tiên bước 1 → bước 3 trước**, bỏ qua bước 2 tạm thời.

Đã viết code cho cả bước 1 và bước 3, chạy song song (share GPU), rồi **user yêu cầu dừng chủ động** (không phải crash/lỗi) giữa chừng:

- `experiments/experiment3e.py` (Component Ablation — bảng: baseline / clip Stage 3 only / clip Stage 4 only / clip Stage 3+4 / Stage 3+4 + object weighting λ=0.5): mới chạy xong ~150/300 ảnh của setting ĐẦU TIÊN (`clip_s3`), setting `clip_s4` **chưa chạy dòng nào**. `outputs/experiment3e/results.json` hiện tại **vẫn là dữ liệu sanity-test n=5 cũ** (script chỉ ghi file sau khi 1 setting n=300 chạy xong hoàn toàn — chưa setting nào xong nên chưa ghi đè) — **KHÔNG đọc file này để lấy số n=300, sẽ nhầm**.
- `experiments/experiment3f.py` (Mechanism Validation — đo cos(g_s,g_t) trước/sau regularization Stage 3-4 k=3 λ=0.5, tái sử dụng population từ `outputs/experiment2a/records.jsonl`): mới chạy ~20-30/300 ảnh. `outputs/experiment3f/records.jsonl` + `summary.json` **cũng vẫn là dữ liệu sanity-test n=5 cũ** (cùng lý do — script chỉ ghi 1 lần ở CUỐI, sau khi xử lý hết toàn bộ n ảnh, chưa xong nên chưa ghi đè). Sanity-test n=5 ban đầu **đã cho tín hiệu rất đúng hướng hypothesis** (đáng chú ý, nhưng KHÔNG đủ tin cậy để kết luận gì): R101 delta=-0.0089 (âm), ConvNeXt delta=+0.0010, Swin delta=+0.0052 (cả 2 cross-family đều dương, same-family âm — đúng chiều mong đợi).

Cả 2 tmux session (`exp3e_run`, `exp3f_run`) đã bị kill sạch, không còn process nào chạy (`ps aux | grep experiment3` rỗng).

### Việc cần làm khi resume: chạy lại từ đầu, không có gì để "tiếp tục"

Cả 2 script đều KHÔNG có checkpoint/cache giữa chừng (giống bài học từ HANDOFF Thí nghiệm 1B trước đây) — resume nghĩa là chạy lại từ đầu:

```bash
cd /workspace/transfer-attack-new
source .venv/bin/activate
# Nếu máy mới hoàn toàn (khác máy này): setup_env.sh -> download_dataset.sh -> tải checkpoint (xem model_registry.md + 2 checkpoint x101/r50_3x) -> build_subset_annotations.py trước
tmux new-session -d -s exp3e_run "source .venv/bin/activate && python3 -u experiments/experiment3e.py 300 > outputs/experiment3e/run.log 2>&1; echo EXP3E_DONE_EXIT=\$? >> outputs/experiment3e/run.log"
tmux new-session -d -s exp3f_run "source .venv/bin/activate && python3 -u experiments/experiment3f.py 300 > outputs/experiment3f/run.log 2>&1; echo EXP3F_DONE_EXIT=\$? >> outputs/experiment3f/run.log"
```

Ước tính ~20-25 phút mỗi job nếu chạy riêng lẻ (không share GPU); nếu chạy song song như lần trước sẽ chậm hơn (quan sát thực tế lần trước: sau ~7 phút chạy song song, 3e mới xong 150/300 setting đầu, 3f mới ~20-30/300 — chậm hơn đáng kể so với chạy riêng, nên cân nhắc chạy TUẦN TỰ thay vì song song nếu muốn nhanh hơn tổng thể).

Nếu máy GPU thuê đã đổi (khác máy này — theo CLAUDE.md luôn giả định vậy trừ khi chắc chắn cùng máy): `outputs/experiment2a/records.jsonl` (population dùng cho 3f), `outputs/experiment3b/results.json` và `outputs/experiment3d/results.json` (dữ liệu tái sử dụng cho 3e) đều KHÔNG commit (gitignore không loại trừ `outputs/` thật ra — xem ghi chú mở phía dưới — nhưng theo quy ước dự án vẫn nên coi là derived data, không tin tưởng tuyệt đối nếu không tự kiểm tra lại) — cần chạy lại toàn bộ pipeline từ Thí nghiệm 2A trở đi nếu các file này không còn.

### Tóm tắt ngữ cảnh đầy đủ (để không phải đọc lại toàn bộ log)

Đã hoàn tất và CHỐT (không cần làm lại): RQ1 khóa, RQ2 khóa (backward sensitivity alignment > forward feature similarity, divergence từ Stage 3-4 — Thí nghiệm 2A/2B/2B.1/2C), Thí nghiệm 3A (causal intervention xác nhận Stage 3-4), Thí nghiệm 3B (object-conditioned weighting, kết hợp clip+mask tốt hơn từng phần riêng), Method v0.1 formalize đầy đủ (`research_plan.md` §9.1, khớp code `attacks/backward_reg_attack.py`), Ablation A (clip bound k=3 tối ưu, non-monotonic, có replicate xác nhận noise floor nhỏ), Ablation B (λ=0.5 tốt nhất nhưng không claim global optimum, có replicate xác nhận thứ hạng tương đối). Quyết định: dừng hyperparameter tuning, chuyển sang component/mechanism validation.

**Đang làm dở (roadmap 4 bước của user, ưu tiên 1→3, bỏ qua 2 tạm thời)**:
1. Component ablation (`experiment3e.py`) — CHƯA CÓ KẾT QUẢ n=300.
2. Mask/control ablation (object mask thật vs uniform vs random cùng diện tích) — CHƯA BẮT ĐẦU, đang hoãn theo yêu cầu ưu tiên 1→3.
3. Mechanism validation (`experiment3f.py`) — CHƯA CÓ KẾT QUẢ n=300.
4. Full baseline comparison (MI-FGSM/DI-FGSM/OSFD/TGR/MIG) — CHƯA BẮT ĐẦU.

**Ghi chú mở (không cấp thiết)**: `.gitignore` hiện KHÔNG loại trừ `outputs/` (chỉ loại `data/`, `checkpoints/`, `*.pth`, `*.pt`, `work_dirs/`, `*.npz`) — khác với giả định ngầm trong nhiều entry trước đây rằng "outputs/ không commit". Chưa ảnh hưởng gì (user tự quản lý commit), nhưng nếu sau này thấy `outputs/*.json`/`*.jsonl` xuất hiện trong `git status` ngoài ý muốn, đây là lý do — cân nhắc thêm `outputs/` vào `.gitignore` nếu không muốn commit chúng.

### Trạng thái git

Chưa commit gì trong toàn bộ session này kể từ lần cuối user tự commit (nếu có) — `git status --short` cho thấy nhiều file mới/sửa (`docs/progress_log.md`, `experiments/experiment3d_replicate.py`, `experiments/experiment3e.py`, `experiments/experiment3f.py`, và các `outputs/experiment3*/`). User biết và tự quản lý việc commit (đã nói trước đó "commit tôi tự làm được").

## 2026-09-21 — Resume máy GPU thuê mới, setup lại toàn bộ, hoàn tất Thí nghiệm 3E (Component Ablation) + 3F (Mechanism Validation) ở n=300

### Setup môi trường trên máy mới

Đúng như CLAUDE.md dự đoán — máy thuê lần này hoàn toàn mới (`.venv` cũ trên máy chỉ mới cài dở `torch`+`mmengine`, thiếu `mmcv`/`mmdet`/`mmpretrain`; không có `third_party/mmdetection`, `checkpoints/`, `data/coco/`). Đã chạy lại full pipeline: `scripts/setup_env.sh` → `scripts/download_dataset.sh` → tải lại 4 checkpoint (surrogate R50 + target R101/ConvNeXt-Tiny/Swin-Tiny, đúng URL trong model_registry.md) → `scripts/build_subset_annotations.py`. Verify: torch 2.1.2+cu118 (cuda_available=True), mmcv 2.1.0, mmdet 3.3.0, mmpretrain 1.2.0, GPU RTX 4000 Ada Generation (driver 560.35.03) — cùng model GPU với máy thuê lần trước (xem entry 2026-09-20), subset build ra đúng 1000 ảnh/7496 annotation khớp các lần trước. Cả 4 checkpoint load đúng backbone (ResNet/ResNet/ConvNeXt/SwinTransformer) qua `init_detector`.

Ghi chú kỹ thuật nhỏ (không cần sửa gì, chỉ lưu lại phòng gặp lại): trong lúc `mim install mmcv`/`mim install mmpretrain`, nội bộ `mim` gọi `get_torch_cuda_version()` bị crash (traceback in ra) do warning `Failed to initialize NumPy: _ARRAY_API not found` (numpy tạm thời bị 1 dependency của `openmim` kéo lên 2.x trước khi script kịp pin lại) — không fatal, `pip` vẫn tự fallback cài đúng version qua PyPI ngay sau đó, bước verify cuối `setup_env.sh` pass đầy đủ (kể cả `mmcv.ops.nms` chạy CUDA thật).

**Máy lần này chậm hơn hẳn về CPU** (Xeon Silver 4114 @ 2.2GHz, 40 core) so với máy thuê lần trước dù cùng GPU: tốc độ attack 10-iteration đo được chỉ ~0.25-0.26 ảnh/s (so với ~0.9-1.3 ảnh/s các lần chạy trước trên cùng loại GPU RTX 4000 Ada). Đã tự profile để xác nhận: GPU chỉ dùng ~21% utilization trong lúc chạy, không có process nào khác tranh GPU — đúng dấu hiệu **CPU launch-bound** (CPU đời cũ/xung nhịp thấp không "bơm" kịp lệnh CUDA cho vòng lặp 10 iteration × nhiều model × backward hook), không phải bug code hay cấu hình sai. Không có cách sửa an toàn (không đổi kết quả) để tăng tốc giữa lúc đang chạy — chấp nhận chạy chậm hơn, dùng tmux để không mất tiến độ nếu mất kết nối.

### Thí nghiệm 3E — Component Ablation (n=300, đầy đủ 5 setting)

Hoàn tất theo đúng thiết kế đã chốt trước đó (xem entry HANDOFF 2026-09-20): chỉ chạy mới 2 điểm `clip_s3`/`clip_s4`, tái sử dụng `baseline`/`clip_s3s4` từ `outputs/experiment3b/results.json` và `clip_s3s4_objw` (λ=0.5) từ `outputs/experiment3d/results.json`. Config: epsilon=8.0, num_iter=10, decay=1.0, objective=cls, k=3.

| Setting | ConvNeXt | Swin | CrossAvg | R101 | WhiteBox | TransferGap |
|---|---|---|---|---|---|---|
| baseline | 0.4507 | 0.3751 | 0.4129 | 0.7115 | 0.9536 | +0.2986 |
| clip_s3 (chỉ Stage 3) | 0.4525 | 0.3922 | 0.4223 | 0.7148 | 0.9590 | +0.2924 |
| clip_s4 (chỉ Stage 4) | 0.4513 | 0.3977 | 0.4245 | 0.7200 | 0.9583 | +0.2955 |
| clip_s3s4 (cả 2, đã có từ 3B) | 0.4631 | 0.3946 | 0.4289 | 0.7161 | 0.9603 | +0.2872 |
| clip_s3s4_objw (đã có từ 3D, λ=0.5) | 0.4714 | 0.4062 | 0.4388 | 0.7056 | 0.9637 | +0.2668 |

**Kết luận**: Stage 3 và Stage 4 đóng góp riêng lẻ với độ lớn gần tương đương nhau (CrossAvg +0.0094 và +0.0116 so với baseline), kết hợp cả hai (`clip_s3s4`) cho gain lớn hơn tổng ước lượng thô một chút (+0.016) — không phải 1 stage "gánh" toàn bộ hiệu ứng còn stage kia vô dụng, cả hai đều cần thiết. Object-weighting (`clip_s3s4_objw`) cộng thêm lợi ích rõ rệt nhất trong toàn bộ chuỗi (CrossAvg lên 0.4388, TransferGap xuống thấp nhất +0.2668) — khớp với kết luận đã có ở 3B. Kết quả: `outputs/experiment3e/results.json`.

### Thí nghiệm 3F — Mechanism Validation (n=300): tín hiệu từ sanity-test n=5 KHÔNG tái lập được ở mẫu đầy đủ

Đo cos(g_s, g_t) trước/sau khi áp regularization (Stage 3-4, k=3, λ=0.5, mode `clip_weight`) lên gradient của surrogate, cùng population object đã dùng ở Thí nghiệm 2A (`outputs/experiment2a/records.jsonl`, 299/300 ảnh dùng được, 4855 record).

| Target | mean_cos_before | mean_cos_after | delta |
|---|---|---|---|
| target_r101 (same-family) | 0.1247 | 0.1104 | −0.0143 |
| target_convnext_t (cross-CNN) | 0.0697 | 0.0658 | **−0.0039** |
| target_swin_t (CNN→Transformer) | 0.0360 | 0.0362 | +0.0002 |

Hypothesis (xem docstring `experiments/experiment3f.py`, đặt ra dựa trên sanity-test n=5 ở entry HANDOFF 2026-09-20: R101 delta=−0.0089, ConvNeXt delta=+0.0010, Swin delta=+0.0052 — cả 2 cross-family dương): regularization phải làm TĂNG cos_sim(g_s,g_t) rõ rệt ở ConvNeXt/Swin (cross-family) trong khi R101 (same-family) không cần tăng.

**Ở n=300 đầy đủ, tín hiệu này không tái lập được**: R101 vẫn âm đúng hướng (−0.0143), nhưng ConvNeXt lại ÂM (−0.0039, ngược dấu so với sanity n=5) và Swin chỉ dương cực nhỏ, gần như bằng 0 (+0.0002) — không đủ lớn để coi là "tăng gradient alignment cross-family" có ý nghĩa. Sanity-test n=5 trước đó ("tín hiệu rất đúng hướng hypothesis") hóa ra là nhiễu mẫu nhỏ, không phải xu hướng thật.

**Diễn giải quan trọng**: cơ chế "regularization tăng ASR cross-family BẰNG CÁCH tăng gradient alignment với target" — vốn là lý do thiết kế Thí nghiệm 3F để nối RQ2→RQ3 — **không được xác nhận trực tiếp bởi dữ liệu này**, dù bản thân việc regularization tăng ASR cross-family là thật (đã confirm chắc chắn qua 3A/3B/3C/3D/3E). Cần diễn giải cẩn trọng, KHÔNG viết "regularization hoạt động thông qua tăng gradient alignment" như 1 cơ chế đã chứng minh — có thể ASR tăng đến từ lý do khác (vd giảm variance/outlier gradient giúp attack ổn định hơn qua các bước iterate, ít bị "lệch hướng" bởi vài giá trị cực trị, không nhất thiết phải đo được bằng cos_sim tổng thể ở granularity per-object hiện tại). Kết quả: `outputs/experiment3f/records.jsonl`, `outputs/experiment3f/summary.json`.

### Bước tiếp theo

Theo roadmap 4 bước đã đặt ở entry HANDOFF 2026-09-20: (1) Component ablation — XONG (3E, kết quả rõ ràng, khớp hypothesis). (2) Mask/control ablation (object mask thật vs uniform vs random cùng diện tích) — vẫn CHƯA làm, đang hoãn. (3) Mechanism validation — XONG nhưng là **negative/inconclusive finding** (3F, không xác nhận được cơ chế gradient-alignment như kỳ vọng) — cần cân nhắc lại cách diễn giải "tại sao regularization giúp transfer" trước khi viết claim cơ chế vào research_plan.md/paper sau này, có thể cần đo thêm 1 metric khác (vd variance/norm của gradient trước-sau, không chỉ hướng) để tìm cơ chế thật. (4) Full baseline comparison (MI-FGSM/DI-FGSM/OSFD/TGR/MIG) — vẫn CHƯA bắt đầu.

Chưa cập nhật research_plan.md để phản ánh finding của 3F (negative) — cần hỏi user hướng diễn giải trước khi sửa phần method/RQ3 trong đó.

## 2026-09-21 — Thí nghiệm 3G: gradient concentration/tail (n=300) — mechanistic checkpoint mới sau negative finding của 3F

### Bối cảnh và thiết kế

Sau 3F (không xác nhận được "regularization tăng cos_sim(g_s,g_t)"), thống nhất với user: lùi lại 1 bước, hỏi câu hẹp và rẻ hơn trước — **regularization thực sự làm gì với backward signal của chính surrogate** (không cần target, không cần cos_sim với ai) — trước khi hỏi tiếp nó giúp transfer bằng cơ chế nào. Đây là bước #1 trong 4 bước đề xuất (order rẻ→sâu): (1) gradient concentration/tail — làm ở entry này, (2) iteration stability, (3) transformation consistency, (4) tuỳ kết quả.

Cách làm (tận dụng 1 tính chất quan trọng): `_make_reg_hook` (`attacks/backward_reg_attack.py`) là hàm THUẦN TÚY của `(grad, mask, k)`, không phụ thuộc gì khác trong graph — nên chỉ cần 1 lần forward+backward trên surrogate (KHÔNG cần target, KHÔNG cần chạy attack 10-iteration), bắt gradient RAW tại Stage 3/4 bằng "spy" hook (chỉ ghi lại, không sửa), rồi tính REG offline bằng cách gọi thẳng `_make_reg_hook` lên đúng RAW đó. Rẻ hơn hẳn 3E/3F (không cần load 3 target model, không cần vòng lặp attack) — tốc độ thực tế 1.38 ảnh/s (so với 0.25-0.55 ảnh/s của 3E/3F cùng máy). Code: `experiments/experiment3g.py`.

Đo 2 biến thể REG cùng lúc (rẻ, chỉ là 2 lệnh elementwise nữa trên cùng RAW đã có): `reg_clip` (mode "clip" thuần, = `reg_s3s4` 3A/3B, không mask) và `reg_clip_weight` (mode "clip_weight", k=3, λ=0.5 — đúng config method hiện tại, khớp 3D/3E/3F). Đơn vị phân tích: **1 ảnh = 1 row** (gradient của loss TOÀN ẢNH, không phải per-object như 2A/2C/3F) — tự nhiên không có vấn đề "nhiều object cùng ảnh không độc lập" mà user lưu ý, nên dùng thẳng Wilcoxon signed-rank (paired theo ảnh, n=300) không cần cluster-correction gì thêm.

6 thống kê trên toàn bộ tensor gradient mỗi stage: `std`, `max_abs`, `kurtosis` (excess, Fisher), `frac_outside_3sigma` (dùng μ,σ của CHÍNH tensor đó — xem giới hạn bên dưới), `top1pct_energy_frac` (tỷ lệ năng lượng nằm trong top 1% phần tử |giá trị| lớn nhất), `linf_over_l2` (proxy concentration rẻ).

### Kết quả (n=300, 600 record = 300 ảnh × 2 stage, Wilcoxon paired mọi p≈6e-51)

| Metric | Stage | raw | reg_clip | reg_clip_weight |
|---|---|---|---|---|
| std | 3 | 7.16e-05 | 3.83e-05 (−46%) | 3.91e-05 (−45%) |
| std | 4 | 9.49e-05 | 4.39e-05 (−54%) | 4.46e-05 (−53%) |
| max_abs | 3 | 0.005885 | 0.000215 (27×↓) | 0.000753 (7.8×↓) |
| max_abs | 4 | 0.006080 | 0.000285 (21×↓) | 0.000604 (10×↓) |
| kurtosis | 3 | 452.2 | 24.2 (19×↓) | 33.8 (13×↓) |
| kurtosis | 4 | 311.0 | 37.3 (8.3×↓) | 45.5 (6.8×↓) |
| top1%-energy | 3 | 0.712 | 0.356 (−50%) | 0.369 (−48%) |
| top1%-energy | 4 | 0.693 | 0.410 (−41%) | 0.416 (−40%) |
| L∞/L2 | 3 | 0.0439 | 0.0030 (14×↓) | 0.0107 (4.1×↓) |
| L∞/L2 | 4 | 0.0372 | 0.0047 (7.9×↓) | 0.0090 (4.1×↓) |
| frac_outside_3σ | 3 | 0.0162 | 0.0309 (tăng ~1.9×) | 0.0305 (tăng ~1.9×) |
| frac_outside_3σ | 4 | 0.0161 | 0.0279 (tăng ~1.7×) | 0.0277 (tăng ~1.7×) |

### Kết luận đã chốt (user, 2026-09-21)

> Clipping không làm gradient biến mất (std chỉ giảm ~45-54%), nhưng giảm cực mạnh concentration/tail dominance: max_abs giảm 8-27×, kurtosis giảm 6.8-19×, top-1% energy giảm ~40-50%, L∞/L2 giảm ~4-14×. Effect nhất quán ở cả Stage 3 và 4, Wilcoxon paired theo ảnh cho p cực nhỏ.

**Giới hạn đã ghi nhận**: `frac_outside_3sigma` đi NGƯỢC chiều dự đoán (tăng ~1.7-1.9× sau reg) — do dùng ngưỡng tự tham chiếu (μ,σ của chính tensor sau khi đã bị nén nhỏ lại), không nên dùng làm evidence chính. Nếu cần một tail-fraction metric trong tương lai, nên dùng ngưỡng CỐ ĐỊNH lấy từ phân phối RAW (vd `P(|g_reg − μ_raw| > 3σ_raw)`) hoặc percentile threshold lấy từ raw áp cho cả raw/reg, thay vì để mỗi tensor tự định nghĩa ngưỡng của chính nó — CHƯA làm lại, để dành nếu cần dùng metric này về sau. 4/5 metric còn lại đồng thuận mạnh, không phụ thuộc vào điểm giới hạn này.

**Gate đạt**: reduction mạnh, ổn định, không phải tautology thuần túy (std không sụp về 0) → đủ điều kiện sang bước #2 (iteration stability). Kết quả: `outputs/experiment3g/records.jsonl`, `outputs/experiment3g/summary.json`.

### Thí nghiệm 3H — Iteration stability (n=300): có ý nghĩa thống kê nhưng effect size khiêm tốn, không mạnh như 3G

Chạy attack thật 10-iteration (baseline vs reg, cùng config k=3/λ=0.5/clip_weight/Stage 3-4), log mỗi iteration `grad_norm` (tensor thực sự được cộng dồn vào momentum trong attack thật). Đơn vị: 1 ảnh = 1 row (không cần target model). Code: `experiments/experiment3h.py`.

| Metric | baseline | reg | diff | p (Wilcoxon paired) | Đúng chiều? |
|---|---|---|---|---|---|
| mean_cos_consecutive | 0.1419 | 0.1731 | +0.0312 | 1.18e-14 | ✓ |
| mean_sign_flip_rate | 0.384 | 0.3824 | −0.0016 | 5.03e-09 | ✓ (cực nhỏ) |
| mean_rel_change_norm | 1.283 | 1.257 | −0.0259 | 6.63e-12 | ✓ |
| mean_cos_drift | 0.0111 | 0.0186 | +0.0075 | 2.32e-11 | ✓ |
| final_cos_drift (g1 vs g10) | −0.0104 | −0.0112 | −0.0007 | 0.139 (không ý nghĩa) | ✗ ngược nhẹ |

**Kết luận đã chốt (user, 2026-09-21)**: 4/5 metric có p cực nhỏ và đúng chiều, nhưng đây là hệ quả của n=300 cặp rất nhất quán (power cao) chứ KHÔNG phải effect lớn — nhìn độ lớn tuyệt đối, `sign_flip_rate` gần như không đổi (38.4%→38.2%), `cos_drift` vẫn rất gần 0 ở cả 2 phía. `final_cos_drift` (so trực tiếp g1 với g10) KHÔNG có ý nghĩa và hơi ngược chiều — hiệu ứng "bớt trôi dạt" rõ ở các iteration giữa nhưng không giữ được tới bước cuối.

> **Iteration stability is a secondary consequence of backward regularization, not sufficient evidence for the primary mechanism.**

Quyết định: KHÔNG nhảy sang transformation consistency (#3, đắt hơn, xa intervention hơn, nguy cơ "tìm metric cho đẹp"). Thay vào đó làm **Thí nghiệm 3I** — linkage test rẻ và targeted hơn, xem bên dưới. Kết quả: `outputs/experiment3h/records.jsonl`, `outputs/experiment3h/summary.json`.

### Thí nghiệm 3I — Linkage test (n=300): kết quả sạch nhất trong chuỗi 3F→3I, đặc thù cross-family

Câu hỏi (thu hẹp từ mechanism story chung, thống nhất với user 2026-09-21, sau khi 3F negative và 3H yếu): thay vì tiếp tục tìm cơ chế tổng quát (cosine alignment — bác bỏ ở 3F; trajectory stability — yếu ở 3H), hỏi trực tiếp và hẹp hơn: **per-image reduction in gradient concentration (ΔC, đã có sẵn từ 3G) có tương quan với việc 1 object "giành lại" được transfer (baseline fail → reg success) hay không?**

Thiết kế rẻ nhờ tái sử dụng tối đa dữ liệu đã có: `evaded_baseline` per object/target lấy thẳng từ `outputs/experiment2a/records.jsonl` (baseline MI-FGSM, cùng config), `ΔC(image)` lấy thẳng từ `outputs/experiment3g/records.jsonl` (raw − reg_clip_weight, trung bình Stage 3+4, metric `top1pct_energy_frac`) — cả 2 KHÔNG cần tính lại. Việc MỚI duy nhất: chạy attack REG (Stage 3-4, k=3, λ=0.5, clip_weight — dùng thẳng `iterative_linf_attack_reg` có sẵn) trên surrogate cho 300 ảnh, predict trên 3 target để lấy `evaded_reg` per object, so khớp object_idx với population 2A (không cần predict lại ảnh sạch). Code: `experiments/experiment3i.py`.

Đơn vị phân tích: **1 (ảnh, target) = 1 row** — với mỗi ảnh/target có ≥1 object baseline-fail, tính `gained_rate = n_gained / n_baseline_fail` (gained = object F→T). Test: Mann-Whitney U so ΔC giữa nhóm ảnh `gained_rate>0` vs `gained_rate==0` (độc lập hoàn toàn giữa các ảnh, không cần cluster-correction) + Spearman correlation(ΔC, gained_rate).

**Kết quả (n=300, 299 ảnh dùng được, 723 record):**

| Target | n (ảnh,target) | ΔC(ảnh có gain) | ΔC(ảnh không gain) | MWU p (1 phía) | Spearman ρ | Spearman p |
|---|---|---|---|---|---|---|
| target_r101 (same-family, **control**) | 190 | 0.337 | 0.321 | 0.199 (không ý nghĩa) | 0.046 | 0.527 (không ý nghĩa) |
| target_convnext_t (cross-CNN) | 260 | 0.378 | 0.301 | **1.32e-04** | **0.205** | **8.84e-04** |
| target_swin_t (CNN→Transformer) | 273 | 0.366 | 0.303 | **1.51e-04** | **0.187** | **1.88e-03** |

### Kết luận đã chốt (user, 2026-09-21) — kết quả sạch nhất trong chuỗi

> Ảnh có ΔC lớn hơn có tỷ lệ "giành lại" object bị né tránh cao hơn — nhưng CHỈ đúng ở 2 target cross-family (ConvNeXt, Swin), KHÔNG đúng ở R101 (same-family, đối chứng). Đây là pattern đặc thù cross-family, không phải hiệu ứng attack mạnh lên chung chung.

Chuỗi cơ chế an toàn để dùng, KHÔNG cần cosine alignment (3F, bác bỏ) hay trajectory stability tổng quát (3H, yếu) làm mắt xích bắt buộc:

$$
\boxed{\text{Suppressing extreme backward-gradient concentration at Stage 3-4} \rightarrow \text{higher per-image rate of newly-evaded objects, specifically for cross-family targets}}
$$

**Tổng kết cả chuỗi 3F-3I (mechanism cho method v0.1)**:
- 3F (cosine alignment với target) — **negative**, không dùng làm mechanism.
- 3G (gradient concentration/tail tại surrogate) — **strong**, giảm mạnh và nhất quán (p≈6e-51).
- 3H (iteration trajectory stability) — **weak but consistent**, effect size nhỏ, không giữ được tới iteration cuối.
- 3I (linkage ΔC ↔ transfer gain, per ảnh/target) — **strong và đặc thù cross-family**, kết quả rõ nhất, có đối chứng same-family sạch.

Kết quả: `outputs/experiment3i/records.jsonl`, `outputs/experiment3i/summary.json`.

### Bước tiếp theo

Cân nhắc cập nhật `research_plan.md` §9.1/RQ3 để phản ánh chuỗi cơ chế cuối cùng (3G+3I là bằng chứng chính, 3F/3H là phụ/negative — không viết "tăng gradient alignment" như cơ chế nữa). Sau đó chuyển sang roadmap còn lại: mask/control ablation (bước 2, đang hoãn) + full baseline comparison với TGR/MIG/OSFD/DI-FGSM (bước 4, chưa bắt đầu) — theo đúng quyết định đã thống nhất, không đào thêm transformation consistency.

**Cập nhật ngay sau đó, cùng ngày**: đã cập nhật `research_plan.md` §9.2 (Mechanism, mới) + RQ2 (thêm ghi chú phạm vi, phân biệt "diagnostic RQ2" với "method mechanism") + RQ3 (trả lời có điều kiện, chưa khóa hoàn toàn) — xem file đó để có công thức đầy đủ. Tóm tắt: claim an toàn dùng cho paper là *"Suppressing extreme mid/deep backward-gradient concentration is strongly associated with improved transfer specifically to cross-family targets, while no corresponding association is observed for the same-family control."*

## 2026-09-21 — Đổi ưu tiên: OSFD matched-budget comparison TRƯỚC mask/control ablation — kết quả bất lợi cho method v0.1

### Quyết định đổi roadmap

User yêu cầu dừng `experiment3j.py` (mask/control ablation, đang chạy dở — đã kill tmux job, không mất gì vì chưa có kết quả n=300 nào) để ưu tiên **so sánh với baseline mạnh trước khi freeze method**, đặc biệt OSFD (Wu et al., AAAI 2024) — lý do: mọi kết quả tới giờ (3A-3I) chỉ so với baseline của chính mình, chưa trả lời được "method có đáng làm contribution chính không". Roadmap mới: `Freeze v0.1 → Strong baseline comparison (n=300) → {competitive: mask ablation + n=1000 | yếu: phân tích/redesign}`.

User tự clone official OSFD repo (`github.com/wakuwu/OSFD`) vào `/workspace/OSFD` để đọc trực tiếp source code, không đoán từ paper text.

### Đọc code OSFD — phát hiện quan trọng về threat model

Đọc `attack/ours/OSFD.py` + `attack/base/RRB.py` + `attack/Attack.py`:
- Loss thật: `MSE(k * feat_clean, feat_adv)` (k=3.0), tính trên **toàn bộ 4 stage backbone**, KHÔNG mask theo GT box. "Object-Aware" trong tên paper nằm ở **RRB** (Random Rotation + adaptive Resizing + Blur — 1 base attack họ combine cùng MI), không phải ở loss: mỗi iteration tạo 2 "view" nối tiếp (rotate quanh 1 GT box ngẫu nhiên → resize thích ứng theo 1 GT box ngẫu nhiên khác, áp lên view đã rotate) rồi blur, loss cộng dồn qua cả 2 view.
- **Threat model gốc khác hẳn dự án**: `steps=10` mỗi epoch nhưng `max_epoch=20`, noise carry-over giữa epoch (buffer) → tổng ~200 bước gradient trong cùng 1 quả epsilon (không phải epsilon nhân theo epoch). `epsilon=5` (không phải 8 như dự án). Native-config đầy đủ (2000 ảnh VOC × 200 bước) không khả thi thời gian trên máy hiện tại (CPU-bound, đã biết từ đầu session).
- **Quyết định scope (thống nhất với user)**: (1) **OSFD-matched** — port đúng thuật toán (loss+RRB+MI) vào pipeline hiện tại, chạy epsilon=8/num_iter=10 y hệt mọi thí nghiệm khác, PRIMARY comparison. (2) **OSFD-extended** — budget lớn hơn (chưa làm), sanity check phụ, KHÔNG dùng để claim thắng/thua. KHÔNG dựng lại native env (mmdet 2.28.2 cũ) trừ khi cần fidelity-check riêng.

### Port `attacks/osfd_attack.py` + `experiments/experiment4a.py`

Port trực tiếp từ code gốc, giữ nguyên hyperparameter RRB/k từ `config/attack_faster_rcnn.yaml` (k=3.0, theta=7.0, l_s=10, rho=0.8, s_max=1.10, sigma=6.0). Khác biệt CỐ Ý (matched-budget): epsilon=8.0, num_iter=10, alpha=epsilon/num_iter thay vì epsilon=5 + alpha=1.0 cố định + 200 bước.

**Bug tự phát hiện + fix trước khi tin số liệu** (user yêu cầu audit lại logic trước khi chốt kết quả — đúng, vì kết quả ban đầu bất ngờ/bất lợi): review từng dòng so với code gốc, verify bằng số học công thức loss (`F.mse_loss` gộp batch N view × N == tổng N `mse_loss` riêng lẻ — verify bằng script Python, khớp tuyệt đối). Phát hiện 1 lỗi thật: `_random_axis_rotation` dùng `[W/2, H/2]` (đúng thứ tự x,y mà `torchvision.transforms.functional.rotate` cần) trong khi code GỐC dùng `[H//2, W//2]` (ĐẢO trục — 1 quirk/bug thật trong `attack/base/RRB.py` dòng 64, không phải cách hiểu sai của mình). Vì mục tiêu port là fidelity với thuật toán họ THỰC SỰ chạy, đã sửa lại để REPLICATE ĐÚNG behavior gốc (kể cả quirk này), không dùng bản "đã sửa lỗi hộ họ". Re-run n=300 sau fix — chênh lệch với bản trước fix chỉ ~0.5-1 điểm % ở mọi ô (nằm trong biên độ nhiễu run-to-run đã biết), xác nhận bug không phải nguyên nhân chính của kết quả bất lợi. Bản trước fix lưu tại `outputs/experiment4a/results_pre_rotation_fix.json` (tham khảo, không dùng).

### Kết quả (n=300, sau fix — số liệu chính thức)

| Method | WhiteBox | R101 | ConvNeXt | Swin | CrossAvg | TransferGap |
|---|---|---|---|---|---|---|
| MI-FGSM (Exp1B n=300, Run B) | 0.9563 | 0.7082 | 0.4530 | 0.3751 | 0.4141 | +0.2942 |
| **OSFD-matched** | 0.8440 | 0.7036 | 0.5629 | 0.5131 | **0.5380** | +0.1656 |
| DI-FGSM (Exp1B n=300, Run E) | 0.8998 | 0.7731 | 0.6273 | 0.5308 | **0.5791** | +0.1941 |
| **Ours v0.1** (clip_s3s4_objw, λ=0.5, Exp3E) | 0.9637 | 0.7056 | 0.4714 | 0.4062 | **0.4388** | +0.2668 |

### Kết luận — theo đúng gate đã đặt trước khi chạy: KHÔNG freeze method

Đọc đúng thứ tự ưu tiên (ConvNeXt → Swin → CrossAvg → R101 → WhiteBox, KHÔNG phải WhiteBox):
- **OSFD thắng rõ Ours ở cả ConvNeXt (0.563 vs 0.471, +0.092) và Swin (0.513 vs 0.406, +0.107)** — không phải thắng nhẹ.
- **DI-FGSM (baseline đơn giản, không có thiết kế mechanism) còn thắng cả OSFD lẫn Ours trên CrossAvg** (0.579 — cao nhất bảng).
- WhiteBox của Ours cao nhất (0.964) nhưng đây KHÔNG phải tiêu chí quan trọng (đã thống nhất từ đầu).

Theo gate: **"OSFD vượt xa → chưa freeze method"**. Giả thuyết làm việc (chưa verify): điểm chung giữa OSFD (RRB) và DI-FGSM (resize+pad) là **input-transformation/augmentation trong lúc tấn công** — method v0.1 hoàn toàn không có augmentation nào (chỉ backward regularization tĩnh). 2 cơ chế này (augmentation vs backward-concentration-suppression) CÓ THỂ bổ trợ nhau thay vì loại trừ.

Kết quả: `outputs/experiment4a/results.json`, `outputs/experiment4a/raw_predictions.json`.

### Bước tiếp theo

Chưa quyết — 3 hướng đã đề xuất cho user chọn: (1) phân tích ablation OSFD-loss KHÔNG RRB (cô lập đóng góp riêng của feature-distortion loss vs augmentation), (2) thử kết hợp backward-reg (method mình) + input-diversity (kiểu DI-FGSM/RRB) xem có cộng hưởng không, (3) khác. `experiments/experiment3j.py` (mask/control ablation) vẫn còn nguyên, CHƯA XONG (đã kill giữa chừng để ưu tiên việc này) — cần quay lại sau khi quyết định hướng method.
