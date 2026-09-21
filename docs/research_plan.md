# Kế hoạch nghiên cứu — Tấn công Né tránh (Evasion) có khả năng Chuyển giao Liên họ Kiến trúc (Cross-Family) cho Object Detection, dùng Single-Surrogate

## 1. Mục tiêu nghiên cứu

Nghiên cứu **các tấn công né tránh (evasion attack) có khả năng chuyển giao (transferable)** cho object detection, trong bối cảnh **black-box với đúng một mô hình surrogate (thay thế) duy nhất**.

Mục tiêu cốt lõi:

> Chỉ với một detector surrogate duy nhất, tạo ra nhiễu đối kháng (adversarial perturbation) có khả năng chuyển giao hiệu quả sang nhiều detector mục tiêu (target) chưa từng thấy, mà các detector này có **kiến trúc feature-extractor thuộc các họ khác nhau**, đặc biệt là chuyển từ CNN → Transformer.

Câu hỏi nghiên cứu chính không đơn thuần là liệu adversarial example có chuyển giao được hay không, mà là:

> Làm thế nào để giảm "khoảng cách chuyển giao" (transferability gap) khi surrogate và target dùng các feature extractor về bản chất khác nhau?

---

## 2. Định nghĩa bài toán

### Đầu vào

- Ảnh sạch \(x\)
- Nhãn ground-truth \(y\)
- Một detector surrogate \(f_s\)

Ví dụ surrogate:

- Faster R-CNN + ResNet-50

### Đầu ra

Tạo ra:

\[
x_{adv} = x + \delta
\]

với ràng buộc:

\[
\|\delta\|_{\infty} \le \epsilon
\]

sao cho ảnh đối kháng làm giảm đáng kể hiệu năng detection trên:

- mô hình surrogate
- các target black-box cùng họ (same-family)
- các target black-box khác họ (cross-family)

mà **không truy cập**:

- gradient của target
- trọng số của target
- query đến target trong quá trình tạo nhiễu

---

## 3. Định nghĩa "Cross-Family" (liên họ)

Nghiên cứu cần phân biệt rõ:

### Họ feature-extractor

Ví dụ:

- ResNet
- DarkNet / CSP
- ConvNeXt
- Swin Transformer
- PVT / ViT

### Kiến trúc detector

Ví dụ:

- Faster R-CNN
- FCOS
- YOLO
- RetinaNet
- Deformable DETR
- DINO

Điểm quan trọng:

> Một detector head dạng Transformer không nhất thiết đồng nghĩa với feature extractor là Transformer.

Ví dụ:

- Deformable DETR + ResNet-50 vẫn dùng feature extractor là CNN.

Do đó trọng tâm nghiên cứu chính là:

\[
\text{Khả năng chuyển giao xuyên Feature-Extractor}
\]

chứ không chỉ là chuyển giao xuyên kiến trúc detector.

---

## 4. Dataset

Dataset chính:

- **MS COCO**

Thiết lập ban đầu đề xuất:

- Tập validation của COCO
- Dùng subset khoảng **500–1.000 ảnh** để thực nghiệm nhanh
- Mở rộng sang protocol đánh giá đầy đủ sau khi có kết quả khả quan

Tất cả các mô hình so sánh nên:

- được pretrain trên COCO
- dùng cùng bộ annotation đánh giá
- có điều kiện tiền xử lý (preprocessing) tương đương

---

## 5. Chỉ số đánh giá

Báo cáo tối thiểu:

- Clean mAP
- Adversarial mAP
- \(\Delta\)mAP
- Attack Success Rate (ASR)

Tách kết quả theo:

- chuyển giao cùng họ (same-family)
- chuyển giao xuyên họ CNN (cross-CNN-family)
- chuyển giao CNN → Transformer

Chỉ số tùy chọn:

\[
TransferGap =
ASR_{same-family}
-
ASR_{cross-family}
\]

Một phương pháp thành công không chỉ tăng ASR trung bình, mà lý tưởng còn phải:

\[
TransferGap \downarrow
\]

---

# 6. Thí nghiệm 1 — Xác minh khoảng cách chuyển giao xuyên họ (Transfer Gap)

## Mục tiêu

Trước khi đề xuất phương pháp mới, cần xác minh tiền đề cốt lõi:

> Khả năng chuyển giao đối kháng có giảm khi họ feature-extractor thay đổi không?

Thí nghiệm này phải cô lập tối đa ảnh hưởng của backbone.

---

## 6.1 Surrogate

**Pivot 2026-09-19** (xem docs/progress_log.md entry cùng ngày để biết đầy đủ lý do): đổi từ Faster R-CNN sang **Mask R-CNN**, lý do ở §6.2 bên dưới. Backbone surrogate vẫn giữ nguyên ResNet-50 như dự định ban đầu.

Dùng một surrogate cố định:

- Mask R-CNN + ResNet-50 (chỉ dùng box output/loss cho tấn công và đánh giá, không dùng nhánh mask)

---

## 6.2 Các mô hình Target

Ưu tiên các target giữ nguyên kiến trúc detector, chỉ thay đổi feature extractor.

**Pivot 2026-09-19**: đã verify trực tiếp trong MMDetection v3.3.0 (`configs/convnext/`, `configs/swin/`) — **không tồn tại** cấu hình Faster R-CNN + ConvNeXt hoặc Faster R-CNN + Swin (2 backbone này trong MMDetection chỉ có ở Mask R-CNN / Cascade Mask R-CNN / RetinaNet). Để giữ đúng nguyên tắc "chỉ đổi feature extractor, không đổi kiến trúc detector" trên cả 3 target (thay vì chỉ 1/3), chuyển toàn bộ — cả surrogate lẫn target — sang **Mask R-CNN**. So sánh trực tiếp phần bbox_head/rpn_head giữa base config Faster R-CNN và Mask R-CNN trong MMDetection cho thấy nhánh box detection giống hệt nhau (cùng RPN, cùng bbox head/loss); Mask R-CNN chỉ thêm nhánh mask_head song song không tham gia vào box prediction. Khác biệt kiến trúc còn lại chỉ là `match_low_quality` trong RPN assigner lúc train (True ở Mask R-CNN, False ở Faster R-CNN) — ảnh hưởng nhỏ, không phải khác kiến trúc detector.

So sánh có kiểm soát được đề xuất (đã verify, xem docs/model_registry.md):

1. Mask R-CNN + ResNet-101
   - Cùng họ backbone

2. Mask R-CNN + ConvNeXt-Tiny
   - Họ backbone CNN khác
   - Lưu ý: checkpoint chính thức duy nhất trong MMDetection dùng schedule 3x + AMP + multi-scale crop, không phải 1x như 2 target còn lại — chênh lệch protocol train (không phải kiến trúc) giữa target này và các target khác, xem model_registry.md phần ghi chú mở.

3. Mask R-CNN + Swin-Tiny
   - CNN → Transformer
   - Checkpoint chính thức dùng schedule 1x, khớp với surrogate và target ResNet-101.

Nếu mô hình chuẩn không có sẵn, chọn mô hình tương đương gần nhất có trong MMDetection.

Các thí nghiệm sau có thể thêm:

- FCOS
- YOLO
- Deformable DETR
- DINO

nhưng Thí nghiệm 1 cần tối thiểu hóa các biến gây nhiễu (confounding variables) trước.

---

## 6.3 Baseline tấn công

Dùng một tấn công chuyển giao đã được kiểm chứng làm baseline ban đầu.

Yêu cầu:

- cùng một tấn công cho mọi target
- cùng ngân sách nhiễu (perturbation budget)
- cùng số vòng lặp
- cùng step size
- cùng tiền xử lý ảnh
- cùng một tập ảnh đối kháng được đánh giá trên mọi target

Không tối ưu riêng cho từng target.

---

## 6.4 Mô hình đe dọa (Threat Model)

Dùng:

\[
L_{\infty}
\]

Ngân sách nhiễu ban đầu đề xuất:

\[
\epsilon = 8/255
\]

trừ khi baseline paper dùng thiết lập chuẩn khác cần tái lập chính xác.

Các mô hình target luôn là black-box tuyệt đối trong quá trình sinh adversarial example.

---

## 6.5 Điều kiện đánh giá

Giữ cố định:

- dataset
- subset ảnh
- độ phân giải ảnh (nếu có thể)
- perturbation budget
- số vòng lặp tấn công
- step size
- tiền xử lý
- cách cài đặt đánh giá

Để phân tích ASR công bằng, nên chỉ đánh giá trên các object/ảnh được detect đúng ở điều kiện sạch (clean).

Nếu tấn công có tính ngẫu nhiên:

- chạy nhiều seed
- báo cáo mean và độ lệch chuẩn

---

## 6.6 Quan sát kỳ vọng

Giả thuyết:

\[
Transfer(R50 \rightarrow R101)
>
Transfer(R50 \rightarrow ConvNeXt)
>
Transfer(R50 \rightarrow Swin)
\]

Thứ tự chính xác không nhất thiết phải đúng tuyệt đối.

Điều kiện quan trọng là:

> Chuyển giao xuyên họ phải cho thấy sự suy giảm rõ ràng và lặp lại được so với chuyển giao cùng họ.

---

## 6.7 Quy tắc quyết định

### Nếu transfer gap rõ ràng

Tiến hành Thí nghiệm 2:

> Tìm hiểu tại sao khả năng chuyển giao xuyên họ giảm.

Điều tra:

- gradient alignment
- feature similarity
- feature importance
- saliency / object evidence
- đặc tính tần số (frequency)
- tính dễ tổn thương theo từng layer

### Nếu transfer gap yếu hoặc không nhất quán

Chưa vội thiết kế tấn công mới.

Kiểm tra trước:

- việc chọn mô hình
- subset dataset
- độ chồng lấp detect đúng ở clean
- cường độ tấn công
- khác biệt tiền xử lý
- các thay đổi ở detector head gây nhiễu

Sau đó xem lại giả thuyết nghiên cứu.

---

## 6.8 Thí nghiệm 1B — Kiểm tra confound, xác nhận quy tắc quyết định §6.7

> **⚠️ SUPERSEDED (2026-09-20, cùng ngày, phát hiện sau)**: toàn bộ số liệu Thí nghiệm 1 + 1B mô tả trong mục này (bao gồm cả bảng trong progress_log.md) được tính từ 1 attack có **bug tọa độ RoI** (GT bbox ở tọa độ ảnh gốc, chưa scale theo `scale_factor` để khớp `feats` tính từ ảnh đã resize — xem docs/progress_log.md entry "Phát hiện + fix bug tọa độ RoI"). Bug làm attack yếu hơn thật rất nhiều (whitebox ASR chỉ ~33% thay vì ~95% sau khi fix). Đã fix bug, re-run Exp1+1B ở n=300 (tier "confirm"), số liệu mới đầy đủ ở progress_log.md entry cùng tên nêu trên. **Kết luận định tính bên dưới (transfer gap tồn tại, thứ tự same>cross-CNN>CNN→Transformer, đủ điều kiện sang Thí nghiệm 2) vẫn ĐÚNG và còn RÕ RÀNG HƠN sau khi fix** — chỉ có con số ASR/TransferGap cụ thể trong các bullet dưới đây là dựa trên bản có bug, đọc bằng progress_log.md để lấy số liệu đúng.

**Cập nhật 2026-09-20**: Thí nghiệm 1 gốc (§6, kết quả đầy đủ ở docs/progress_log.md entry "Kết quả Thí nghiệm 1") cho transfer gap rõ ràng, nhưng tự flag 3 giới hạn có thể là confound thay vì do kiến trúc backbone: (1) model-strength, (2) attack cụ thể (MI-FGSM), (3) attack objective (chỉ classification). Thí nghiệm 1B kiểm tra cả 3, cộng thêm 1 kiểm tra bổ sung (augmentation-based attack mạnh hơn, theo §11) — 4 run trên cùng subset 1000 ảnh cố định, chi tiết đầy đủ + bảng số liệu ở docs/progress_log.md entry "Kết quả Thí nghiệm 1B":

- **Run B** (model-strength): mở rộng model zoo trong cùng họ ResNet/ResNeXt (thêm ResNeXt-101, ResNet-50 train 3x). Kết quả: `target_r50_3x` (cùng backbone surrogate, train mạnh hơn) có ASR cao hơn hẳn 2 target same-family còn lại dù clean AP nằm giữa — model-strength không đơn điệu với ASR trong cùng họ, loại được confound này làm nguyên nhân chính.
- **Run C** (attack robustness — BIM/I-FGSM, bỏ momentum): thứ tự same-family > cross-CNN > CNN→Transformer giữ nguyên.
- **Run D** (objective robustness — thêm loss bbox): pattern gần như trùng khớp Run B/Thí nghiệm 1 gốc.
- **Run E** (DI-FGSM, input diversity — baseline augmentation-based theo §11): ASR tăng ở mọi model nhưng thứ tự vẫn giữ; TransferGap tuyệt đối co lại phần nào (đặc biệt so ConvNeXt) nhưng không biến mất.

**Kết luận**: TransferGap (định nghĩa §5) dương và có ý nghĩa ở cả 4 run, không phụ thuộc việc đổi model-zoo/attack/objective/augmentation. Theo đúng quy tắc quyết định §6.7 ("nếu transfer gap rõ ràng → tiến hành Thí nghiệm 2"), tiền đề cốt lõi giờ đã được xác nhận vững hơn (không chỉ 1 thiết lập cụ thể như sau Thí nghiệm 1 gốc) — **đủ điều kiện chuyển sang Thí nghiệm 2 (§7)**.

**Giới hạn còn lại, chưa giải quyết bởi Thí nghiệm 1B**: confound training-recipe riêng của ConvNeXt-Tiny (checkpoint duy nhất dùng schedule 3x+AMP+ms-crop, không có bản 1x để so sánh — xem model_registry.md) vẫn chưa được cô lập; toàn bộ 4 run dùng đúng 1 seed/1 subset, chưa có nhiều seed để báo cáo mean/std.

**Bằng chứng độc lập củng cố tiền đề**: Winter et al., "Benchmarking Adversarial Robustness and Adversarial Training Strategies for Object Detection", arXiv:2602.16494 (nộp 2026-02-18) báo cáo hiện tượng "modern adversarial attacks... significant lack of transferability to transformer-based architectures" — khớp trực tiếp với phát hiện của Thí nghiệm 1/1B, độc lập với thiết lập trong dự án này. Nên cân nhắc dùng làm 1 baseline/related-work khi so sánh ở §11.

---

# 7. Thí nghiệm 2 — Tìm cơ chế (Mechanism)

Mục tiêu là xác định thuộc tính đo lường được nào tương quan với khả năng chuyển giao.

Các hướng ứng viên:

### A. Gradient alignment

Đo xem:

\[
Similarity(
\nabla_x L_s,
\nabla_x L_t
)
\]

có giảm theo họ kiến trúc không.

**Thiết kế Thí nghiệm 2A (chốt 2026-09-20)** — thứ tự rẻ → sâu, gradient alignment làm bước đầu tiên vì chi phí tính toán thấp nhất so với feature-level/saliency:

1. Với cùng ảnh sạch/object, tính cosine similarity giữa gradient của surrogate và từng target:
   \[
   \cos(g_s, g_t) = \frac{g_s^\top g_t}{\|g_s\| \|g_t\|}
   \]
   Giả thuyết thứ tự: \(\cos(R50, R101) > \cos(R50, ConvNeXt) > \cos(R50, Swin)\).
2. **Không** chỉ tương quan ở mức 3 điểm (3 target), vì quá ít điểm để kết luận thống kê. Tính **per-image hoặc per-object**, sau đó so sánh phân phối (distribution) của \(\cos(g_s, g_t)\) giữa nhóm object **evaded** (né tránh thành công sau attack) và nhóm **not evaded** — kiểm tra:
   \[
   GradientSimilarity \uparrow \iff TransferSuccess \uparrow
   \]
   ở mức per-object, không chỉ mức tổng hợp per-model.
3. Rẽ nhánh theo kết quả:
   - Nếu gradient alignment giải thích được gap (pattern rõ như `R50→R101` similarity cao/ASR cao, `R50→ConvNeXt` giữa, `R50→Swin` thấp/ASR thấp) → đi tiếp xuống **feature-level alignment** (mục B) để tìm layer nào gây divergence.
   - Nếu gradient alignment yếu/không rõ → chuyển sớm sang **object evidence / saliency / transformation consistency** (mục C, D), không cố bám vào gradient alignment nếu dữ liệu không ủng hộ.

Chuỗi suy luận mong muốn của Thí nghiệm 2A:

\[
\boxed{\text{Architecture family} \rightarrow \text{gradient alignment} \rightarrow \text{transfer success}}
\]

Nếu đạt được, đây là bước chuyển từ "empirical transfer gap" (Thí nghiệm 1/1B) sang **mechanistic explanation** — nền tảng trực tiếp để đặt câu hỏi trung tâm cho việc thiết kế method: **làm sao từ một surrogate duy nhất tạo ra gradient/feature direction ít architecture-specific hơn?**

---

### B. Feature similarity

So sánh biểu diễn trung gian giữa surrogate và target.

Điều tra:

- layer nông (shallow)
- layer giữa (middle)
- layer sâu (deep)

Câu hỏi:

> Mức layer nào dự đoán tốt nhất khả năng chuyển giao xuyên họ?

---

### C. Object-conditioned saliency / evidence

Với mỗi object phát hiện được \(o\), tính:

\[
E(x,o)
\]

bằng phương pháp giải thích hoặc attribution phù hợp.

Kiểm tra xem:

\[
EvidenceSimilarity(f_s,f_t)
\]

có tương quan với:

\[
Transferability(f_s \rightarrow f_t)
\]

hay không.

---

### D. Transformation consistency

Ước lượng feature/evidence nào của surrogate ổn định qua các phép biến đổi:

\[
T_1(x), T_2(x), ..., T_K(x)
\]

Các phép biến đổi khả dĩ:

- resize
- translation
- crop
- scale
- input diversity
- feature dropout
- frequency perturbation

Giả thuyết:

> Feature hoặc object evidence ổn định qua nhiều "view" có thể ít đặc thù cho surrogate hơn và chuyển giao tốt hơn xuyên kiến trúc.

---

# 8. Chiến lược phát triển phương pháp

**Không** cố định phương pháp cuối cùng từ trước.

Dùng vòng lặp sau:

1. Tìm literature
2. Đặt giả thuyết
3. Cài đặt test đơn giản nhất
4. Chạy thí nghiệm có kiểm soát
5. Kiểm tra hiệu ứng có lặp lại được không
6. Thực hiện ablation
7. Chỉ khi đó mới xây phương pháp mạnh hơn quanh cơ chế đã được xác nhận
8. Sau đó mới phát triển lý thuyết / biện luận phân tích

Luồng làm việc là:

\[
Search
\rightarrow
Idea
\rightarrow
Experiment
\rightarrow
Evidence
\rightarrow
Method
\rightarrow
Theory
\]

Không ép lý thuyết trước khi xác nhận hiện tượng bằng thực nghiệm.

---

# 9. Hướng phương pháp tiềm năng

Giả thuyết hứa hẹn hiện tại:

> Khả năng chuyển giao được cải thiện khi nhiễu tấn công vào "evidence" (bằng chứng) liên quan đến object mà ít đặc thù cho kiến trúc surrogate.

Một công thức tiềm năng trong tương lai:

\[
E_s(x)
=
E_{shared}(x)
+
E_{specific}(x)
\]

Tấn công mong muốn nên phá vỡ hoặc suy giảm:

\[
E_{shared}
\]

đồng thời tránh overfit vào:

\[
E_{specific}
\]

Khái niệm phương pháp tiềm năng:

**Architecture-Invariant Object Evidence Disruption** (Phá vỡ bằng chứng đối tượng bất biến theo kiến trúc)

Thành phần tiềm năng:

- object-conditioned feature importance
- saliency consistency
- transformation consistency
- multi-layer feature analysis
- feature disruption
- frequency-domain regularization

---

## 9.1 Method v0.1 (formalized, chốt 2026-09-20 — sau Thí nghiệm 3A + 3B)

**Thay thế phần lý thuyết chung chung ở §9 phía trên** (viết trước khi có bằng chứng thực nghiệm) bằng công thức cụ thể, KHỚP CHÍNH XÁC với code hiện tại (`attacks/backward_reg_attack.py`) — không viết công thức "đẹp hơn" implementation thật (quy tắc đã thống nhất với user).

### Ký hiệu

- \(x\): ảnh (đang craft adversarial).
- \(F_l = f_l(x)\): feature map thô của backbone tại stage \(l \in \{1,2,3,4\}\) — tức `model.backbone(x)[l-1]`, TRƯỚC neck/FPN.
- \(L\): loss của attack (cross-entropy tại đúng RoI của GT, `objective="cls"` — `_bbox_cls_loss`).
- \(g_l = \partial L / \partial F_l\): gradient chảy qua stage \(l\) trong lúc `torch.autograd.grad(L, x)` — chặn bằng `F_l.register_hook(...)`.
- \(s_l \in \{4,8,16,32\}\): stride của stage \(l\) (verify 2026-09-20, giống nhau ở cả 4 model — xem Thí nghiệm 2B).

### Khối 1 — Stage Selection

\[
\mathcal{S}^* = \{3, 4\}
\]

Chọn dựa trên bằng chứng backward-sensitivity đo được ở Thí nghiệm 2C (không phải heuristic/toàn mạng như TGR/PAS/GRA), và xác nhận causal ở Thí nghiệm 3A. Với \(l \notin \mathcal{S}^*\): \(\hat g_l = g_l\) (không đổi gì — code không đăng ký hook ở stage đó).

### Khối 2 — Backward Regularization (variance clipping)

Với mỗi \(l \in \mathcal{S}^*\), tính trên TOÀN BỘ tensor \(g_l\) (1 cặp số vô hướng \(\mu_l,\sigma_l\) cho cả tensor — không phải per-channel/per-pixel):

\[
\mu_l = \text{mean}(g_l), \quad \sigma_l = \text{std}(g_l)
\]

\[
\tilde g_l = \text{clip}\big(g_l;\ \mu_l - k\sigma_l,\ \mu_l + k\sigma_l\big)
\]

\(k = 3\) (hằng số cố định trong code hiện tại — chưa tune, đối tượng của **Ablation A**).

### Khối 3 — Object-Conditioned Weighting

Mask không gian tại stage \(l\), xây từ union các GT box (quy đổi tọa độ qua \(s_l\)), per-pixel \(p\):

\[
M_l(p) = \begin{cases} 1 & p \in \bigcup_i \text{box}_i / s_l \\ \beta & \text{ngược lại} \end{cases}
\]

\(\beta = 0.3\) (hằng số cố định trong code — `OBJECT_MASK_BG_WEIGHT`, chưa tune, đối tượng của **Ablation B**). Mask hiện tại là **hard box, không làm mượt biên** (đối tượng của **Ablation C** — mask shape).

### Công thức tổng (mode `"clip_weight"` — setting tốt nhất ở 3B)

\[
\hat g_l =
\begin{cases}
g_l & l \notin \mathcal{S}^* \\[4pt]
M_l \odot \tilde g_l + (1 - M_l) \odot g_l & l \in \mathcal{S}^*
\end{cases}
\]

**Đây là 1 convex blend per-pixel giữa gradient ĐÃ CLIP và gradient GỐC (chưa clip), điều chỉnh theo mask** — KHÔNG phải \(M_l \odot \tilde g_l\) (multiplicative thuần) và KHÔNG phải \((1+\lambda M_l)\odot \tilde g_l\) (residual form) như phác thảo ban đầu (đã sửa sau khi đối chiếu trực tiếp với code, 2026-09-20). Cụ thể ngoài vùng object (\(M_l(p)=\beta=0.3\)): \(\hat g_l(p) = 0.3\,\tilde g_l(p) + 0.7\,g_l(p)\) — vẫn còn 30% ảnh hưởng của clip, không phải gradient gốc thuần túy.

Setting `"weight"` (đã test ở 3B, KHÔNG có tác dụng đứng một mình) là 1 nhánh riêng, không đi qua clip:

\[
\hat g_l = M_l \odot g_l \quad (\text{không có bước clip})
\]

### Chuẩn bị cho Ablation B (object-weight strength) — cần refactor nhỏ

Để sweep "độ mạnh" của object-weighting với 1 tham số liên tục \(\lambda\) sao cho \(\lambda=0\) khớp CHÍNH XÁC lại `reg_s3s4` (yêu cầu của user), tổng quát hóa \(\beta\) thành hàm của \(\lambda\):

\[
\beta(\lambda) = \frac{1}{1+\lambda}
\]

Tại \(\lambda=0\): \(\beta=1 \Rightarrow M_l(p)=1 \ \forall p \Rightarrow \hat g_l = \tilde g_l\) — đúng bằng `reg_s3s4`, mọi pixel đều bị clip đều tay, không phân biệt object/background. \(\lambda\) tăng → \(\beta\) giảm → blend ngoài vùng object nghiêng dần về gradient gốc (chưa clip). Config đã chạy ở Thí nghiệm 3B (\(\beta=0.3\)) tương ứng \(\lambda = 1/\beta - 1 \approx 2.333\) — **tái sử dụng được luôn làm 1 điểm dữ liệu trong sweep Ablation B, không cần chạy lại**.

### 3 hằng số cần ablation (one-factor-at-a-time quanh config 3B, theo đúng thứ tự đã thống nhất)

| Ablation | Tham số | Grid đề xuất | Ghi chú |
|---|---|---|---|
| A | \(k\) (clip bound) | \(\{1, 2, 3, 4, \text{no-clip}\}\) | \(k=3\) là điểm hiện tại (3A/3B) |
| B | \(\lambda\) (object-weight strength) | \(\{0, 0.25, 0.5, 1, 2\}\) | \(\lambda=0\) = `reg_s3s4` (tái dùng); \(\lambda\approx 2.33\) đã có từ 3B |
| C | Mask shape | hard box (hiện tại) / soft-Gaussian / dilated / objectness-derived | làm sau cùng, sau khi đã chọn \(k^*,\lambda^*\) |

**Quy tắc chọn hyperparameter** (thống nhất với user, 2026-09-20): dựa trên **cross-family average ASR** và **TransferGap**, KHÔNG dựa riêng vào 1 target (ConvNeXt hoặc Swin):

\[
\text{CrossFamilyASR} = \frac{ASR_{ConvNeXt} + ASR_{Swin}}{2} \quad \uparrow, \qquad \text{TransferGap} \quad \downarrow
\]

Một setting tăng Swin nhưng giảm ConvNeXt (hoặc ngược lại) KHÔNG được gọi là "improvement" tổng quát — phải cải thiện cả 2 hoặc ít nhất không đánh đổi cái này lấy cái kia.

Đây chỉ là hướng nghiên cứu, chưa phải phương pháp cố định.

## 9.2 Mechanism (chốt 2026-09-21, sau Thí nghiệm 3F–3I — xem docs/progress_log.md entry cùng ngày)

Thay thế giả thuyết cơ chế ban đầu ("regularization giúp transfer bằng cách làm gradient surrogate giống gradient target hơn", ngầm định khi thiết kế 3F) bằng bằng chứng thực nghiệm trực tiếp. Chuỗi 4 thí nghiệm, thứ tự rẻ→sâu:

- **3F (cosine alignment với target) — bác bỏ**: đo cos(g_s, g_t) trước/sau regularization, n=300. Kết quả KHÔNG xác nhận alignment tăng ở ConvNeXt (Δ=−0.0039) hay Swin (Δ=+0.0002, ~0) dù ASR cross-family tăng thật (đã confirm ở 3A-3E). Method **không** hoạt động bằng cách tăng gradient alignment với target.
- **3G (gradient concentration/tail tại surrogate) — mạnh**: so RAW vs REG (Stage 3-4, k=3, λ=0.5) trên 6 thống kê toàn-tensor, n=300. 4/5 metric giảm mạnh và nhất quán dưới reg: `max_abs` giảm 8-27×, `kurtosis` giảm 6.8-19×, `top1%-energy` giảm ~40-50%, `L∞/L2` giảm ~4-14× (Wilcoxon paired p≈6e-51 mọi metric). `std` chỉ giảm ~45-54% — tín hiệu gradient không bị triệt tiêu, chỉ bớt bị vài giá trị cực trị "thống trị".
- **3H (iteration trajectory stability) — yếu nhưng nhất quán**: chạy attack thật 10-iteration, log gradient mỗi bước. 4/5 metric có p cực nhỏ và đúng chiều (cos giữa gradient liên tiếp tăng, sign-flip giảm, trôi dạt so với hướng ban đầu giảm), nhưng effect size nhỏ và `final_cos_drift` (so g1 với g10, bước cuối) KHÔNG có ý nghĩa — đây là hệ quả phụ (secondary consequence), không đủ làm cơ chế chính.
- **3I (linkage: ΔC per-ảnh ↔ tỷ lệ giành lại transfer) — mạnh và đặc thù cross-family**: với mỗi ảnh, ΔC = mức giảm `top1%-energy` (Stage 3-4, raw−reg). So ΔC giữa ảnh có "giành lại" object bị né tránh (baseline fail → reg success) vs ảnh không giành lại được nào: có ý nghĩa ở ConvNeXt (MWU p=1.32e-04, Spearman ρ=0.205) và Swin (p=1.51e-04, ρ=0.187), **KHÔNG có ý nghĩa** ở R101 same-family (control, p=0.199, ρ=0.046).

**Kết luận cơ chế (claim an toàn, dùng nguyên văn khi viết paper — không viết mạnh hơn)**:

> Suppressing extreme mid/deep backward-gradient concentration is strongly associated with improved transfer specifically to cross-family targets, while no corresponding association is observed for the same-family control.

3I là bằng chứng **linkage/correlational ở mức per-ảnh** giữa ΔC và gained-transfer-rate — không phải "chứng minh causal mediation" theo nghĩa chặt (chưa làm mediation analysis chính thức). Kết hợp với 3A (causal intervention ở mức Stage selection) và 3G (regularization thực sự giảm concentration), bằng chứng đủ mạnh để thay thế hoàn toàn giả thuyết "gradient alignment" (§7.A cũ, đã bác bỏ bởi 3F) làm câu chuyện cơ chế chính thức của method v0.1.

---

# 10. Mục tiêu về tính mới (Novelty)

Việc đánh giá cross-family bản thân nó **không đủ** để tạo ra tính mới.

Tính mới nên đến từ:

> Một cách mới để phát hiện hoặc khai thác các lỗ hổng bất biến theo kiến trúc, chỉ dùng một surrogate duy nhất.

Claim tiềm năng:

> Chúng tôi xác định một biểu diễn đối tượng bất biến theo kiến trúc, có thể đo lường được, dự đoán khả năng chuyển giao đối kháng xuyên họ, và thiết kế một tấn công single-surrogate phá vỡ trực tiếp biểu diễn này.

Claim này phải được hỗ trợ bằng thực nghiệm.

---

# 11. Mục tiêu SOTA

Không claim "SOTA adversarial attack" chung chung.

Nhắm vào claim hẹp và bảo vệ được:

> State-of-the-art cho khả năng chuyển giao xuyên feature-extractor, black-box, single-surrogate, cho object detection.

Để hỗ trợ điều này, so sánh dưới cùng threat model với các baseline mạnh như:

- các tấn công chuyển giao lặp (iterative) chuẩn
- các tấn công chuyển giao dựa trên feature
- các tấn công chuyển giao chuyên biệt cho object detection
- OSFD
- các phương pháp chuyển giao dựa trên augmentation gần đây
- các phương pháp liên quan khác tìm được qua literature search

Tất cả so sánh phải dùng:

- cùng dataset
- cùng perturbation budget
- cùng surrogate
- cùng target đánh giá
- ngân sách vòng lặp tương đương

---

# 12. Câu hỏi nghiên cứu

### RQ1

Khả năng chuyển giao đối kháng thay đổi thế nào khi kiến trúc feature-extractor giữa surrogate và target khác nhau?

**KHÓA (locked) — 2026-09-20**, dựa trên Thí nghiệm 1 + 1B (§6, §6.8):

> **Cross-family transfer gap là hiện tượng ổn định trong setting hiện tại, không phụ thuộc riêng vào attack, objective hay clean model strength.**

Cách diễn đạt claim (cố ý, không claim nhân quả tuyệt đối): **"strong evidence that backbone architecture contributes to the transfer gap"** — không phải "backbone architecture is the cause". Lý do dùng từ "contributes" thay vì "is the cause": vẫn còn ít nhất 1 confound chưa cô lập (training-recipe của ConvNeXt-Tiny, xem model_registry.md) và chưa test nhiều seed/subset — đủ để khóa RQ1 làm tiền đề cho Thí nghiệm 2, nhưng chưa đủ để claim quan hệ nhân quả tuyệt đối. Không sửa lại cách diễn đạt này trừ khi có bằng chứng mới đủ mạnh để claim chặt hơn hoặc yếu hơn.

**Cập nhật cùng ngày (sau khi khóa)**: phát hiện + fix 1 bug tọa độ RoI trong attack (xem §6.8 và docs/progress_log.md) khiến toàn bộ số liệu ASR/TransferGap dùng để khóa RQ1 ở trên là từ bản có bug (attack yếu hơn nhiều so với thiết kế). Đã re-run ở n=300 với bản đã fix — **claim khóa ở trên vẫn đứng vững, bằng chứng còn mạnh hơn** (TransferGap sau fix lớn hơn 3-5 lần số liệu cũ, xem progress_log.md). Không cần mở khóa lại RQ1, chỉ cần lưu ý số liệu cụ thể trích dẫn từ giờ nên lấy từ progress_log.md entry sau ngày fix, không lấy từ entry "Kết quả Thí nghiệm 1/1B" cũ (đã đánh dấu superseded).

### RQ2

Thuộc tính biểu diễn đo lường được nào giải thích tốt nhất khoảng cách chuyển giao xuyên họ?

Ứng viên (đã kiểm tra bằng thực nghiệm — Thí nghiệm 2A/2B/2B.1/2C, docs/progress_log.md):

- gradient similarity — **có liên hệ đúng chiều** (2A: cos_sim cao hơn ở nhóm evaded, p cực nhỏ, effect size trung bình-lớn; 2C: mean cos_sim theo target đúng thứ tự ASR từ Stage 3 trở đi)
- feature similarity (raw forward, đo bằng linear CKA) — **đã bác bỏ** làm cơ chế chính (2B: thứ tự CKA sai theo target; 2B.1: CKA_evaded < CKA_not_evaded ở deep stage, ngược hypothesis)
- saliency similarity, object evidence similarity, transformation consistency — chưa kiểm tra (không cần nữa, xem câu trả lời dưới)

**KHÓA (locked) — 2026-09-20**, dựa trên Thí nghiệm 2A + 2B + 2B.1 + 2C:

> **RQ2 — Answered**: Cross-family transferability is better explained by **backward sensitivity alignment** than by raw forward feature similarity. Mid-to-deep backbone stages (Stage 3–4) are where architecture-dependent backward divergence emerges and aligns with the observed transfer gap.

Không cần điều tra thêm saliency/object evidence/transformation consistency ở mức forward representation nữa — dữ liệu đã đủ rõ để chuyển trọng tâm sang RQ3 (thiết kế method dựa trên backward sensitivity).

**Lưu ý phạm vi (thêm 2026-09-21, sau Thí nghiệm 3F–3I — xem §9.2)**: RQ2 ở trên là 1 câu hỏi **diagnostic/correlational** về transfer gap GỐC (dùng gradient RAW, chưa can thiệp gì) — kết luận "backward sensitivity alignment giải thích transfer gap tốt hơn forward feature similarity" **vẫn đúng, không đổi**. Đây KHÁC với câu hỏi "method v0.1 (regularization) có hoạt động BẰNG CÁCH tăng chính alignment này không?" — câu hỏi thứ hai đó được kiểm tra riêng ở Thí nghiệm 3F và **bị bác bỏ** (regularization không làm tăng cos(g_s,g_t) một cách có ý nghĩa). Không nhầm lẫn 2 câu hỏi này: RQ2 nói về ĐO ĐẠC (measurement) giải thích gap tốt nhất, còn cơ chế thật của METHOD lại là 1 phát hiện riêng (giảm concentration, §9.2), không phải trực tiếp từ RQ2.

### RQ3

**Cập nhật 2026-09-20** — cụ thể hóa từ câu hỏi gốc, dựa trực tiếp trên RQ2 đã khóa (không còn là "object evidence" chung chung mà là "backward sensitivity tại Stage 3-4" cụ thể):

> **Can we design a single-surrogate attack that makes mid/deep backward signals less architecture-specific and thereby improves cross-family transferability?**

Câu hỏi gốc (tham khảo, đã thay thế bởi câu trên): "Một tấn công single-surrogate có thể khai thác 'object evidence' bất biến theo kiến trúc để giảm khoảng cách chuyển giao xuyên họ hay không?" — vẫn đúng tinh thần, nhưng "object evidence bất biến theo kiến trúc" giờ đã được cụ thể hóa thành "backward sensitivity ít architecture-specific hơn tại Stage 3-4", nhờ RQ2 đã trả lời.

**Cập nhật 2026-09-21 — trả lời có điều kiện (không khóa hoàn toàn)**, dựa trên Thí nghiệm 3A + 3F–3I (chi tiết §9.2):

> **RQ3 — Partially answered**: method v0.1 (Stage 3-4 backward-variance clipping + object-conditioned weighting) làm mid/deep backward signal của surrogate ÍT bị vài giá trị cực trị "thống trị" hơn (3G, mạnh), KHÔNG làm signal đó "giống target hơn" theo nghĩa cosine alignment (3F, bác bỏ trực tiếp giả thuyết ban đầu của câu hỏi RQ3). Mức độ giảm concentration đó (ΔC per-ảnh) tương quan có ý nghĩa với việc giành lại transfer, đặc thù ở target cross-family, không ở same-family control (3I, mạnh). Trajectory optimization ổn định hơn (3H) là hệ quả phụ, effect nhỏ.

Nói cách khác: câu trả lời cho RQ3 là **có, nhưng không theo cơ chế "architecture-invariant" ban đầu hình dung** (không phải làm signal giống nhau giữa các kiến trúc hơn) — mà là làm signal của chính surrogate bớt cực đoan/tập trung hơn, và mức giảm đó liên hệ trực tiếp với transfer gain ở đúng nhóm target mà nghiên cứu nhắm tới (cross-family). Chưa khóa hoàn toàn vì 3I là bằng chứng correlational (chưa phải causal mediation chính thức) và mask/control ablation (kiểm tra vai trò thật của "object-conditioned", phân biệt với spatial weighting bất kỳ) chưa chạy — xem docs/progress_log.md để biết trạng thái mới nhất.

---

# 13. Nhiệm vụ trước mắt cho Agent

Chỉ bắt đầu với **Thí nghiệm 1**.

Agent cần:

1. Kiểm tra repo hiện tại và model registry
2. Xác định surrogate và target hiện có
3. Xác minh model nào thực sự dùng:
   - ResNet
   - ConvNeXt
   - Swin / backbone Transformer khác
4. Đề xuất bộ model kiểm soát nhỏ nhất cho Thí nghiệm 1
5. Tái sử dụng cài đặt tấn công hiện có nếu được
6. Cài đặt đánh giá có thể tái lập (reproducible)
7. Báo cáo:
   - clean mAP
   - adversarial mAP
   - mức giảm mAP
   - ASR
   - chuyển giao same-family vs cross-family
8. Lưu toàn bộ config và seed của thí nghiệm
9. Chưa cài đặt tấn công mới
10. Tóm tắt liệu có quan sát được khoảng cách chuyển giao xuyên họ có ý nghĩa hay không

Milestone đầu tiên **không phải** là SOTA.

Milestone đầu tiên là:

> **Xác lập xem "khoảng cách chuyển giao xuyên feature-extractor" có thật, đo lường được, và tái lập được dưới thiết lập single-surrogate có kiểm soát hay không.**

**Trạng thái (cập nhật 2026-09-20): milestone đầu tiên đã đạt.** Thí nghiệm 1 (§6) xác lập transfer gap; Thí nghiệm 1B (§6.8) xác nhận gap này đứng vững qua 4 confound-check (model-strength, attack choice, attack objective, augmentation-based attack). Theo quy tắc §6.7, bước tiếp theo là **Thí nghiệm 2 (§7)** — tìm cơ chế giải thích transfer gap.
