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

Dùng một surrogate cố định:

- Faster R-CNN + ResNet-50

---

## 6.2 Các mô hình Target

Ưu tiên các target giữ nguyên kiến trúc detector, chỉ thay đổi feature extractor.

So sánh có kiểm soát được đề xuất:

1. Faster R-CNN + ResNet-101
   - Cùng họ backbone

2. Faster R-CNN + ConvNeXt
   - Họ backbone CNN khác

3. Faster R-CNN + Swin Transformer
   - CNN → Transformer

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

Đây chỉ là hướng nghiên cứu, chưa phải phương pháp cố định.

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

### RQ2

Thuộc tính biểu diễn đo lường được nào giải thích tốt nhất khoảng cách chuyển giao xuyên họ?

Ứng viên:

- gradient similarity
- feature similarity
- saliency similarity
- object evidence similarity
- transformation consistency

### RQ3

Một tấn công single-surrogate có thể khai thác "object evidence" bất biến theo kiến trúc để giảm khoảng cách chuyển giao xuyên họ hay không?

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
