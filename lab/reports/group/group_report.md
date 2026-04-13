# Group Report — Quyết định kỹ thuật cấp nhóm (Chốt chọn Recursive)

## 1. Bối cảnh và mục tiêu

Nhóm triển khai RAG pipeline cho bài toán CS + IT Helpdesk, với mục tiêu kỹ thuật ưu tiên:
1. Trả lời bám sát tài liệu nội bộ, có thể truy vết nguồn.
2. Hạn chế hallucination ở câu thiếu dữ liệu.
3. Tối ưu chất lượng tổng thể theo scorecard (Faithfulness, Relevance, Recall, Completeness).

Sau khi chạy song song hai hướng chunking (**Recursive** và **Semantic**) và ghi nhận trong `docs/tuning-log.md`, nhóm chốt lựa chọn vận hành là **Recursive**.

---

## 2. Các quyết định kỹ thuật chính

### 2.1. Giữ kiến trúc pipeline 4 tầng

Nhóm thống nhất kiến trúc: **Indexing -> Retrieval -> Grounded Generation -> Evaluation**.

- `index.py`: preprocess/chunk/embed/index vào ChromaDB.
- `rag_answer.py`: retrieve (dense/hybrid), optional rerank, build grounded prompt, gọi LLM.
- `eval.py`: chấm scorecard và so sánh baseline/variant.

Quyết định này giúp nhóm debug theo failure mode rõ ràng (indexing vs retrieval vs generation), thay vì chỉnh đồng thời nhiều nơi.

### 2.2. Chọn chunking Recursive làm cấu hình chính

Cấu hình được chốt cho hướng Recursive baseline:
- `chunk_size = 320`
- `overlap = 60`
- heading-based split + fallback paragraph/sentence
- metadata đầy đủ để trace nguồn

Lý do: cấu hình này cho độ ổn định cao hơn trong fact lookup và giữ cân bằng tốt giữa độ chi tiết và nhiễu context.

### 2.3. Áp dụng A/B rule nghiêm ngặt

Ở mỗi nhánh tuning, nhóm chỉ đổi **một biến** trong Variant 1 để đảm bảo khả năng quy kết nguyên nhân:
- Recursive: đổi `retrieval_mode` từ dense sang hybrid.
- Semantic: đổi `retrieval_mode` + cấu hình chunk của variant đã ghi log, nhưng kết quả metric vẫn không cải thiện.

### 2.4. Ưu tiên guardrail chống hallucination

Nhóm giữ nguyên nguyên tắc grounded answer:
- chỉ trả lời theo context retrieve,
- thiếu dữ liệu thì abstain,
- ưu tiên citation/sources.

Đây là quyết định xuyên suốt vì rubric phạt nặng hallucination.

---

## 3. So sánh kết quả Recursive vs Semantic (theo tuning-log)

### 3.1. Recursive (tốt nhất hiện tại: Baseline dense)
- Faithfulness: **4.60/5**
- Relevance: **4.30/5**
- Context Recall: **5.00/5**
- Completeness: **3.10/5**

Variant hybrid của Recursive có trade-off:
- Faithfulness giảm còn 3.80
- Completeness tăng lên 3.30
- Kết luận nội bộ: hybrid chưa vượt baseline do tụt faithfulness.

### 3.2. Semantic (baseline và variant hiện tại)
- Faithfulness: **3.70/5**
- Relevance: **3.40/5**
- Context Recall: **5.00/5**
- Completeness: **2.10/5**

Variant 1 Semantic không tạo delta (0.00 ở cả 4 metric), nên chưa chứng minh được lợi ích so với baseline Semantic.

### 3.3. Kết luận so sánh

Với cùng recall 5.00, **Recursive vượt Semantic rõ ràng ở Faithfulness, Relevance và Completeness**.  
Vì vậy nhóm **chọn Recursive làm hướng chính** cho demo, grading run và các vòng tối ưu tiếp theo.

---

## 4. Quyết định cuối cùng của nhóm

Nhóm chốt cấu hình vận hành:

```txt
chunking: Recursive
retrieval_mode: dense (mốc ổn định tốt nhất hiện tại)
top_k_search: 10
top_k_select: 3
use_rerank: False (tạm thời)
```

Lý do chốt dense trên Recursive ở thời điểm hiện tại:
1. Đây là cấu hình có điểm tổng thể tốt nhất trong các kết quả đã có.
2. Giữ được faithfulness cao, phù hợp mục tiêu “đúng nguồn trước, tối ưu độ đầy đủ sau”.
3. Giảm rủi ro khi bước vào grading/questions ẩn.

---

## 5. Kế hoạch cải tiến tiếp theo (trên nền Recursive)

Để tăng chất lượng mà không phá độ ổn định hiện tại, nhóm sẽ thử theo thứ tự:
1. Giữ Recursive, bật rerank để xử lý lỗi grounding/chọn evidence (đặc biệt q03, q10).
2. Nếu cần, tăng `top_k_search` có kiểm soát rồi giữ `top_k_select` nhỏ để giảm nhiễu.
3. Siết prompt grounding theo evidence/citation để giảm câu trả lời “đúng chủ đề nhưng thiếu căn cứ”.

Nguyên tắc thực hiện vẫn giữ: **mỗi lần chỉ đổi một biến** và đánh giá lại bằng scorecard trước khi chốt.
