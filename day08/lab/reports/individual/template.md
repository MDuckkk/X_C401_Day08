# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Vũ Văn Huân
**Vai trò trong nhóm:** Eval Owner
**Ngày nộp:** 13/04/2026
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này, tôi đảm nhận vai trò Retrieval Owner và tập trung chủ yếu vào Sprint 2 và Sprint 3 của pipeline RAG.

Ở Sprint 2, tôi đã implement baseline Dense Retrieval sử dụng ChromaDB. Cụ thể, tôi xây dựng hàm retrieve_dense() để embed query và tìm kiếm các chunk liên quan dựa trên cosine similarity. Đồng thời, tôi kết hợp với grounded prompt để đảm bảo output có citation và hạn chế hallucination.

Sang Sprint 3, tôi phát triển thêm Hybrid Retrieval bằng cách kết hợp Dense và BM25 (sparse search). Tôi implement BM25 với rank_bm25 và xây dựng cơ chế merge kết quả bằng Reciprocal Rank Fusion (RRF). Ngoài ra, tôi cũng xử lý caching BM25 để tối ưu hiệu năng.

Công việc của tôi kết nối trực tiếp với Eval Owner, vì kết quả retrieval là đầu vào quan trọng để đánh giá hiệu năng toàn hệ thống thông qua scorecard.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, tôi hiểu sâu hơn về Hybrid Retrieval và vai trò của từng thành phần trong pipeline RAG.

Trước đây, tôi nghĩ chỉ cần dùng embedding là đủ, nhưng qua thực hành tôi nhận ra Dense Retrieval rất dễ bỏ sót các keyword quan trọng như mã lỗi (ERR-403) hoặc thuật ngữ đặc thù. Ngược lại, BM25 lại rất mạnh trong việc match chính xác từ khóa nhưng không hiểu ngữ nghĩa.

Hybrid Retrieval giúp kết hợp điểm mạnh của cả hai: Dense xử lý ngữ nghĩa, còn Sparse đảm bảo không miss keyword quan trọng. Tuy nhiên, việc kết hợp không đơn giản, vì mỗi phương pháp có thang điểm khác nhau, dẫn đến khó khăn khi merge và threshold.

Ngoài ra, tôi cũng hiểu rõ hơn rằng Retrieval không chỉ là “lấy tài liệu”, mà là bước quyết định trực tiếp đến chất lượng của toàn bộ hệ thống RAG.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều khiến tôi bất ngờ nhất là việc Hybrid Retrieval cho kết quả kém hơn Dense trong một số trường hợp, trái với kỳ vọng ban đầu.

Ban đầu, tôi nghĩ rằng việc kết hợp Dense + BM25 chắc chắn sẽ cải thiện chất lượng vì “nhiều thông tin hơn”. Tuy nhiên, khi chạy scorecard, tôi thấy điểm Completeness của variant lại giảm nhẹ.

Sau khi debug, tôi phát hiện vấn đề nằm ở việc chuẩn hóa score. Dense sử dụng cosine similarity (0 → 1), trong khi BM25 và RRF lại có thang điểm rất khác (thường rất nhỏ). Khi đưa vào chung một pipeline với threshold cố định, hệ thống dễ đánh giá sai rằng “không đủ dữ liệu”.

Ngoài ra, việc implement RRF cũng có bug ban đầu (chưa khởi tạo rrf_scores), gây sai lệch kết quả. Đây là lỗi nhỏ nhưng ảnh hưởng lớn đến toàn bộ pipeline.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** q07 - "Approval Matrix để cấp quyền hệ thống là tài liệu nào?"

**Phân tích:**

Baseline (Dense): Trả lời đúng một phần. Điểm: F=5, R=5, Rc=5, C=3.  
Variant (Hybrid): Trả lời "Không đủ dữ liệu". Điểm: F=3, R=5, Rc=5, C=1.

Dựa vào điểm Context Recall = 5/5, có thể thấy Retriever đã lấy được đúng tài liệu liên quan trong cả hai trường hợp. Tuy nhiên, sự khác biệt nằm ở bước xử lý sau retrieval.

Ở baseline, Dense Retrieval trả về các chunk có score cao và ổn định, nên vượt qua threshold và được đưa vào LLM để sinh câu trả lời. Tuy nhiên, do context chưa đầy đủ nên Completeness chỉ đạt 3/5.

Ở variant Hybrid, do sử dụng RRF với công thức 1/(60+rank), score của các chunk bị giảm xuống rất thấp. Khi đưa vào logic kiểm tra threshold (MIN_SCORE_THRESHOLD), hệ thống hiểu nhầm rằng không có đủ dữ liệu và chặn luôn bước generation.

Như vậy, lỗi không nằm ở retrieval mà nằm ở thiết kế scoring và threshold trong code. Hybrid không thất bại về mặt ý tưởng, mà thất bại do cách integrate vào pipeline chưa đúng.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Chuẩn hóa lại score giữa Dense và Hybrid: Tôi sẽ áp dụng normalization (ví dụ Min-Max Scaling) để đưa tất cả score về cùng một thang đo trước khi so sánh với threshold, vì kết quả eval cho thấy Hybrid đang bị đánh giá sai do khác hệ quy chiếu.

Thêm rerank bằng cross-encoder: Tôi sẽ thử thêm bước rerank sau hybrid retrieval để lọc ra top chunk chất lượng cao hơn, vì hiện tại top_k_select=3 có thể vẫn chứa noise và ảnh hưởng đến chất lượng câu trả lời.

---
