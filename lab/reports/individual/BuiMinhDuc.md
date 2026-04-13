# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Bùi Minh Đức  
**Vai trò trong nhóm:** Retrieval Owner  
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Tôi thiết kế rag với recursive chunking để so sánh với semantic của nhóm.
Phần indexing, tôi thiết kế toàn bộ logic chunking trong `index.py`: implement `preprocess_document()` để extract metadata từ header file (source, department, effective_date, access), `chunk_document()` theo chiến lược recursive — split theo heading `=== ... ===` trước, rồi paragraph, rồi câu, cuối cùng mới fallback theo ký tự. Tôi cũng viết `_estimate_chunk_settings()` để tự động ước lượng chunk size/overlap từ dữ liệu thực tế thay vì hardcode.
Phần retrieval, tôi implement `retrieve_dense()`, `retrieve_sparse()` dùng BM25, và `retrieve_hybrid()` dùng Reciprocal Rank Fusion (RRF). Phần generation, tôi implement `build_grounded_prompt()` và `call_llm()` với fallback logic. Phần evaluation, tôi implement toàn bộ 4 hàm chấm điểm LLM-as-Judge trong `eval.py` (faithfulness, relevance, context recall, completeness), chạy A/B comparison và tạo grading log.

Kết quả A/B được ghi lại trong `docs/tuning-log.md` và `results/`.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, tôi hiểu rõ hơn về **recursive chunking** và tại sao nó phù hợp hơn fixed-size chunking cho tài liệu có cấu trúc.

Tôi cũng hiểu rõ hơn về **overlap**: không phải cứ overlap nhiều là tốt. Overlap quá lớn làm tăng nhiễu trong context, còn quá nhỏ thì mất ngữ cảnh giữa các chunk. Hàm `_build_overlap_text()` tôi viết ưu tiên lấy theo paragraph thay vì cắt cứng theo ký tự, giúp overlap có nghĩa hơn.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều tôi ngạc nhiên nhất là hybrid retrieval không phải lúc nào cũng tốt hơn dense.

Ban đầu tôi kỳ vọng hybrid sẽ cải thiện toàn diện vì nó kết hợp cả semantic và keyword. Nhưng kết quả scorecard cho thấy ngược lại: variant hybrid làm Faithfulness giảm từ 4.90 xuống 4.60, cụ thể câu q03 bị hallucination nghiêm trọng (Faithfulness = 1/5). Pipeline hybrid trả lời "Line Manager phải phê duyệt để cấp quyền Level 3" — bỏ sót IT Admin và IT Security.

Sau khi debug, tôi xác định lỗi nằm ở retrieval: BM25 boost quá mạnh cho chunk chứa từ "Level 3" và "Line Manager" nhưng không chứa đủ thông tin về IT Admin. RRF fusion đã đẩy chunk đó lên top, khiến LLM chỉ thấy thông tin một phần. Đây là bài học thực tế về việc sparse retrieval có thể gây bias khi corpus có nhiều điều khoản liên quan đến cùng một keyword.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** q03 — "Ai phải phê duyệt để cấp quyền Level 3?"  
**Expected answer:** Level 3 (Elevated Access) cần phê duyệt từ Line Manager, IT Admin, và IT Security.

**Phân tích:**

Baseline (dense) trả lời: "Để cấp quyền Level 3, cần có sự phê duyệt của Line Manager và IT Security review [1]." — Faithfulness 4/5, Completeness 3/5. Pipeline đúng về nguồn nhưng bỏ sót IT Admin.

Variant (hybrid) trả lời: "Line Manager phải phê duyệt để cấp quyền Level 3 [1]." — Faithfulness 1/5, Completeness 2/5. Đây là hallucination vì câu trả lời không sai hoàn toàn nhưng bịa ra rằng chỉ cần Line Manager, trong khi tài liệu yêu cầu ba bên phê duyệt.

Lỗi nằm ở **retrieval**: thông tin về Level 3 access nằm rải rác trong nhiều đoạn của `access_control_sop.txt`. Chunk được retrieve chỉ chứa phần đề cập Line Manager, không chứa đoạn liệt kê đủ ba bên phê duyệt. Đây là failure mode của **indexing** — chunk bị cắt tại ranh giới không tốt, tách rời các bước trong cùng một quy trình.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Tôi sẽ thử hai cải tiến cụ thể:

1. Bật rerank với cross-encoder trên nền Recursive dense. Scorecard cho thấy q03 và q07 đều bị Completeness thấp (3/5 và 2/5) do pipeline chỉ lấy được một phần thông tin. Rerank có thể ưu tiên chunk chứa đủ các điều kiện thay vì chỉ chunk có keyword khớp nhất.

2. Tăng `top_k_select` từ 3 lên 5 cho các câu hỏi multi-condition (q03, q05, q06). Evidence từ scorecard cho thấy các câu này đều bị Completeness 3-4/5 — dấu hiệu pipeline đang thiếu context, không phải thiếu retrieval.
