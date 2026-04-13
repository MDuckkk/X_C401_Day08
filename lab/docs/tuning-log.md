# Nhóm có research 2 phương pháp chunking Recursive và Semantic 

================================================================
# Recursive
# Tuning Log — RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 13/4/2026
**Config:**
```
retrieval_mode = "dense"
chunk_size = 320 tokens
overlap = 60 tokens
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = gpt-4o-mini
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 4.90/5 |
| Relevance | 5.00/5 |
| Context Recall | 5.00/5 |
| Completeness | 3.90/5 |

**Câu hỏi yếu nhất (điểm thấp):**
> q07 (Access Control) - Completeness = 2/5; câu trả lời xác định đúng tên tài liệu nhưng không liệt kê đủ các điều kiện/phạm vi trong policy.
> q03 (Access Control) - Completeness = 3/5; thông tin đúng nhưng chưa đầy đủ các điều kiện cụ thể.
> q08 (HR Policy) - Completeness = 3/5; câu trả lời phản ánh đúng thông tin nhưng thiếu chi tiết về các trường hợp ngoại lệ.


**Giả thuyết nguyên nhân (Error Tree):**
- [x] Indexing: Chunking cắt giữa điều khoản → thiếu điều kiện/phạm vi đầy đủ (q03, q07)
- [ ] Indexing: Metadata thiếu effective_date
- [ ] Retrieval: Dense bỏ lỡ exact keyword / alias
- [ ] Retrieval: Top-k quá ít → thiếu evidence
- [x] Generation: Prompt không đủ grounding → model không liệt kê hết chi tiết (q07, q08)
- [ ] Generation: Context quá dài → lost in the middle

---

## Variant 1 (Sprint 3)

**Ngày:** 13/4/2026
**Biến thay đổi:** Retrieval mode (dense -> hybrid)  
**Lý do chọn biến này:**
> Baseline cho thấy Completeness là điểm yếu chính (3.90/5), tập trung ở các câu hỏi Access Control (q03, q07) và HR Policy (q08) — nơi câu trả lời đúng nhưng thiếu chi tiết điều kiện/phạm vi.
> Giả thuyết: chunking recursive có thể cắt giữa điều khoản, khiến evidence bị phân mảnh; hybrid retrieval kỳ vọng lấy thêm chunk liên quan qua lexical match để bổ sung context còn thiếu.
> Corpus có cả ngôn ngữ tự nhiên (policy/SLA) và nhiều term định danh (approval matrix, nhãn SLA), phù hợp với chiến lược hybrid.

**Config thay đổi:**
```
retrieval_mode = "hybrid"  
# Các tham số còn lại giữ nguyên như baseline
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 4.90/5 | 4.60/5 | -0.30 |
| Answer Relevance | 5.00/5 | 4.90/5 | -0.10 |
| Context Recall | 5.00/5 | 5.00/5 | +0.00 |
| Completeness | 3.90/5 | 4.00/5 | +0.10 |

**Nhận xét:**
> Cải thiện nhẹ ở Completeness (3.90 -> 4.00), cụ thể q01 tăng từ 3 -> 5, cho thấy hybrid giúp lấy thêm ngữ cảnh đầy đủ hơn ở câu hỏi SLA.
> Tuy nhiên Faithfulness giảm (4.90 -> 4.60) do q03 tụt mạnh (Faithfulness: 4 -> 1) — câu trả lời thêm thông tin không có trong context (hallucination), dù recall vẫn cao.
> Relevance giảm nhẹ (5.00 -> 4.90) do q04 giảm từ 5 -> 4.
> Các câu còn lại (q02, q05, q06, q08, q09, q10) không đổi hoặc cải thiện nhỏ.

**Kết luận:**
> Variant 1 **chưa tốt hơn baseline** nếu xét tổng thể.
> Bằng chứng: Faithfulness giảm (4.90 -> 4.60) và Relevance giảm nhẹ (5.00 -> 4.90), trong khi Completeness chỉ tăng +0.10. Dù có cải thiện cục bộ (q01), việc tụt Faithfulness mạnh ở q03 là rủi ro lớn hơn lợi ích thu được.

---

## Variant 2 (nếu có thời gian)

**Biến thay đổi:** ___________  
**Config:**
```
# TODO
```

**Scorecard Variant 2:**
| Metric | Baseline | Variant 1 | Variant 2 | Best |
|--------|----------|-----------|-----------|------|
| Faithfulness | ? | ? | ? | ? |
| Answer Relevance | ? | ? | ? | ? |
| Context Recall | ? | ? | ? | ? |
| Completeness | ? | ? | ? | ? |

---

## Tóm tắt học được

> TODO (Sprint 4): Điền sau khi hoàn thành evaluation.

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > Lỗi phổ biến nhất là **generation không liệt kê đủ chi tiết từ context**: hệ thống retrieve đúng (Recall 5.00/5) nhưng model không tổng hợp hết các điều kiện/phạm vi trong policy, dẫn đến Completeness thấp (q03, q07, q08).

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > **retrieval_mode (dense ↔ hybrid)** có tác động rõ nhất: cải thiện Completeness nhẹ (+0.10) nhưng gây tụt Faithfulness (-0.30) do hallucination ở q03. Trade-off này cho thấy bottleneck thực sự nằm ở generation/grounding hơn là retrieval.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > Bật **rerank** (giữ hybrid) và tăng độ chặt grounding trong prompt (ép model chỉ dùng evidence có trong context): ưu tiên khắc phục hallucination ở q03 và tăng Completeness cho q07/q08 mà không làm giảm Faithfulness.

================================================================

# Semantic
# Tuning Log — RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 13/4/2026
**Config:**
```
retrieval_mode = "dense"
chunking_mode = "semantic"
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = gpt-4o-mini
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 3.70/5 |
| Relevance | 3.40/5 |
| Context Recall | 5.00/5 |
| Completeness | 2.10/5 |

**Câu hỏi yếu nhất (điểm thấp):**
> Q07 (Error Code) - Faithfulness = 1/5, Relevance = 1/5, Completeness = 1/5, Recall = None; không có evidence phù hợp cho truy vấn mã lỗi nên câu trả lời không hữu ích.
> Q09 (Security) - Faithfulness = 1/5, Relevance = 1/5, Completeness = 1/5 (dù Recall = 5); retrieve được context nhưng trả lời vẫn nói thiếu thông tin, cho thấy grounding/chọn evidence chưa ổn định.
> Q05 (Network) và Q10 (Printer) - Relevance/Completeness rất thấp; câu trả lời thiên về hướng dẫn chung (mở ticket/liên hệ IT) hơn là trả lời đúng trọng tâm câu hỏi.

**Giả thuyết nguyên nhân (Error Tree):**
- [ ] Indexing: Semantic chunk tách theo ngữ nghĩa nhưng có thể làm rơi chi tiết điều kiện cụ thể
- [ ] Indexing: Metadata chưa hỗ trợ đủ cho các câu hỏi thao tác (network/printer)
- [x] Retrieval: Dense không bắt tốt query dạng mã lỗi/keyword đặc thù
- [x] Retrieval: Candidate context có nhưng chưa được chọn đúng phần để trả lời
- [x] Generation: Prompt grounding chưa ép model bám sát evidence đủ chặt
- [ ] Generation: Context ordering chưa tối ưu cho câu hỏi đa điều kiện

---

## Variant 1 (Sprint 3)

**Ngày:** 13/4/2026
**Biến thay đổi:** Retrieval mode (dense -> hybrid)  
**Lý do chọn biến này:**
> Baseline có dấu hiệu yếu ở nhóm câu hỏi keyword/mã lỗi (Q07) và các câu cần bám sát chi tiết nghiệp vụ (Q05, Q09, Q10), nên nhóm thử hybrid để tăng khả năng bắt exact term kết hợp semantic.
> Với corpus có nhiều câu hỏi vận hành IT (Error Code, Security, Network, Printer), hybrid kỳ vọng cải thiện Relevance/Completeness mà vẫn giữ Recall cao.
> Đây là thay đổi ưu tiên vì bám sát giả thuyết lỗi retrieval trong baseline.

**Config thay đổi:**
```
retrieval_mode = "hybrid"
chunk_size = 500 tokens
overlap = 100 tokens
# Các tham số còn lại giữ nguyên như baseline
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 3.70/5 | 3.70/5 | +0.00 |
| Answer Relevance | 3.40/5 | 3.40/5 | +0.00 |
| Context Recall | 5.00/5 | 5.00/5 | +0.00 |
| Completeness | 2.10/5 | 2.10/5 | +0.00 |

**Nhận xét:**
> Variant 1 **không tạo khác biệt đo được** so với baseline: toàn bộ metric trung bình giữ nguyên.
> Các câu yếu vẫn giữ nguyên pattern (Q07, Q09, Q05, Q10), nên bottleneck hiện tại nhiều khả năng nằm ở grounding/chọn evidence hoặc cấu hình variant chưa thực sự tạo khác biệt retrieval.
> Kết quả này hữu ích vì loại trừ được giả thuyết “chỉ cần đổi dense -> hybrid là đủ”.

**Kết luận:**
> Variant 1 **không tốt hơn baseline** và cũng không kém hơn theo scorecard hiện tại.
> Bằng chứng: Faithfulness/Relevance/Recall/Completeness đều có delta = 0.00; các câu lỗi chính không đổi.

---

## Variant 2 (nếu có thời gian)

**Biến thay đổi:** ___________  
**Config:**
```
# TODO
```

**Scorecard Variant 2:**
| Metric | Baseline | Variant 1 | Variant 2 | Best |
|--------|----------|-----------|-----------|------|
| Faithfulness | ? | ? | ? | ? |
| Answer Relevance | ? | ? | ? | ? |
| Context Recall | ? | ? | ? | ? |
| Completeness | ? | ? | ? | ? |

---

## Tóm tắt học được

> TODO (Sprint 4): Điền sau khi hoàn thành evaluation.

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > Lỗi phổ biến nhất là **answer grounding chưa bám đúng evidence**, dẫn đến Relevance/Completeness thấp dù Recall cao.

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > Với nhánh Semantic hiện tại, biến `retrieval_mode` (dense -> hybrid) **chưa cho tác động đo được**; cần thử biến khác như rerank hoặc prompt grounding.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > Bật **rerank** và ép citation theo từng mệnh đề trả lời; sau đó tăng `top_k_search` có kiểm soát để xem có cải thiện Q05/Q09/Q10 hay không.
