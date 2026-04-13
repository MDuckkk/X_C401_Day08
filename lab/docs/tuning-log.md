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
| Faithfulness | 4.60/5 |
| Relevance | 4.30/5 |
| Context Recall | 5.00/5 |
| Completeness | 3.10/5 |

**Câu hỏi yếu nhất (điểm thấp):**
> q09 (Insufficient Context) - Faithfulness = 1/5, Relevance = 1/5, Completeness = 1/5, Recall = None; model trả lời "Tôi không biết", cho thấy không truy hồi được evidence phù hợp để trả lời.
> q10 (Refund) - Relevance = 2/5, Completeness = 2/5 (dù Faithfulness = 5/5); câu trả lời đúng ý chính nhưng thiếu chiều sâu/chi tiết cần thiết.
> q03 (Access Control) - Completeness = 2/5; thông tin đúng nhưng chưa đầy đủ các điều kiện/phạm vi trong policy.


**Giả thuyết nguyên nhân (Error Tree):**
- [ ] Indexing: Chunking cắt giữa điều khoản
- [ ] Indexing: Metadata thiếu effective_date
- [x] Retrieval: Dense bỏ lỡ exact keyword / alias
- [x] Retrieval: Top-k quá ít → thiếu evidence
- [x] Generation: Prompt không đủ grounding
- [ ] Generation: Context quá dài → lost in the middle

---

## Variant 1 (Sprint 3)

**Ngày:** 13/4/2026
**Biến thay đổi:** Retrieval mode (dense -> hybrid)  
**Lý do chọn biến này:**
> Baseline cho thấy lỗi retrieval là nút thắt chính (`Dense bỏ lỡ exact keyword / alias` và `Top-k quá ít -> thiếu evidence` trong Error Tree).
> Điểm yếu tập trung ở q09 (Insufficient Context: Faithfulness/Relevance/Completeness đều 1) và q10 (Relevance 2, Completeness 2), nên ưu tiên thử hybrid để kết hợp semantic match + lexical match cho alias/keyword đặc thù.
> Corpus có cả ngôn ngữ tự nhiên (policy/SLA) và nhiều term định danh (approval matrix, mã lỗi, nhãn SLA), phù hợp với chiến lược hybrid.

**Config thay đổi:**
```
retrieval_mode = "hybrid"   # hoặc biến khác
# Các tham số còn lại giữ nguyên như baseline
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 4.60/5 | 3.80/5 | -0.80 |
| Answer Relevance | 4.30/5 | 4.30/5 | +0.00 |
| Context Recall | 5.00/5 | 5.00/5 | +0.00 |
| Completeness | 3.10/5 | 3.30/5 | +0.20 |

**Nhận xét:**
> Cải thiện rõ nhất ở q01 (Completeness: 3 -> 5), cho thấy hybrid giúp lấy thêm ngữ cảnh đầy đủ hơn ở câu hỏi SLA.
> Một số câu không đổi (q02, q04, q05, q06, q07, q08, q09) nên tác động của hybrid không đồng đều.
> Kém hơn ở q03 và q10 (Faithfulness: 5 -> 1), do câu trả lời thêm/diễn giải sai chi tiết dù recall vẫn cao; đây là dấu hiệu context retrieval đủ nhưng grounding/chọn evidence chưa ổn định.

**Kết luận:**
> Variant 1 **chưa tốt hơn baseline** nếu xét tổng thể.
> Bằng chứng: Faithfulness giảm mạnh (4.60 -> 3.80), trong khi Relevance và Recall giữ nguyên, Completeness chỉ tăng nhẹ (3.10 -> 3.30). Dù có cải thiện cục bộ (q01), việc tụt Faithfulness ở q03 và q10 là rủi ro lớn hơn lợi ích thu được.

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
   > Lỗi phổ biến nhất là **retrieval/generation mismatch**: hệ thống có lấy được context (Recall cao) nhưng answer vẫn thiếu ý hoặc sai chi tiết khi grounding chưa ổn định (thể hiện ở Completeness thấp và tụt Faithfulness ở một số câu như q03, q10).

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > **retrieval_mode (dense ↔ hybrid)** là biến có tác động lớn nhất trong vòng này: chỉ đổi 1 biến nhưng làm thay đổi rõ trade-off giữa Faithfulness và Completeness.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > Bật **rerank** (giữ hybrid) và chạy lại A/B: ưu tiên tăng Faithfulness cho q03/q10 mà không làm giảm Completeness; nếu chưa đủ, thử tăng `top_k_search` lên 12-15 rồi giữ `top_k_select=3` để tránh nhiễu context.

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
