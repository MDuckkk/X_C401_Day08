# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Lê Nguyễn Chí Bảo  
**Vai trò trong nhóm:** Documentation Owner  
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này, tôi tập trung vai trò Documentation Owner ở Sprint 3 và 4, với mục tiêu biến kết quả kỹ thuật của nhóm thành quyết định rõ ràng, có căn cứ. Tôi tổng hợp log tuning từ hai hướng chunking (Recursive và Semantic), đối chiếu scorecard baseline/variant theo từng metric (Faithfulness, Relevance, Recall, Completeness), rồi chuẩn hóa thành bảng so sánh dùng chung cho cả nhóm. Từ dữ liệu đó, tôi đề xuất kết luận chốt: giữ pipeline 4 tầng (Indexing → Retrieval → Grounded Generation → Evaluation), chọn Recursive làm hướng chính vì ổn định hơn ở chất lượng câu trả lời. Ngoài ra, tôi chịu trách nhiệm hoàn thiện các tài liệu `group_report`, `architecture` và `tuning-log`: mô tả cấu hình đang chạy, nêu lý do chọn/không chọn từng phương án, ghi nhận failure mode và đề xuất bước thử tiếp theo theo đúng nguyên tắc A/B chỉ đổi một biến mỗi lần.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, tôi hiểu rõ hơn hai điểm. Thứ nhất là chunking không chỉ là “chia nhỏ cho vừa context”, mà là quyết định ảnh hưởng trực tiếp đến khả năng truy xuất chứng cứ đúng. Với dữ liệu policy/SLA có cấu trúc điều khoản, Recursive split theo heading + paragraph giúp giữ ranh giới ý tốt hơn, nên khi retrieve thường lấy được đoạn “đúng phần việc” thay vì đoạn liên quan chung chung. Thứ hai là evaluation loop giúp tách lỗi rất rõ: Recall cao chưa chắc answer đã tốt. Nhóm tôi có nhiều câu Recall = 5 nhưng Completeness thấp, thậm chí Faithfulness tụt khi đổi retrieval mode. Điều đó cho thấy vấn đề không nằm hoàn toàn ở index, mà còn ở bước chọn evidence và grounding trong generation. Nhờ nhìn scorecard theo từng câu, tôi học được cách quyết định dựa trên dữ liệu thay vì dựa vào cảm giác “hybrid nghe có vẻ mạnh hơn”.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều làm tôi ngạc nhiên nhất là variant hybrid không tạo cải thiện tổng thể như kỳ vọng ban đầu. Trước khi chạy eval, tôi giả thuyết rằng kết hợp semantic + lexical sẽ tăng cả độ liên quan lẫn độ đúng, nhất là với các câu chứa alias hoặc từ khóa đặc thù. Nhưng kết quả thực tế cho thấy trade-off: Completeness có tăng nhẹ, trong khi Faithfulness lại giảm đáng kể ở một số câu khó (đặc biệt q03, q10). Phần mất nhiều thời gian debug nhất là xác định lỗi nằm ở retrieval hay generation, vì nhìn bề ngoài mô hình vẫn trả lời “có vẻ hợp lý”. Khi đối chiếu kỹ score theo từng câu và ghi chú của evaluator, tôi nhận ra có tình huống retrieve đủ ngữ cảnh (Recall cao) nhưng mô hình suy diễn thêm hoặc chọn sai mệnh đề trọng tâm. Bài học rút ra là không nên đánh giá chỉ bằng điểm trung bình; cần soi theo failure case cụ thể để tránh kết luận sai.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

> Chọn 1 câu hỏi trong test_questions.json mà nhóm bạn thấy thú vị.
> Phân tích:
> - Baseline trả lời đúng hay sai? Điểm như thế nào?
> - Lỗi nằm ở đâu: indexing / retrieval / generation?
> - Variant có cải thiện không? Tại sao có/không?

**Câu hỏi:** q10 — "Nếu cần hoàn tiền khẩn cấp cho khách hàng VIP, quy trình có khác không?"

**Phân tích:**

Câu q10 thú vị vì đây là dạng “thiếu context đặc biệt” dễ làm mô hình tự suy diễn. Ở baseline dense, câu trả lời được chấm Faithfulness 5, Relevance 2, Completeness 2. Nghĩa là hệ thống bám đúng tài liệu ở mức cốt lõi (không có chính sách VIP riêng), nhưng trả lời còn ngắn và chưa nêu đủ ngữ cảnh như quy trình chuẩn 3-5 ngày làm việc. Sang variant hybrid, điểm không cải thiện: Relevance và Completeness vẫn thấp, đồng thời Faithfulness rơi xuống 1 ở scorecard variant. Theo tôi, đây không phải lỗi indexing vì nguồn tài liệu refund đã có sẵn và các câu refund khác vẫn hoạt động tốt. Lỗi chính nằm ở retrieval + generation coupling: hybrid có thể đưa thêm chunk “có vẻ liên quan”, nhưng generation lại không giữ grounding chặt, dẫn tới diễn giải vượt ngoài chứng cứ. Trường hợp này cho thấy ưu tiên hiện tại nên là ổn định độ trung thực trước (faithfulness-first), rồi mới tối ưu độ đầy đủ bằng rerank/citation constraints thay vì chỉ đổi mode retrieve.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

> 1-2 cải tiến cụ thể bạn muốn thử.
> Không phải "làm tốt hơn chung chung" mà phải là:
> "Tôi sẽ thử X vì kết quả eval cho thấy Y."

Nếu có thêm thời gian, tôi sẽ thử bật rerank nhưng giữ nguyên chunking Recursive và các tham số còn lại, vì eval cho thấy lỗi lớn nằm ở chọn evidence/grounding chứ không phải thiếu recall. Tôi cũng sẽ chỉnh prompt theo hướng bắt buộc mỗi kết luận phải gắn citation tương ứng, vì q03 và q10 cho thấy mô hình dễ thêm chi tiết khi chứng cứ chưa đủ chặt. Hai thử nghiệm này phù hợp nguyên tắc A/B và dễ quy kết nguyên nhân.

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*
*Ví dụ: `reports/individual/nguyen_van_a.md`*
