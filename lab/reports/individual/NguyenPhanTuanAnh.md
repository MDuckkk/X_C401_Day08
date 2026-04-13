# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Phan Tuấn Anh - 2A2026004
**Vai trò trong nhóm:** Tech Lead
**Ngày nộp:** 13/4/2026 
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Em làm chủ yếu sprint 1, 2, 3 và 4. 
Trong sprint 1 em implement phương thức chunking semantic, và xử lý metadata, lưu vào firebase, sprint 2 implement dense và hybrid retrival với ChromaDB và OpenAI embedding, sprint 3 implement và so sánh dense với hybrid retrival dùng semantic chunking với chunking size là 500 overlap là 100. Sprint 4 so sánh phương pháp đã implement ở sprint 3 với dense và hybrid retrival dùng recursive chunking 320 overlap 60. Công việc của em kết nối bằng cách ở sprint 4 tổng hợp lại và so sánh với kết quả của phương pháp khác, ngoài ra trong lúc làm frontend và backend em còn tiếp nhận và tiếp thu các ý kiến, đóng góp của các bạn về design. 

_________________

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, em hiểu rõ hơn về chunking và hybrid retrieval. Trước đây em nghĩ chỉ cần chia tài liệu thành các đoạn nhỏ là đủ, nhưng khi làm lab em thấy cách chia ảnh hưởng trực tiếp đến chất lượng truy xuất. Semantic chunking giúp giữ các câu có liên quan ở cùng một đoạn nên ngữ cảnh không bị đứt gãy, còn recursive chunking thì dễ kiểm soát kích thước nhưng đôi khi cắt mất ý nghĩa của một phần tài liệu. Em cũng hiểu rõ hơn hybrid retrieval vì nó kết hợp ưu điểm của dense search và lexical search: dense search bắt được ý gần nghĩa, còn lexical search giữ được các từ khóa chính xác. Khi hai nguồn này được phối hợp tốt, hệ thống trả lời ổn định hơn và ít bỏ sót thông tin quan trọng hơn.

_________________

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều mà khiến em gặp khó khăn hóa ra không phải là bản thân phần bài lab này mà là do bản thân sai lầm của em khi đã không đọc rõ yêu cầu sprint 1, sprint 1 yêu cầu rằng phải trích xuất trường metadata ra trước khi nhập tài liệu vào database nhưng em đã nhập toàn bộ tài liệu mà không trích xuất metadata vào khiến cho query đưa ra trích dẫn sai khiến cho phần chạy eval đưa ra recall là 0 khiến em phải quay lại debug toàn bộ từ đầu và cuối cùng với thời gian có hạn phải đưa ra quyết định là viết wrapper biến output từ citation như [Nguồn: SLA_P1_2026] thành [support/sla-p1-2026.pdf].

_________________

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

> Chọn 1 câu hỏi trong test_questions.json mà nhóm bạn thấy thú vị.
> Phân tích:
> - Baseline trả lời đúng hay sai? Điểm như thế nào?
> - Lỗi nằm ở đâu: indexing / retrieval / generation?
> - Variant có cải thiện không? Tại sao có/không?

**Câu hỏi:** Ai phải phê duyệt để cấp quyền Level 3?

**Phân tích:**

Với câu hỏi này, baseline trả lời gần đúng nhưng chưa đầy đủ. Scorecard baseline cho thấy faithfulness 4/5, relevance 5/5, recall 5/5 và completeness 3/5. Nghĩa là mô hình đã lấy đúng tài liệu và trả lời đúng hướng, nhưng vẫn bỏ sót một phần quan trọng là IT Admin. Lỗi ở đây chủ yếu nằm ở generation, không phải indexing hay retrieval, vì context recall đạt 5/5 và nguồn cần thiết đã được truy xuất đầy đủ. Ở variant, tình hình không cải thiện mà còn tệ hơn: faithfulness giảm xuống 1/5 dù relevance và recall vẫn giữ ở mức 5/5. Điều này cho thấy retrieval vẫn tìm đúng tài liệu, nhưng câu trả lời sinh ra lại bị sai và thiếu căn cứ, chỉ giữ được Line Manager mà làm mất IT Admin và IT Security. Vì vậy, variant không giúp cải thiện chất lượng trả lời cho câu hỏi này mà còn làm tăng hallucination.

_________________

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, em sẽ thử chuẩn hóa và kiểm tra metadata trước khi đưa tài liệu vào database, vì lỗi ở sprint 1 cho thấy chỉ cần mapping citation sai là recall và trích dẫn bị lệch ngay. Em cũng sẽ thử thêm một lớp kiểm tra grounding cho prompt, hoặc rerank lại kết quả retrieval trước khi sinh câu trả lời, vì ở câu q03 variant vẫn lấy đúng tài liệu nhưng faithfulness lại giảm mạnh xuống 1/5. Hai thay đổi này đều nhắm trực tiếp vào lỗi đã lộ ra trong eval, không phải cải tiến chung chung.

_________________

---

