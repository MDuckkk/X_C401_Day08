# Báo Cáo Cá Nhân - Lab Day 08: RAG Pipeline

**Họ và tên:** Hoàng Quang Thắng
**Vai trò trong nhóm:** Retrieval Owner
**Ngày nộp:** 13/04/2026 
**Độ dài yêu cầu:** 500-800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này tôi chủ yếu tham gia ở Sprint 2 và Sprint 3, tập trung vào retrieval và tuning pipeline. 

Ở Sprint 2, tôi làm việc với baseline dense retrieval trong `rag_answer.py`, kiểm tra cách query ChromaDB và cách build grounded answer từ các chunk đã retrieve. 

Ở Sprint 3, tôi tập trung vào variant hybrid, so sánh dense với hybrid và theo dõi xem thay đổi retrieval mode có cải thiện được câu trả lời hay không. Tôi dùng các file kết quả trong `results/` để xem câu nào hệ thống trả lời tốt, câu nào còn fail và vì sao. Phần việc của tôi kết nối trực tiếp với `index.py`, vì chất lượng retrieval ở Sprint 2 và 3 phụ thuộc nhiều vào chunking, metadata và embedding từ bước indexing.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này tôi hiểu rõ hơn hai concept là dense retrieval và hybrid retrieval. 

Trước đây tôi nghĩ retrieval chủ yếu là “tìm ra đúng tài liệu”, nhưng khi làm Sprint 2 và 3 tôi thấy chưa đủ. Dense retrieval có thể kéo đúng source nhưng answer vẫn thiếu ý, vì các chunk được đưa vào prompt chưa chắc đã bao phủ đủ điều kiện, ngoại lệ hoặc chi tiết quan trọng. 

Hybrid retrieval giúp cải thiện ở chỗ nó giữ được cả semantic match lẫn keyword match, đặc biệt khi câu hỏi chứa alias hoặc từ khóa đặc thù như `Approval Matrix`, `P1`, `ERR-403`. Tôi cũng hiểu rõ hơn A/B rule: khi tuning chỉ nên đổi một biến, ví dụ đổi retrieval mode từ dense sang hybrid, để biết chính xác thay đổi nào tạo ra cải thiện thay vì đổi nhiều thứ cùng lúc.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều làm tôi ngạc nhiên nhất là dense retrieval có thể lấy đúng source nhưng câu trả lời vẫn chưa tốt. 

Ban đầu tôi nghĩ nếu retriever kéo đúng tài liệu thì baseline sẽ ổn, nhưng thực tế có nhiều câu như Q05, Q09 và Q10 vẫn trả lời rất yếu. Khó khăn lớn nhất là phân biệt lỗi retrieval với lỗi generation trong phạm vi Sprint 2 và 3. Với một số câu, dense và hybrid đều lấy được tài liệu liên quan, nhưng hybrid không cải thiện nhiều vì phần answer vẫn chưa tổng hợp được đúng ý cần thiết. 

Ngoài ra, việc tuning cũng khó ở chỗ một số câu hỏi mang tính alias hoặc nhiều keyword đặc thù, nên dense retrieval đơn thuần chưa đủ mạnh, còn hybrid thì có cải thiện nhưng không phải lúc nào cũng rõ ràng ở mọi câu.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** `Q05 - Quy trình xử lý sự cố mạng là gì?`

**Phân tích:**

Đây là câu tôi thấy thú vị nhất vì nó cho thấy khá rõ giới hạn của baseline dense retrieval. 

Ở baseline, câu này có `faithfulness = 2`, `relevance = 1`, `context recall = 5`, `completeness = 1`. 

Answer của baseline là kiểu “Tôi không tìm thấy thông tin này trong tài liệu hiện có. Bạn vui lòng tạo ticket hoặc liên hệ IT Helpdesk…”. 

Sang variant hybrid rerank, kết quả gần như không cải thiện: relevance vẫn `1`, completeness vẫn `1`, thậm chí faithfulness còn giảm xuống `1`. Theo tôi, lỗi chính nằm ở retrieval của Sprint 2 và 3 chưa kéo được đúng loại evidence cho câu hỏi này. 

Dù scorecard cho thấy recall cao, các chunk được chọn vẫn thiên về helpdesk chung chung chứ không chứa đúng quy trình xử lý sự cố mạng. Vì vậy khi sang bước generation, model chỉ tạo câu trả lời an toàn kiểu “hãy liên hệ IT”. 

Variant hybrid không giúp nhiều vì truy vấn này cần evidence rất đặc thù, còn corpus hiện tại không thật sự hỗ trợ tốt cho dạng network troubleshooting process.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi sẽ tiếp tục thử các cải tiến trong phạm vi Sprint 2 và 3.

 Hướng đầu tiên là tăng chất lượng retrieval cho các câu fail nặng như Q05, Q09, Q10 bằng query transformation hoặc rerank tốt hơn. Hướng thứ hai là điều chỉnh cách chọn top chunks để hybrid không chỉ lấy đúng source mà còn lấy đúng phần evidence quan trọng. 
 
 Kết quả hiện tại cho thấy vấn đề lớn nhất không phải chỉ là tìm đúng tài liệu, mà là tìm đủ và đúng chunk cần thiết.

---