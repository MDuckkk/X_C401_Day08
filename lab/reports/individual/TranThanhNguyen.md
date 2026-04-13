# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Trần Thanh Nguyên  
**Vai trò trong nhóm:** Retrieval Owner  
**Ngày nộp:** 13/4/2026  

## 1. Tôi đã làm gì trong lab này?

Trong lab này, tôi chủ yếu phụ trách phần retrieval của pipeline, tập trung nhiều nhất ở Sprint 3 và có liên kết trực tiếp với Sprint 4 evaluation. Cụ thể, tôi tham gia quyết định và triển khai cấu hình truy xuất cho hai phiên bản baseline và variant, bao gồm việc chọn giữa dense retrieval và hybrid retrieval, thiết lập các tham số như `top_k_search`, `top_k_select`, đồng thời bật cơ chế rerank cho variant. Mục tiêu của tôi là cải thiện chất lượng ngữ cảnh được đưa vào bước generation, thay vì chỉ tăng số lượng chunk được lấy về.

Phần việc của tôi kết nối rất rõ với các thành viên khác trong nhóm. Nếu indexing làm chưa tốt thì retrieval sẽ không có dữ liệu phù hợp để lấy ra; ngược lại, nếu retrieval chọn sai chunk thì prompt và answer generation cũng khó cho ra câu trả lời đúng. Ngoài ra, kết quả retrieval còn ảnh hưởng trực tiếp đến Sprint 4 vì các metric như Context Recall và Faithfulness phụ thuộc rất nhiều vào chất lượng các chunk được chọn. Vì vậy, phần tôi làm đóng vai trò cầu nối giữa indexing, prompting và evaluation.

## 2. Điều tôi hiểu rõ hơn sau lab này

Sau lab này, tôi hiểu rõ hơn hai khái niệm là **hybrid retrieval** và **evaluation loop**. Trước đây, tôi hiểu hybrid retrieval theo hướng khá đơn giản là “kết hợp nhiều cách tìm kiếm thì sẽ tốt hơn”. Nhưng khi làm lab, tôi thấy điều quan trọng không chỉ là kết hợp dense và lexical, mà là kết hợp đúng lúc và đo được hiệu quả bằng scorecard. Trong dữ liệu nhỏ và câu hỏi khá sát với tài liệu như bài lab này, baseline dense đã rất mạnh, nên hybrid không tự động tạo ra khác biệt lớn. Điều đó giúp tôi hiểu rằng retrieval strategy phải được đánh giá trên dữ liệu thật, không thể chỉ dựa vào cảm giác.

Tôi cũng hiểu rõ hơn evaluation loop là một vòng lặp thực nghiệm chứ không phải bước kiểm tra cuối cùng. Mỗi thay đổi nhỏ trong retrieval hoặc rerank cần được phản ánh qua các metric như Faithfulness, Relevance, Context Recall và Completeness. Scorecard không chỉ dùng để “chấm điểm” mà còn để chỉ ra bottleneck đang nằm ở retrieval hay generation. Đây là điểm tôi thấy rất thực tế khi xây dựng RAG pipeline.

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

Điều làm tôi ngạc nhiên nhất là baseline đã đạt kết quả rất cao ngay từ đầu, đặc biệt là **Context Recall = 5.00** ở hầu hết các câu. Ban đầu tôi giả thuyết rằng khi chuyển từ dense sang hybrid và bật rerank thì điểm sẽ tăng rõ ở nhiều metric, nhất là Relevance và Completeness. Tuy nhiên, thực tế cho thấy variant chỉ tăng nhẹ ở Faithfulness từ 4.80 lên 4.90, còn các metric còn lại gần như giữ nguyên.

Khó khăn lớn nhất là phân biệt xem lỗi nằm ở retrieval hay generation. Khi một câu trả lời bị chấm Completeness thấp, phản xạ ban đầu của tôi là nghĩ retriever chưa lấy đủ tài liệu. Nhưng khi nhìn vào Context Recall vẫn đạt tối đa, tôi nhận ra vấn đề không còn nằm ở việc “lấy đúng tài liệu” mà là model chưa diễn đạt đủ ý từ chính ngữ cảnh đã có. Việc debug vì thế mất thời gian ở khâu phân tích scorecard hơn là sửa code. Bài học lớn là không nên đánh đồng mọi lỗi của câu trả lời với lỗi retrieval.

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** Sản phẩm kỹ thuật số có được hoàn tiền không?

**Phân tích:**

Tôi chọn câu q04 vì đây là trường hợp hiếm mà variant có cải thiện so với baseline. Ở baseline, câu trả lời vẫn đúng hướng nhưng Faithfulness chỉ đạt 3, trong khi Relevance và Context Recall đều đạt 5, còn Completeness là 3. Điều này cho thấy retriever đã mang về đúng nguồn tài liệu cần thiết, nhưng phần answer generation có một số chi tiết chưa bám sát hoàn toàn vào context, hoặc diễn đạt khiến giám khảo đánh giá là có yếu tố suy diễn nhẹ. Nói cách khác, lỗi chính không nằm ở indexing, và cũng không hẳn là retrieval thiếu tài liệu, mà nằm ở chất lượng các chunk được chọn cho bước trả lời.

Ở variant, khi dùng hybrid retrieval kết hợp rerank, Faithfulness tăng từ 3 lên 4. Đây là một cải thiện nhỏ nhưng có ý nghĩa, vì nó cho thấy câu trả lời đã grounded hơn trên ngữ cảnh được cung cấp. Tuy vậy, Completeness vẫn giữ ở mức 3, nghĩa là câu trả lời vẫn chưa bao quát hết các điều kiện hoặc ngoại lệ liên quan đến hoàn tiền cho sản phẩm kỹ thuật số. Theo tôi, kết quả này cho thấy rerank đã giúp chọn được ngữ cảnh “sạch” hơn, nhưng chưa giải quyết được vấn đề diễn đạt thiếu ý. Nếu tiếp tục tối ưu, hướng phù hợp hơn sẽ là cải thiện prompt trả lời thay vì chỉ tập trung thêm vào retrieval.

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

Nếu có thêm thời gian, tôi sẽ tách thí nghiệm A/B thành từng bước nhỏ hơn, trước hết là so sánh **dense vs dense + rerank** để đo riêng tác động của rerank. Hiện tại variant thay đổi cả hybrid và rerank cùng lúc nên khó kết luận tuyệt đối biến nào tạo ra cải thiện. Ngoài ra, tôi sẽ thử chỉnh answer prompt theo hướng yêu cầu model liệt kê đầy đủ điều kiện và ngoại lệ, vì scorecard cho thấy Completeness đang thấp hơn các metric còn lại. Điều này có khả năng tạo ra delta rõ hơn so với việc tiếp tục tối ưu retrieval trong bộ dữ liệu hiện tại.
