# Báo Cáo Cá Nhân - Lab Day 08: RAG Pipeline

**Họ và tên:** Hồ Trọng Duy Quang - 2A202600081
**Vai trò trong nhóm:** Retrieval Owner  
**Ngày nộp:** 2026-04-13  

---

## 1. Tôi đã làm gì trong lab này?

Phần tôi phụ trách chính là `index.py`, tức phần đầu của pipeline trước khi các bạn khác làm retrieval và generation. Tôi implement luồng đọc tài liệu từ `data/docs/`, tách metadata từ header như `Source`, `Department`, `Effective Date`, `Access`, sau đó chia tài liệu thành chunk theo heading `=== ... ===`. Với section dài, tôi xử lý tiếp theo paragraph để hạn chế cắt giữa một điều khoản. Tôi cũng giữ metadata cho từng chunk, đặc biệt là `source`, `section`, `effective_date`, `department`, vì các field này được dùng về sau để citation và tính context recall. Ngoài ra tôi chỉnh `get_embedding()` để dùng OpenAI embedding, rồi nối bước embed vào `build_index()` để lưu vào ChromaDB collection `rag_lab`. Phần của tôi là làm cho dữ liệu đầu vào đủ sạch để retrieval phía sau không phải đoán mò.

## 2. Điều tôi hiểu rõ hơn sau lab này

Trước lab này tôi nghĩ indexing chỉ là “cắt nhỏ document rồi embed”, nhưng khi làm thật thì thấy chunking ảnh hưởng rất mạnh đến chất lượng trả lời. Nếu cắt quá nhỏ, câu hỏi cần nhiều điều kiện sẽ bị thiếu context; nếu cắt quá to, retriever có thể lấy đúng file nhưng LLM lại bị nhiễu. Metadata cũng không phải phần trang trí. Khi scorecard kiểm tra expected source, nếu chunk không có `source` chuẩn như `policy/refund-v4.pdf` hay `it/access-control-sop.md`, việc debug retrieval gần như mù. Các file policy trong lab này có heading rõ, nên giữ ranh giới section giúp câu trả lời bám vào đúng điều khoản hơn.

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

Điều làm tôi mất thời gian nhất là những dòng nằm trước heading đầu tiên. Ví dụ file access control có ghi chú rằng tài liệu này trước đây tên là “Approval Matrix for System Access”. Nếu preprocess bỏ qua phần này vì nghĩ đó là header thừa, câu q07 sẽ mất alias quan trọng. Ban đầu tôi chỉ chú ý parse các dòng `Source`, `Department`, `Effective Date`, nhưng sau đó nhận ra không phải dòng nào trước `=== Section ... ===` cũng nên bỏ. Vì vậy chunk đầu tiên có section `General` vẫn cần được giữ nếu nó chứa nội dung thật. Một khó khăn khác là cân bằng chunk size 400 và overlap 80 để tránh cắt ngang giữa các bước hoặc list điều kiện.

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** q07 - “Approval Matrix để cấp quyền hệ thống là tài liệu nào?”

Tôi chọn câu này vì nó liên quan trực tiếp đến indexing. Expected answer yêu cầu hệ thống nhận ra “Approval Matrix for System Access” là tên cũ của tài liệu “Access Control SOP”, source là `it/access-control-sop.md`. Đây không phải câu hỏi chỉ match theo section chính; nó phụ thuộc vào dòng ghi chú nằm trước `Section 1`. Nếu preprocess xóa dòng đó, dense hoặc hybrid retrieval vẫn có thể tìm access control bằng các từ “cấp quyền”, nhưng sẽ khó trả lời chính xác phần alias.

Trong kết quả của nhóm, context recall của q07 đạt 5, nghĩa là retrieval lấy đúng expected source. Điều này cho thấy indexing đã giữ được thông tin alias và metadata source. Tuy nhiên completeness của q07 vẫn chưa tối đa: trong một số output, câu trả lời chỉ nói “Approval Matrix for System Access” mà chưa nói rõ tên hiện tại là “Access Control SOP” hoặc chưa nêu file `access-control-sop.md`. Root cause vì vậy không nằm ở indexing; evidence đã có trong chunk `General` của access control. Lỗi nằm nhiều hơn ở generation/completeness: model dùng đúng chứng cứ nhưng diễn đạt chưa đủ ý expected answer. Fix cụ thể là chỉnh prompt để khi câu hỏi hỏi “tài liệu nào” thì phải nêu cả tên cũ, tên mới và source file.

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

Nếu có thêm thời gian, tôi sẽ thêm một bước kiểm tra chất lượng chunk tự động sau `build_index()`: in ra các chunk có section `General` và các chunk chứa alias như “Approval Matrix”, “P1”, “store credit”. Kết quả eval cho thấy retrieval đã đúng source, nhưng completeness vẫn thiếu ở q07 và q10, nên tôi muốn thêm metadata phụ kiểu `aliases` để retrieval và prompt dễ nhận biết các tên cũ/tên mới hơn.
