# Báo Cáo Cá Nhân - Lab Day 08: RAG Pipeline

**Họ và tên:** Hồ Trọng Duy Quang - 2A202600081  
**Vai trò trong nhóm:** Retrieval Owner (chunking + metadata) + Documentation Owner (báo cáo nhóm)
<br>
**Ngày nộp:** 2026-04-13

---

## 1. Tôi đã làm gì trong lab này?

Trong lab này, phần tôi phụ trách chính là xây dựng và kiểm tra phần indexing của pipeline RAG. Tôi làm việc với `index.py` để đọc tài liệu trong `data/docs/`, tách metadata từ header như `source`, `department`, `effective_date`, `access`, sau đó chunk tài liệu theo ranh giới tự nhiên của section/paragraph trước khi embed và lưu vào ChromaDB. Ngoài ra, tôi còn viết báo cáo nhóm.

Pipeline của nhóm gồm các bước:

```text
[Raw Docs]
    -> [index.py: Preprocess -> Chunk -> Embed -> Store]
    -> [ChromaDB Vector Store]
    -> [rag_answer.py: Query -> Retrieve -> (Rerank) -> Generate]
    -> [Grounded Answer + Citation]
```

Bộ tài liệu được index có 5 file nội bộ: Refund Policy, SLA P1, Access Control SOP, IT Helpdesk FAQ và HR Leave Policy. Tổng cộng hệ thống tạo ra 30 chunks. Nhóm chốt cấu hình chunking khoảng `CHUNK_SIZE=400` tokens và `CHUNK_OVERLAP=80` tokens, vì mức này đủ giữ ngữ cảnh của từng section nhưng không làm prompt quá dài. Tôi cũng kiểm tra các metadata quan trọng như `source`, `section`, `department`, `effective_date`, `access`, vì các trường này ảnh hưởng trực tiếp đến citation, context recall và debug khi retrieval sai.

Ngoài indexing, tôi tham gia đọc scorecard và tuning log để phân tích vì sao một số câu grading vẫn chỉ đạt Partial/Zero dù source đã được retrieve đúng. Điểm quan trọng tôi rút ra là retrieval đúng file chưa đủ; hệ thống còn phải lấy đúng section, xếp hạng đúng chunk và prompt phải ép model trả lời đủ các chi tiết expected.

## 2. Điều tôi hiểu rõ hơn sau lab này

Trước lab này tôi nghĩ indexing chủ yếu là "cắt document rồi embed", nhưng kết quả grading cho thấy chunking và metadata quyết định rất nhiều đến chất lượng RAG. Nếu chunk không giữ section heading hoặc metadata source, khi câu trả lời sai rất khó biết lỗi đến từ retrieval, rerank hay generation.

Hệ thống của nhóm dùng `text-embedding-3-small` để embed, ChromaDB `PersistentClient` làm vector store, cosine similarity cho dense retrieval, và `gpt-4o-mini` với temperature 0 để sinh câu trả lời grounded. Prompt generation yêu cầu model chỉ trả lời từ retrieved context, thiếu context thì abstain, và cite source bằng chỉ mục chunk.

Tôi cũng hiểu rõ hơn về A/B testing trong RAG. Nhóm không đổi nhiều biến cùng lúc mà chạy theo chuỗi:

```text
Baseline dense -> Variant A hybrid -> Variant B hybrid + rerank
```

Nhờ vậy nhóm biết được Hybrid một mình không chắc tốt hơn Dense. Variant A tăng Faithfulness từ 4.60 lên 4.80 nhưng làm Relevance giảm từ 4.50 xuống 4.20 và Completeness giảm từ 3.30 xuống 3.00, vì BM25 đưa thêm noise candidate vào prompt. Khi bật rerank ở Variant B, CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2` re-score lại candidate pool, giúp Faithfulness đạt 4.90, Relevance phục hồi lên 4.50, Context Recall giữ 5.00 và Completeness tăng lại lên 3.20.

## 3. Điều tôi gặp khó khăn hoặc thấy đáng chú ý

Khó khăn lớn nhất là phân biệt lỗi retrieval với lỗi generation. Có những câu context recall đạt 5.00, nghĩa là hệ thống đã lấy được expected source, nhưng answer vẫn thiếu chi tiết. Ví dụ `gq02` lấy được cả HR Leave Policy và IT Helpdesk FAQ, nhưng câu trả lời vẫn thiếu tên phần mềm "Cisco AnyConnect" và tham chiếu HR Policy. Điều này cho thấy lỗi không chỉ nằm ở vector search mà còn nằm ở việc chọn top chunk cuối cùng và cách model tổng hợp câu trả lời.

Một điểm đáng chú ý khác là hybrid retrieval không tự động cải thiện mọi thứ. BM25 rất hữu ích với keyword như "VPN", "Admin", "P1", nhưng cũng có thể kéo vào chunk nhiễu nếu keyword xuất hiện ở section không đúng. Vì vậy rerank là bước quan trọng: nó không làm Context Recall thay đổi, nhưng giúp sắp xếp lại candidates để chunk phù hợp nhất đi vào prompt.

## 4. Phân tích grading question

### gq05 - Contractor có thể được cấp Admin Access không?

**Câu hỏi:** "Contractor từ bên ngoài công ty có thể được cấp quyền Admin Access không? Nếu có, cần bao nhiêu ngày và có yêu cầu đặc biệt gì?"

**Expected answer:** Có. Access Control SOP áp dụng cho nhân viên, contractor và third-party vendor. Admin Access Level 4 cần IT Manager và CISO phê duyệt, thời gian xử lý 5 ngày làm việc, và cần training bắt buộc về security policy.

Đây là failure mode nổi bật nhất trong grading run. Ở Variant B, câu này có Faithfulness=4, Relevance=5, Context Recall=5 nhưng Completeness chỉ bằng 1, nên kết quả bị đánh Zero. Pipeline retrieve đúng file `it/access-control-sop.md`, nhưng lại lấy nhầm ngữ cảnh Section 4 về emergency temporary access trong 24 giờ thay vì Section 2 về Level 4 Admin Access cần 5 ngày làm việc. Vì vậy câu trả lời mô tả quy trình khẩn cấp: Tech Lead phê duyệt bằng lời, quyền tối đa 24 giờ, phải có ticket sau đó, log vào Security Audit. Những thông tin này grounded với context, nhưng không trả lời đúng expected answer của câu hỏi.

Root cause của `gq05` là retrieval/ranking nhầm section trong cùng một document. Keyword "Admin" và "IT Admin" xuất hiện mạnh ở Section 4, khiến hybrid + BM25 dễ đưa emergency path lên cao. Rerank đã giúp tăng Faithfulness và Relevance so với Variant A, nhưng vẫn chưa đủ để phân biệt "Admin Access Level 4" với "temporary emergency access".

Fix đề xuất của tôi là prepend section heading vào chunk text khi index, ví dụ đưa rõ "Section 2 - Access Level Matrix" hoặc "Section 4 - Emergency Temporary Access" vào nội dung chunk trước khi embed/rerank. Ngoài ra có thể tăng số chunk đưa vào rerank hoặc thêm rule trong prompt: nếu câu hỏi hỏi về Level 4/Admin Access thông thường thì phải ưu tiên section access level/procedure, không dùng emergency path trừ khi câu hỏi có ngữ cảnh khẩn cấp.

### gq07 - Mức phạt khi vi phạm SLA P1

**Câu hỏi:** "Công ty sẽ phạt bao nhiêu nếu team IT vi phạm cam kết SLA P1?"

**Expected answer:** Tài liệu hiện có không quy định mức phạt hay hình thức xử lý khi vi phạm SLA P1. `sla-p1-2026.pdf` chỉ mô tả quy trình xử lý và SLA target, không có điều khoản penalty.

Ở Variant B, model trả lời "Tôi không biết." Đây là câu trả lời an toàn vì không bịa ra mức phạt, nên Faithfulness=5. Tuy nhiên Relevance=1 và Completeness=1 vì answer quá ngắn, không giải thích rằng thông tin penalty không có trong tài liệu hiện có. Điểm này cho thấy lỗi không nằm ở retrieval hay rerank; đây là lỗi ở tầng generation/prompt.

Fix đề xuất là chỉnh grounded prompt rõ hơn: nếu context không đủ để trả lời, model phải nói cụ thể thông tin nào không có trong tài liệu được cung cấp. Ví dụ câu tốt hơn sẽ là: "Tài liệu hiện có không nêu mức phạt nếu team IT vi phạm SLA P1; tài liệu SLA P1 chỉ mô tả target xử lý và quy trình on-call." Câu này vẫn abstain, nhưng trả lời đúng trọng tâm grading hơn.

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

Nếu có thêm thời gian, tôi sẽ ưu tiên hai cải tiến. Thứ nhất là cải thiện indexing cho các chunk bằng cách chèn section heading và một số metadata quan trọng trực tiếp vào text trước khi embed/rerank, đặc biệt với `access_control_sop.md`, để giảm lỗi nhầm section như `gq05`. Thứ hai là chỉnh prompt abstain để câu trả lời thiếu context không chỉ nói "Tôi không biết", mà phải nêu rõ thông tin nào không có trong retrieved documents. Hai thay đổi này nhắm trực tiếp vào hai lỗi lớn còn lại của Variant B: `gq05` ở retrieval/ranking và `gq07` ở generation.
