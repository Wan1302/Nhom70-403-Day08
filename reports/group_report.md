# Báo Cáo Nhóm - Lab Day 08: RAG Pipeline

**Nhóm:** 70  
**Phòng:** 403  
**Ngày nộp:** 2026-04-13  
**Log grading chính:** `logs/grading_run_variant_b.json`  
**Scorecard chính:** `results/scorecard_variant_b.md`
<br>
**Thành viên:** 
- Hồ Trọng Duy Quang - 2A202600081
- Hồ Trần Đình Nguyên - 2A202600080
- Hồ Đắc Toàn - 2A202600057


## 1. Pipeline nhóm đã xây dựng

Nhóm xây dựng một pipeline RAG end-to-end cho trợ lý nội bộ CS + IT Helpdesk, trả lời câu hỏi về SLA P1, refund policy, access control, IT Helpdesk FAQ và HR Leave Policy. Luồng tổng thể là:

```text
[Raw Docs]
  -> [index.py: Preprocess -> Chunk -> Embed -> Store]
  -> [ChromaDB Vector Store]
  -> [rag_answer.py: Query -> Retrieve -> (Rerank) -> Generate]
  -> [Grounded Answer + Citation]
```

Ở Sprint 1, `index.py` đọc 5 file trong `data/docs/`, extract metadata từ header (`source`, `section`, `department`, `effective_date`, `access`), chunk theo heading `=== ... ===`, sau đó fallback theo paragraph nếu section dài. Nhóm dùng `CHUNK_SIZE=400` tokens và `CHUNK_OVERLAP=80` tokens để giữ đủ ngữ cảnh ở biên chunk. Tổng số chunk được ghi trong `docs/architecture.md` là 30 chunks.

Embedding dùng OpenAI `text-embedding-3-small`, lưu vào ChromaDB `PersistentClient` tại `chroma_db_runtime/`, similarity metric là cosine. Ở Sprint 2, baseline dùng dense retrieval với `top_k_search=10`, `top_k_select=3`, không rerank. Ở Sprint 3, nhóm thử hybrid retrieval bằng dense + BM25 qua Reciprocal Rank Fusion, rồi thêm CrossEncoder rerank `cross-encoder/ms-marco-MiniLM-L-6-v2`. Generation dùng `gpt-4o-mini`, temperature 0, max tokens 512, prompt bắt buộc trả lời từ retrieved context và citation dạng `[1]`.

## 2. Quyết định kỹ thuật quan trọng nhất

Quyết định quan trọng nhất là chọn cấu hình **Hybrid + CrossEncoder rerank** làm best variant thay vì chỉ dùng dense hoặc hybrid đơn thuần.

Bối cảnh vấn đề: corpus có cả câu tự nhiên tiếng Việt và nhiều keyword/tên riêng như `P1`, `VPN`, `Admin Access`, `store credit`, nên dense retrieval có lợi về semantic matching, còn BM25 có lợi với exact keyword. Tuy nhiên tuning log cho thấy hybrid không rerank kéo thêm noise vào candidate pool.

| Phương án | Ưu điểm | Nhược điểm |
|-----------|---------|------------|
| Dense baseline | Ổn định, ít keyword noise | Có thể hụt exact term/cross-document context |
| Hybrid dense + BM25 | Bắt keyword tốt hơn, mở rộng candidate pool | BM25 có thể đưa sai section/source lên cao |
| Hybrid + rerank | Lọc lại candidate bằng CrossEncoder trước khi prompt | Tốn thêm thời gian/model local |

Bằng chứng rõ nhất nằm ở kết quả A/B. Baseline dense đạt Faithfulness 4.60, Relevance 4.50, Context Recall 5.00, Completeness 3.30. Variant A chỉ đổi một biến `retrieval_mode: dense -> hybrid`, Faithfulness tăng lên 4.80 nhưng Relevance giảm xuống 4.20 và Completeness giảm xuống 3.00. Điều này cho thấy hybrid một mình chưa đủ.

Sau đó Variant B chỉ đổi một biến so với Variant A: `use_rerank: False -> True`. Kết quả tăng lên Faithfulness 4.90, Relevance 4.50, Context Recall giữ 5.00, Completeness 3.20. Vì vậy nhóm chốt Variant B: `retrieval_mode="hybrid"`, `top_k_search=10`, `top_k_select=3`, `use_rerank=True`.

## 3. Kết quả grading questions

Nhóm chạy 10 câu grading bằng Variant B lúc 17:50 ngày 2026-04-13, log tại `logs/grading_run_variant_b.json`. Theo phân loại trong `docs/tuning-log.md`, kết quả gồm 9 câu Partial và 1 câu Zero (`gq05`), không có câu bị hallucination penalty rõ ràng ở best variant. Nếu quy đổi theo rubric Partial = 50% điểm câu, ước tính raw score là **44/98**, tương đương khoảng **13.47/30** cho phần grading questions.

Câu xử lý tốt nhất là `gq06`: pipeline lấy đúng cả `it/access-control-sop.md` và `support/sla-p1-2026.pdf`, trả lời đúng emergency temporary access tối đa 24 giờ, cần Tech Lead phê duyệt bằng lời, ticket chính thức sau 24 giờ và log Security Audit. Câu này còn thiếu hotline ext. 9999 nên chỉ xem là Partial.

Câu fail rõ nhất là `gq05`. Pipeline retrieve đúng document `it/access-control-sop.md`, nhưng lấy nhầm Section 4 về emergency temporary access 24 giờ thay vì Section 2 về Level 4 Admin Access cần IT Manager + CISO, 5 ngày làm việc và security training. Đây là lỗi retrieval/ranking nhầm section trong cùng document, không phải lỗi thiếu source.

`gq07` là câu insufficient context. Model trả lời "Tôi không biết.", không bịa mức phạt nên an toàn về Faithfulness, nhưng bị Partial vì không nói rõ rằng tài liệu hiện có không quy định penalty cho SLA P1.

## 4. A/B Comparison

Nhóm tuân thủ quy tắc A/B: mỗi lần chỉ đổi một biến.

**Compare 1 - Dense baseline vs Hybrid không rerank**

| Metric | Baseline | Variant A | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 4.60 | 4.80 | +0.20 |
| Relevance | 4.50 | 4.20 | -0.30 |
| Context Recall | 5.00 | 5.00 | 0.00 |
| Completeness | 3.30 | 3.00 | -0.30 |

Kết luận: Hybrid không rerank kém hơn baseline ở Relevance và Completeness. BM25 giúp mở rộng candidates nhưng đưa thêm noise, ví dụ `gq02` Relevance giảm từ 5 xuống 3.

**Compare 2 - Hybrid không rerank vs Hybrid + rerank**

| Metric | Variant A | Variant B | Delta |
|--------|-----------|-----------|-------|
| Faithfulness | 4.80 | 4.90 | +0.10 |
| Relevance | 4.20 | 4.50 | +0.30 |
| Context Recall | 5.00 | 5.00 | 0.00 |
| Completeness | 3.00 | 3.20 | +0.20 |

Kết luận: rerank là thay đổi có ích nhất sau khi bật hybrid. CrossEncoder không làm mất expected sources (Context Recall giữ 5.00), nhưng sắp xếp lại candidate tốt hơn trước khi build prompt. Variant B là cấu hình tốt nhất trong ba bản đã thử, dù Completeness vẫn thấp hơn baseline một chút do `gq05` bị C=1 kéo xuống.

## 5. Phân công và đánh giá nhóm

| Thành viên | Phần đã làm | Sprint |
|------------|-------------|--------|
| Hồ Trọng Duy Quang | Indexing: preprocess docs, chunking, metadata, embedding/index analysis | Sprint 1 |
| Hồ Trần Đình Nguyên | RAG answering: dense/sparse/hybrid retrieval, RRF, rerank, grounded prompt | Sprint 2-3 |
| Hồ Đắc Toàn | Evaluation: LLM-as-Judge, scorecard, A/B comparison, tuning documentation | Sprint 4 |

Điều nhóm làm tốt là chạy được đủ pipeline end-to-end và có logging/scorecard cho 3 cấu hình thay vì chỉ nộp một bản. `eval.py` có LLM-as-Judge cho Faithfulness, Relevance, Context Recall và Completeness, có fallback heuristic nếu judge lỗi. `docs/architecture.md` và `docs/tuning-log.md` ghi rõ chunking decision, retrieval config và lý do chọn Variant B.

Điều nhóm làm chưa tốt là chưa xử lý triệt để các lỗi sau tuning. `gq05` vẫn nhầm section trong cùng `access_control_sop.md`, còn prompt abstain quá ngắn làm `gq07` chỉ trả lời "Tôi không biết." thay vì giải thích thiếu context. Một số câu multi-detail như `gq02` và `gq09` cũng thiếu chi tiết nhỏ dù retrieve đúng source.

## 6. Nếu có thêm 1 ngày, nhóm sẽ làm gì?

Nhóm sẽ ưu tiên hai cải tiến có bằng chứng từ scorecard. Thứ nhất, prepend section heading vào chunk text khi index, ví dụ "Section 2 - Level 4 Admin Access" hoặc "Section 4 - Emergency Temporary Access", để fix `gq05` nhầm section. Thứ hai, chỉnh `build_grounded_prompt()` để khi thiếu context, model phải nói rõ thông tin nào không có trong tài liệu; thay đổi này nhắm trực tiếp vào `gq07`.
