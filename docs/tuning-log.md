# Tuning Log — RAG Pipeline (Day 08 Lab)

> Completed after running Sprint 4 evaluation.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 2026-04-13  
**Config:**
```
retrieval_mode = "dense"
chunk_size = 400 tokens
overlap = 80 tokens
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = "gpt-4o-mini"
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 4.70 /5 |
| Answer Relevance | 4.80 /5 |
| Context Recall | 5.00 /5 |
| Completeness | 4.10 /5 |

**Câu hỏi yếu nhất (điểm thấp):**
- q10 (VIP refund): Faithfulness = 2, Completeness = 2. Model nhận ra context không có quy trình riêng cho VIP, nhưng câu trả lời vẫn chưa bao phủ tốt expected answer về quy trình tiêu chuẩn 3-5 ngày làm việc.
- q09 (ERR-403-AUTH): Relevance = 3, Context Recall = None vì đây là câu thiếu expected source. Pipeline abstain đúng, nhưng relevance chỉ trung bình do không đưa thêm hướng dẫn xử lý ngoài việc nói thiếu dữ liệu.
- q04/q07/q08: Completeness = 3. Các câu trả lời đúng nguồn và đúng ý chính, nhưng thiếu một số chi tiết như ngoại lệ refund, tên mới `Access Control SOP`, hoặc điều kiện Team Lead phê duyệt remote.

**Giả thuyết nguyên nhân (Error Tree):**
- [ ] Indexing: Chunking cắt giữa điều khoản
- [ ] Indexing: Metadata thiếu effective_date
- [ ] Retrieval: Dense bỏ lỡ exact keyword / alias
- [ ] Retrieval: Top-k quá ít → thiếu evidence
- [x] Generation: Prompt/answer chưa đủ completeness ở các câu có ngoại lệ hoặc thiếu context đặc biệt
- [ ] Generation: Context quá dài → lost in the middle

---

## Variant 1 (Sprint 3)

**Ngày:** 2026-04-13  
**Biến thay đổi:** `retrieval_mode`: `"dense"` → `"hybrid"`  
**Lý do chọn biến này:**
Chọn hybrid vì bộ tài liệu có nhiều exact terms và alias: `P1`, `Level 3`, `Approval Matrix`, `ERR-403-AUTH`, `store credit`. Dense baseline đã có context recall tốt, nhưng hybrid được kỳ vọng làm retrieval ổn định hơn khi câu hỏi dùng keyword/alias. Theo A/B rule, chỉ đổi `retrieval_mode`, giữ nguyên chunking, prompt, top-k và rerank.

**Config thay đổi:**
```
retrieval_mode = "hybrid"
chunk_size = 400 tokens
overlap = 80 tokens
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = "gpt-4o-mini"
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 4.70/5 | 4.70/5 | 0.00 |
| Answer Relevance | 4.80/5 | 4.80/5 | 0.00 |
| Context Recall | 5.00/5 | 5.00/5 | 0.00 |
| Completeness | 4.00/5 | 4.00/5 | 0.00 |

**Nhận xét:**
Hybrid không tạo delta định lượng trong scorecard: cả 10 câu đều tie theo tổng F/R/Rc/C. Tuy vậy, output hybrid có cải thiện nhẹ về chất lượng diễn đạt ở q01 và q06: q01 nêu cả first response 15 phút và resolution 4 giờ; q06 tập trung hơn vào điều kiện escalate sau 10 phút. Không có câu nào variant bị kém hơn baseline theo bảng A/B.

**Kết luận:**
Variant hybrid chưa tốt hơn baseline theo điểm trung bình vì delta của 4 metrics đều bằng 0.00. Nguyên nhân chính là corpus nhỏ và dense baseline đã retrieve đúng expected source cho toàn bộ câu có expected source (context recall = 5.00/5). Nếu cần cải thiện tiếp, nên đổi sang biến khác như prompt/abstain instruction hoặc rerank, vì lỗi còn lại nằm ở completeness và faithfulness của generation hơn là retrieval.

---

## Variant 2 (Sprint 3 - Rerank)

**Ngày:** 2026-04-13  
**Biến thay đổi:** `use_rerank`: `False` → `True`  

**Lý do chọn biến này:**
Sau khi thử hybrid, điểm context recall đã đạt 5.00/5 nhưng completeness vẫn còn thấp ở một số câu. Vì vậy nhóm thử rerank để xem việc sắp xếp lại top candidates trước khi đưa vào prompt có giúp chọn context tập trung hơn không. Theo A/B rule, chỉ đổi `use_rerank`, giữ nguyên `retrieval_mode="dense"`, chunking, top-k và prompt.

**Config thay đổi:**
```
retrieval_mode = "dense"
chunk_size = 400 tokens
overlap = 80 tokens
top_k_search = 10
top_k_select = 3
use_rerank = True
llm_model = "gpt-4o-mini"
```

**Scorecard Variant 2:**
| Metric | Baseline | Variant 2 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 4.70/5 | 4.80/5 | +0.10 |
| Answer Relevance | 4.80/5 | 4.80/5 | 0.00 |
| Context Recall | 5.00/5 | 5.00/5 | 0.00 |
| Completeness | 4.10/5 | 3.80/5 | -0.30 |

**Nhận xét:**
Rerank tạo cải thiện nhỏ ở faithfulness, chủ yếu do q10 tăng từ 2 lên 3. Tuy nhiên completeness giảm từ 4.10 xuống 3.80. Hai câu bị kém hơn baseline là q06 và q08: q06 giảm completeness từ 5 xuống 3 vì rerank đưa thêm context access escalation tạm thời, làm câu trả lời lệch khỏi escalation P1 trong SLA; q08 giảm completeness từ 4 xuống 3 vì câu trả lời thiếu điều kiện phê duyệt remote qua Team Lead/HR Portal. Các câu q01-q05, q07 và q09 không đổi.

**Kết luận:**
Variant rerank không tốt hơn baseline tổng thể. Dù faithfulness tăng +0.10, completeness giảm -0.30, nên trade-off không đáng chọn làm cấu hình cuối. Kết quả cho thấy lỗi chính không phải thiếu expected source, vì context recall vẫn 5.00/5, mà là selection/generation đôi khi chọn hoặc trình bày thiếu chi tiết quan trọng. Nếu tiếp tục tune, nên thử prompt instruction cho completeness hoặc rerank model mạnh hơn, nhưng cần đo riêng từng biến.

---

## Tóm tắt học được

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > Retrieval đã ổn trên bộ test hiện tại, nhưng generation đôi khi thiếu completeness hoặc suy luận thêm ở câu thiếu context đặc biệt, rõ nhất là q10.

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > Hybrid retrieval chưa tạo tác động định lượng vì dense baseline đã đạt context recall 5.00/5. Rerank có tác động hai chiều: faithfulness tăng nhẹ +0.10 nhưng completeness giảm -0.30, nên chưa phải lựa chọn tốt nhất.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > Thử sửa grounded prompt để yêu cầu tách rõ "không có thông tin trong tài liệu" và "quy trình tiêu chuẩn có trong tài liệu", sau đó chạy lại q10 và các câu completeness thấp. Nếu thử rerank tiếp, nên dùng cross-encoder mạnh hơn hoặc điều chỉnh selection để tránh kéo nhầm context như q06.
