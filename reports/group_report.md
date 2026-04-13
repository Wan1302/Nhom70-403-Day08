# Báo Cáo Nhóm — Lab Day 08: RAG Pipeline

**Nhóm:** 70  
**Môn:** 403  
**Ngày nộp:** 2026-04-13

---

## 1. Tổng quan hệ thống

Hệ thống được xây dựng là trợ lý nội bộ dạng RAG (Retrieval-Augmented Generation) phục vụ khối CS và IT Helpdesk. Hệ thống trả lời câu hỏi về SLA, chính sách hoàn tiền, access control và HR policy dựa trên tài liệu nội bộ, có citation rõ nguồn và có thể audit.

### Kiến trúc pipeline

```
[Tài liệu nội bộ]
    → index.py: Preprocess → Chunk → Embed → Lưu ChromaDB
    → rag_answer.py: Query → Retrieve (Hybrid/Dense/Sparse) → [Rerank] → Generate
    → Câu trả lời có citation
```

### Tài liệu được index

| File | Nguồn | Department | Số chunk |
|------|-------|------------|----------|
| `policy_refund_v4.txt` | `policy/refund-v4.pdf` | CS | 6 |
| `sla_p1_2026.txt` | `support/sla-p1-2026.pdf` | IT | 5 |
| `access_control_sop.txt` | `it/access-control-sop.md` | IT Security | 8 |
| `it_helpdesk_faq.txt` | `support/helpdesk-faq.md` | IT | 6 |
| `hr_leave_policy.txt` | `hr/leave-policy-2026.pdf` | HR | 5 |

**Tổng:** 30 chunks — Chunk size ~400 tokens, overlap ~80 tokens.

### Thông số kỹ thuật

| Thành phần | Lựa chọn |
|------------|----------|
| Embedding model | `text-embedding-3-small` (OpenAI) |
| Vector store | ChromaDB `PersistentClient` |
| Similarity | Cosine |
| LLM sinh câu trả lời | `gpt-4o-mini`, temperature=0, max_tokens=512 |
| Đánh giá (judge) | `gpt-4o` — LLM-as-Judge trên 4 metrics |

---

## 2. Thiết kế thực nghiệm A/B

Nhóm chạy 2 vòng thực nghiệm theo chuỗi, mỗi vòng **chỉ thay đổi một biến**, đánh giá trên 10 câu hỏi thuộc 5 danh mục: SLA, Refund, Access Control, IT Helpdesk, Insufficient Context.

```
Baseline (dense) → Variant A (hybrid) → Variant B (hybrid + rerank)
     Compare 1: Δ dense→hybrid            Compare 2: Δ no rerank→rerank
```

**Metrics đánh giá (thang 1–5):**
- **Faithfulness** — câu trả lời có bám sát context không
- **Relevance** — câu trả lời có đúng trọng tâm câu hỏi không
- **Context Recall** — có retrieve đúng source cần thiết không
- **Completeness** — có bao phủ đủ các điểm chính trong expected answer không

---

## 3. Kết quả A/B

### Compare 1 — Baseline (dense) vs Variant A (hybrid, không rerank)

**Biến thay đổi:** `retrieval_mode: dense → hybrid (dense + BM25)`

| Metric | Baseline | Variant A | Δ |
|--------|---------|-----------|---|
| Faithfulness | 4.50/5 | 4.70/5 | +0.20 |
| Relevance | **4.50/5** | **4.10/5** | **−0.40** |
| Context Recall | 5.00/5 | 5.00/5 | 0.00 |
| Completeness | **3.90/5** | **3.60/5** | **−0.30** |

**Câu nổi bật:**
- `q06` (SLA escalation P1): Relevance 5→3, Completeness 4→1 — BM25 retrieve nhầm chunk access control ("IT Admin") thay vì chunk SLA escalation.
- `q09` (Insufficient context): Relevance 4→1 — hybrid lấy thêm chunk nhiễu khiến model suy diễn thay vì abstain.

**Kết luận Compare 1:** Hybrid không rerank làm **giảm** Relevance và Completeness. BM25 mở rộng candidate pool nhưng cũng đưa vào noise.

---

### Compare 2 — Variant A (hybrid) vs Variant B (hybrid + rerank)

**Biến thay đổi:** `use_rerank: False → True`  
**Reranker:** `CrossEncoder: cross-encoder/ms-marco-MiniLM-L-6-v2`

| Metric | Variant A | Variant B | Δ |
|--------|-----------|-----------|---|
| Faithfulness | 4.70/5 | 4.70/5 | 0.00 |
| Relevance | **4.10/5** | **4.80/5** | **+0.70** |
| Context Recall | 5.00/5 | 5.00/5 | 0.00 |
| Completeness | **3.60/5** | **4.00/5** | **+0.40** |

**Câu nổi bật:**
- `q06` (SLA escalation P1): Relevance 3→5, Completeness 1→5 — rerank đẩy đúng chunk SLA lên top, loại bỏ noise từ access control.
- `q09` (Insufficient context): Relevance 1→5 — rerank giúp model nhận ra context không đủ và abstain đúng.
- Faithfulness giữ nguyên 4.70 — rerank không làm giảm độ bám context.

**Kết luận Compare 2:** Rerank cải thiện Relevance **+0.70** và Completeness **+0.40** — mức cải thiện rõ nhất trong toàn bộ experiment.

---

## 4. So sánh tổng hợp 3 cấu hình

| Cấu hình | Faithfulness | Relevance | Context Recall | Completeness |
|----------|-------------|-----------|----------------|-------------|
| Baseline (dense) | 4.50 | 4.50 | 5.00 | 3.90 |
| Variant A (hybrid) | 4.70 | 4.10 | 5.00 | 3.60 |
| **Variant B (hybrid + rerank)** | **4.70** | **4.80** | **5.00** | **4.00** |

---

## 5. Kết luận — Chọn Variant B

**Cấu hình được chốt: `hybrid + CrossEncoder rerank`**

**Lý do:**
1. **Hybrid một mình không đủ** — BM25 mở rộng candidate pool nhưng đưa vào noise. Nếu không lọc lại, Relevance và Completeness giảm so với baseline (thấy rõ ở Variant A).
2. **Rerank là bước then chốt** — CrossEncoder re-score lại toàn bộ candidates từ hybrid, đẩy chunk đúng nhất lên đầu trước khi build prompt. Case `q06` là bằng chứng trực tiếp: Variant A lấy sai chunk, Variant B sửa đúng nhờ rerank.
3. **Context Recall không bị ảnh hưởng** — giữ nguyên 5.00/5 ở cả 3 cấu hình, chứng tỏ retrieval bao phủ đúng source; vấn đề nằm ở ranking, không phải coverage.

**Vấn đề còn tồn tại cần xử lý tiếp:**
- `q04`: Faithfulness=2 (cả 3 cấu hình) — LLM tự thêm exception "lỗi nhà sản xuất" không có trong context; cần thêm rule abstain trong prompt.
- `q09`: Completeness=2 (cả 3 cấu hình) — câu hỏi về ERR-403-AUTH không có trong tài liệu; cần tune prompt abstain rõ hơn.
- `q10`: Completeness=3 — thiếu thông tin "3–5 ngày làm việc"; cần bổ sung chunk hoặc tune citation rule.
