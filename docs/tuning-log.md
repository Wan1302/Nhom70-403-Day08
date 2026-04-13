# Tuning Log - RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được. A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Config gốc (code):**
```python
retrieval_mode = "dense"
chunk_size = 400 tokens
overlap = 80 tokens
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = "gpt-4o-mini"
```

**Kết quả baseline:**
| Metric | Score |
|--------|-------|
| Faithfulness | 4.50/5 |
| Answer Relevance | 4.50/5 |
| Context Recall | 5.00/5 |
| Completeness | 3.90/5 |

---

## Variant A (Hybrid, không rerank)

**Ngày chạy:** 2026-04-13 16:57  
**File kết quả:**
- `results/scorecard_variant_a.md`
- `results/ab_comparison_baseline_vs_a.csv`

**Biến thay đổi:**
- `retrieval_mode: dense → hybrid`
- `use_rerank` giữ `False`

**Config Variant A:**
```python
retrieval_mode = "hybrid"
top_k_search = 10
top_k_select = 3
use_rerank = False
label = "variant_hybrid"
```

**So sánh: Baseline vs Variant A:**
| Metric | Baseline | Variant A | Δ |
|--------|----------|-----------|---|
| Faithfulness | 4.50/5 | 4.70/5 | +0.20 |
| Answer Relevance | 4.50/5 | 4.10/5 | **−0.40** |
| Context Recall | 5.00/5 | 5.00/5 | 0.00 |
| Completeness | 3.90/5 | 3.60/5 | **−0.30** |

**Câu thụt lùi rõ nhất:**
- `q06` (SLA escalation P1): Relevance 5→3, Completeness 4→1 — hybrid retrieve nhầm chunk access control thay vì chunk escalation SLA.
- `q09` (Insufficient context): Relevance 4→1 — hybrid lấy thêm chunk nhiễu khiến model tự suy diễn thay vì abstain.

**Nhận xét:**
- Faithfulness tăng nhẹ (+0.20) vì hybrid lấy đúng nguồn hơn ở một số câu.
- Tuy nhiên, Relevance và Completeness đều giảm do BM25 đưa vào noise candidates không đúng topic.
- `q06` là case điển hình: BM25 match keyword "IT Admin" từ access control doc thay vì SLA doc.

**Kết luận Variant A:**
- Hybrid một mình chưa đủ — BM25 mở rộng candidate pool nhưng cần bước lọc thêm.

---

## Variant B (Hybrid + rerank)

**Ngày chạy:** 2026-04-13 16:59  
**File kết quả:**
- `results/scorecard_variant_b.md`
- `results/ab_comparison_a_vs_b.csv`

**Biến thay đổi (so với Variant A):**
- Trên nền Hybrid, bật `use_rerank: False → True`

**Config Variant B:**
```python
retrieval_mode = "hybrid"
top_k_search = 10
top_k_select = 3
use_rerank = True
label = "variant_hybrid"
```

**So sánh: Variant A vs Variant B:**
| Metric | Variant A | Variant B | Δ |
|--------|-----------|-----------|---|
| Faithfulness | 4.70/5 | 4.70/5 | 0.00 |
| Answer Relevance | 4.10/5 | 4.80/5 | **+0.70** |
| Context Recall | 5.00/5 | 5.00/5 | 0.00 |
| Completeness | 3.60/5 | 4.00/5 | **+0.40** |

**Câu cải thiện rõ nhất:**
- `q06` (SLA escalation P1): Relevance 3→5, Completeness 1→5 — rerank đẩy đúng chunk escalation lên top, loại noise access control.
- `q09` (Insufficient context): Relevance 1→5 — rerank giúp model nhận ra context không đủ và abstain đúng.

**Nhận xét:**
- Rerank (CrossEncoder `ms-marco-MiniLM-L-6-v2`) re-score lại toàn bộ candidates từ hybrid, đẩy chunk đúng nhất lên vị trí đầu trước khi build prompt.
- Relevance tăng +0.70 — mức cải thiện rõ ràng nhất trong toàn bộ experiment.
- Context Recall giữ nguyên 5.00 — rerank không làm mất source, chỉ sắp xếp lại thứ tự.

**Kết luận Variant B:**
- Tốt nhất trong 3 cấu hình đã thử.
- Chốt cấu hình: `hybrid + rerank`.

---

## Tổng hợp để chốt cấu hình

| Cấu hình | Faithfulness | Relevance | Context Recall | Completeness |
|----------|-------------|-----------|----------------|-------------|
| Baseline (dense) | 4.50 | 4.50 | 5.00 | 3.90 |
| Variant A (hybrid) | 4.70 | 4.10 | 5.00 | 3.60 |
| **Variant B (hybrid + rerank)** | **4.70** | **4.80** | **5.00** | **4.00** |

**Kết luận cuối**
- `Variant B (hybrid + rerank)` là lựa chọn tốt nhất — dẫn đầu ở Relevance (+0.30 so với baseline, +0.70 so với Variant A) và Completeness (+0.10 so với baseline, +0.40 so với Variant A).
- Vì cả Variant A và Variant B đang để chung `label = "variant_hybrid"`, nên tiếp tục lưu file tách riêng như hiện tại (`*_variant_a*`, `*_variant_b*`) để tránh ghi đè kết quả.

---

## Tóm tắt học được

1. Hybrid một mình chưa đủ; vẫn cần rerank để lọc noise trước khi generate.
2. Rerank không làm mất context recall, nhưng giúp tăng relevance/completeness rõ ràng (+0.70/+0.40).
3. `q06` (SLA escalation P1) là câu chỉ điểm rõ nhất: hybrid lấy sai chunk, rerank sửa được.
4. Nếu có thêm thời gian, nên tune prompt abstain và rule citation để cải thiện tiếp q04, q09, q10.
