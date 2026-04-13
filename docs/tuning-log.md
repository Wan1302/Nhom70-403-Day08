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

---

## Variant A (Hybrid, không rerank)

**Ngày chạy:** 2026-04-13 16:22 -> 16:23  
**File kết quả:**
- `results/scorecard_baseline_variant_a.md`
- `results/scorecard_variant_a.md`
- `results/ab_comparison_variant_a.csv`

**Biến thay đổi:**
- `retrieval_mode: dense -> hybrid`
- `use_rerank` giữ `False`

**Config Variant A:**
```python
retrieval_mode = "hybrid"
top_k_search = 10
top_k_select = 3
use_rerank = False
label = "variant_hybrid"
```

**So sánh trung bình (Baseline A vs Variant A):**
| Metric | Baseline A | Variant A | Delta |
|--------|------------|-----------|-------|
| Faithfulness | 4.60/5 | 4.60/5 | +0.00 |
| Answer Relevance | 4.70/5 | 4.20/5 | -0.50 |
| Context Recall | 5.00/5 | 5.00/5 | +0.00 |
| Completeness | 3.80/5 | 3.50/5 | -0.30 |

**Nhận xét:**
- Variant A không cải thiện tổng thể.
- Giảm rõ nhất ở `q06` và `q09`.
- `q10` có cải thiện relevance, nhưng không đủ để kéo điểm tổng.

**Kết luận Variant A:**
- Không tốt hơn so với baseline ở lần chạy này.

---

## Variant B (Hybrid + rerank)

**Ngày chạy:** 2026-04-13 16:26 -> 16:27  
**File kết quả:**
- `results/scorecard_baseline_variant_b.md`
- `results/scorecard_variant_b.md`
- `results/ab_comparison_variant_b.csv`

**Biến thay đổi:**
- Trên nền Hybrid, đổi `use_rerank: False -> True`

**Config Variant B:**
```python
retrieval_mode = "hybrid"
top_k_search = 10
top_k_select = 3
use_rerank = True
label = "variant_hybrid"
```

**So sánh trung bình (Baseline B vs Variant B):**
| Metric | Baseline B | Variant B | Delta |
|--------|------------|-----------|-------|
| Faithfulness | 4.60/5 | 4.60/5 | +0.00 |
| Answer Relevance | 4.50/5 | 4.80/5 | +0.30 |
| Context Recall | 5.00/5 | 5.00/5 | +0.00 |
| Completeness | 3.70/5 | 4.10/5 | +0.40 |

**Nhận xét:**
- Cải thiện rõ relevance và completeness.
- Câu cải thiện nhiều: `q06`, `q08`, `q09`, `q10`.
- Context recall giữ nguyên 5.00, cho thấy rerank chủ yếu nâng chất lượng chunk dựa vào prompt.

**Kết luận Variant B:**
- Tốt nhất trong 2 variant đã thử.
- Nên chốt cấu hình tạm thời là `hybrid + rerank`.

---

## Tổng hợp để chốt cấu hình

| Cấu hình | Faithfulness | Relevance | Context Recall | Completeness |
|----------|--------------|-----------|----------------|--------------|
| Baseline A | 4.60 | 4.70 | 5.00 | 3.80 |
| Variant A | 4.60 | 4.20 | 5.00 | 3.50 |
| Baseline B | 4.60 | 4.50 | 5.00 | 3.70 |
| Variant B | 4.60 | 4.80 | 5.00 | 4.10 |

**Kết luận cuối**
- `Variant B (hybrid + rerank)` là lựa chọn tốt nhất.
- Vì cả Variant A và Variant B đang để chung `label = "variant_hybrid"`, nên tiếp tục lưu file tách riêng như hiện tại (`*_variant_a*`, `*_variant_b*`) để tránh ghi đè kết quả.

---

## Tóm tắt học được

1. Hybrid một mình chưa đủ; vẫn cần rerank để lọc noise trước khi generate.
2. Rerank không làm mất context recall, nhưng giúp tăng relevance/completeness rõ ràng.
3. Nếu có thêm thời gian, nên tune prompt abstain và rule citation để cải thiện tiếp q04, q09, q10.
