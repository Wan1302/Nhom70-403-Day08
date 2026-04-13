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
| Faithfulness | 4.60/5 |
| Answer Relevance | 4.50/5 |
| Context Recall | 5.00/5 |
| Completeness | 3.30/5 |

**Câu nổi bật ở baseline:**
- `gq05` (Contractor Admin Access): Faithfulness=1 — model tự suy diễn contractors có thể được cấp quyền Admin Access trong khi context chỉ đề cập DevOps/SRE/IT Admin. Đây là lỗi hallucination nghiêm trọng nhất.
- `gq07` (Insufficient context): Relevance=1 — model abstain bằng "Tôi không biết" mà không giải thích lý do thông tin không có trong tài liệu.
- `gq02` (Cross-document VPN): Completeness=2 — answer chỉ trả lời số thiết bị, thiếu tên phần mềm "Cisco AnyConnect" và quy định VPN bắt buộc từ HR Policy.

---

## Variant A (Hybrid, không rerank)

**Ngày chạy:** 2026-04-13 17:48  
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
| Faithfulness | 4.60/5 | 4.80/5 | +0.20 |
| Answer Relevance | 4.50/5 | 4.20/5 | **−0.30** |
| Context Recall | 5.00/5 | 5.00/5 | 0.00 |
| Completeness | 3.30/5 | 3.00/5 | **−0.30** |

**Câu thụt lùi rõ nhất:**
- `gq02` (VPN remote): Relevance 5→3 — BM25 đưa vào candidates thiếu ngữ cảnh, answer chỉ nêu số thiết bị mà không đủ source verification.
- `gq05` (Contractor Admin Access): Faithfulness vẫn thấp (3/5) — hybrid retrieve được chunk nhưng model thêm claim không có trong context (1 ngày, Line Manager phê duyệt) thay vì đúng quy trình 5 ngày + IT Manager + CISO.
- `gq03` (Flash Sale refund): Completeness 4→3 — thiếu giải thích hai ngoại lệ rõ ràng.

**Nhận xét:**
- Faithfulness tăng nhẹ (+0.20) vì hybrid mở rộng candidate pool, lấy được chunk có liên quan hơn ở một số câu.
- Tuy nhiên Relevance và Completeness đều giảm do BM25 đưa vào noise candidates không đúng topic.
- `gq02` là case điển hình: BM25 match keyword "VPN" và "thiết bị" từ nhiều nguồn khác nhau, nhưng không giữ được context chéo giữa HR Policy và IT Helpdesk FAQ.

**Kết luận Variant A:**
- Hybrid một mình chưa đủ — BM25 mở rộng candidate pool nhưng cần bước lọc thêm.

---

## Variant B (Hybrid + rerank)

**Ngày chạy:** 2026-04-13 17:50  
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
| Faithfulness | 4.80/5 | 4.90/5 | **+0.10** |
| Answer Relevance | 4.20/5 | 4.50/5 | **+0.30** |
| Context Recall | 5.00/5 | 5.00/5 | 0.00 |
| Completeness | 3.00/5 | 3.20/5 | **+0.20** |

**Câu cải thiện rõ nhất:**
- `gq01` (SLA P1): Completeness 3→4 — rerank đẩy đúng chunk version history lên top, model có thêm context để trả lời chi tiết hơn.
- `gq02` (VPN remote): Relevance 3→5 — rerank ưu tiên chunk có đầy đủ ngữ cảnh hơn, answer phục hồi hoàn toàn.
- `gq05` (Contractor Admin Access): Faithfulness 3→4, Relevance 4→5 — rerank cải thiện nhưng Completeness vẫn =1 vì pipeline vẫn lấy nhầm Section 4 (emergency 24h) thay vì Section 2 (Level 4 procedure 5 ngày).

**Câu không cải thiện:**
- `gq07` (Insufficient context): Relevance vẫn =1 — model abstain đúng nhưng không giải thích lý do thiếu thông tin. Rerank không giải quyết được lỗi ở tầng generation/prompt.

**Nhận xét:**
- Rerank (CrossEncoder `ms-marco-MiniLM-L-6-v2`) re-score lại toàn bộ candidates từ hybrid, đẩy chunk đúng nhất lên vị trí đầu trước khi build prompt.
- Relevance tăng +0.30, Faithfulness +0.10 — mức cải thiện rõ ràng nhất trong chain A→B.
- Context Recall giữ nguyên 5.00 — rerank không làm mất source, chỉ sắp xếp lại thứ tự.

**Kết luận Variant B:**
- Tốt nhất trong 3 cấu hình đã thử.
- Chốt cấu hình: `hybrid + rerank`.

---

## Tổng hợp để chốt cấu hình

| Cấu hình | Faithfulness | Relevance | Context Recall | Completeness |
|----------|-------------|-----------|----------------|-------------|
| Baseline (dense) | 4.60 | 4.50 | 5.00 | 3.30 |
| Variant A (hybrid) | 4.80 | 4.20 | 5.00 | 3.00 |
| **Variant B (hybrid + rerank)** | **4.90** | **4.50** | **5.00** | **3.20** |

**Kết luận cuối**
- `Variant B (hybrid + rerank)` dẫn đầu ở Faithfulness (+0.30 so với baseline) và Relevance (bằng baseline, +0.30 so với Variant A).
- Completeness của Variant B (3.20) thấp hơn baseline (3.30) một chút do `gq05` vẫn có C=1 ở Variant B (emergency path retrieval failure) kéo điểm xuống.
- Vì cả Variant A và Variant B đang để chung `label = "variant_hybrid"`, tiếp tục lưu file tách riêng như hiện tại (`*_variant_a*`, `*_variant_b*`) để tránh ghi đè kết quả.

---

## Grading Questions Run (gq01–gq10)

**Ngày chạy:** 2026-04-13 17:52–17:53  
**Config:** Variant B (hybrid + rerank) — cấu hình tốt nhất  
**File kết quả:** `logs/grading_run.json`

**Kết quả theo câu:**
| ID | Câu hỏi tóm tắt | Kết quả | Ghi chú |
|----|----------------|---------|---------|
| gq01 | SLA P1 thay đổi thế nào? | **Partial** | Đúng 4h/6h, thiếu version cũ v2025.3 |
| gq02 | Remote VPN + thiết bị? | **Partial** | Đúng 2 thiết bị, thiếu "Cisco AnyConnect" và HR Policy |
| gq03 | Flash Sale + kích hoạt → hoàn tiền? | **Partial** | Đúng kết luận, thiếu tham chiếu Điều 3 |
| gq04 | Store credit bao nhiêu %? | **Partial** | Đúng 110% nhưng thiếu "tùy chọn, không bắt buộc" |
| gq05 | Contractor + Admin Access? | **Zero** | Describe emergency 24h access thay vì Level 4 procedure 5 ngày |
| gq06 | P1 lúc 2am + cấp quyền tạm thời? | **Partial** | Đủ quy trình 24h, thiếu hotline ext. 9999 |
| gq07 | Phạt vi phạm SLA P1? | **Partial** | Abstain "Tôi không biết" — không hallucinate nhưng không giải thích lý do |
| gq08 | Nghỉ phép vs nghỉ ốm: "3 ngày"? | **Partial** | Phân biệt đúng 2 ngữ cảnh, thiếu HR Portal |
| gq09 | Mật khẩu đổi mấy ngày? | **Partial** | Đúng 90/7 ngày, thiếu kênh đổi (SSO portal / ext. 9000) |
| gq10 | Effective date 01/02/2026? | **Partial** | Kết luận rõ, thiếu effective date v4 |

**Phân tích 2 failure mode nổi bật:**

**gq05 — Retrieval nhầm section trong cùng document:**  
Pipeline retrieve đúng `access_control_sop.md` nhưng lấy Section 4 (emergency temp access, 24 giờ) thay vì Section 2 (Level 4 Admin procedure, 5 ngày + IT Manager + CISO). BM25 match mạnh trên "Admin" và "IT Admin" từ Section 4 vì keyword xuất hiện dày hơn. Kết quả: Completeness=1 ở Variant B dù Faithfulness=4, Relevance=5. Fix đề xuất: prepend section heading vào chunk text khi index, giúp CrossEncoder phân biệt ngữ cảnh section.

**gq07 — Abstain không đủ rõ ràng:**  
Model trả lời "Tôi không biết" — không hallucinate (tốt), nhưng không giải thích lý do (thiếu một câu như "Thông tin về mức phạt không có trong tài liệu hiện có"). Relevance=1 ở cả 3 cấu hình, cho thấy đây là lỗi ở tầng generation/prompt, không phải retrieval. Fix đề xuất: thêm vào grounded prompt: *"If you cannot answer, explicitly state that the information is not found in the provided documents."*

---

## Tóm tắt học được

1. Hybrid một mình chưa đủ; vẫn cần rerank để lọc noise trước khi generate.
2. Rerank không làm mất context recall, nhưng giúp tăng relevance (+0.30) và faithfulness (+0.10).
3. `gq05` (Contractor Admin Access) là câu phức tạp nhất: baseline hallucinate (F=1), Variant A vẫn sai (F=3), Variant B cải thiện faithfulness/relevance nhưng completeness=1 vì vẫn lấy nhầm section.
4. `gq07` (abstain) là failure mode cố định ở mọi cấu hình (R=1) — lỗi ở prompt generation, không phải retrieval.
5. Nếu có thêm thời gian: (a) prepend section heading vào chunk để fix gq05, (b) cải thiện abstain prompt để fix gq07.
