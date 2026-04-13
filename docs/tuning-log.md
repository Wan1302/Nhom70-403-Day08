# Tuning Log - RAG Pipeline (Day 08 Lab)

> A/B Rule: moi vong chi doi 1 bien de biet bien nao tao ra cai thien that su.

---

## Baseline (Sprint 2)

**Ngay:** 2026-04-13  
**Config:**
```python
retrieval_mode = "dense"
chunk_size = 400 tokens
overlap = 80 tokens
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = "gpt-4o-mini"
```

**Scorecard Baseline (lan so sanh voi Variant A):**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 4.50 /5 |
| Answer Relevance | 4.60 /5 |
| Context Recall | 5.00 /5 |
| Completeness | 3.80 /5 |

**Cau hoi yeu nhat (diem thap):**
- `q04` (Refund digital product): faithfulness thap do answer them ngoai le khong co trong context.
- `q09` (ERR-403-AUTH): completeness thap vi khong co du lieu trong corpus de tra loi day du.
- `q10` (VIP refund): relevance thap do cau tra loi abstain qua chung chung.

**Gia thuyet nguyen nhan (Error Tree):**
- [ ] Indexing: Chunking cat giua dieu khoan
- [ ] Indexing: Metadata thieu effective_date
- [x] Retrieval: Dense bo lo keyword/intent dac thu
- [ ] Retrieval: Top-k qua it -> thieu evidence
- [x] Generation: Prompt abstain chua toi uu cho cau hoi "khong du du lieu"
- [ ] Generation: Context qua dai -> lost in the middle

---

## Variant 1 (Sprint 3) - Variant A

**Ngay:** 2026-04-13  
**Bien thay doi:** `retrieval_mode: dense -> hybrid` (giu nguyen rerank=False)

**Ly do chon bien nay:**
Baseline co nhom cau hoi pha tron natural language + keyword/ma loi. Hybrid du kien giup lay du context on dinh hon dense thuần.

**Config thay doi:**
```python
retrieval_mode = "hybrid"
use_rerank = False
# Cac tham so con lai giu nguyen
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 4.50/5 | 4.60/5 | +0.10 |
| Answer Relevance | 4.60/5 | 4.20/5 | -0.40 |
| Context Recall | 5.00/5 | 5.00/5 | +0.00 |
| Completeness | 3.80/5 | 3.50/5 | -0.30 |

**Nhan xet:**
- Cai thien nhe o faithfulness.
- Giam ro o relevance/completeness, giam manh o `q06` va `q09`.
- `q10` tot hon baseline (relevance tang), nhung tong the van kem baseline.

**Ket luan Variant 1:**
Variant A chua tot hon baseline o tong the. Hybrid khong rerank co dau hieu lay du context nhung chua loc duoc chunk nhiễu.

---

## Variant 2 (Sprint 3) - Variant B

**Ngay:** 2026-04-13  
**Bien thay doi:** `use_rerank: False -> True` (tren nen Hybrid)

**Config:**
```python
retrieval_mode = "hybrid"
top_k_search = 10
top_k_select = 3
use_rerank = True
```

**Scorecard Variant 2 (so voi baseline o cung lan chay):**
| Metric | Baseline | Variant 2 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 4.60/5 | 4.70/5 | +0.10 |
| Answer Relevance | 4.60/5 | 4.80/5 | +0.20 |
| Context Recall | 5.00/5 | 5.00/5 | +0.00 |
| Completeness | 3.70/5 | 4.10/5 | +0.40 |

**Nhan xet:**
- Cai thien ro o `q06`, `q08`, `q10`; `q09` cung on dinh hon.
- Context recall giu nguyen 5.00, cho thay rerank chu yeu giup quality cua context dua vao prompt.

---

## Tong hop 3 cau hinh

| Metric | Baseline | Variant 1 (Hybrid) | Variant 2 (Hybrid + Rerank) | Best |
|--------|----------|--------------------|-----------------------------|------|
| Faithfulness | 4.60 | 4.60 | 4.70 | Variant 2 |
| Answer Relevance | 4.60 | 4.20 | 4.80 | Variant 2 |
| Context Recall | 5.00 | 5.00 | 5.00 | Tie |
| Completeness | 3.70 | 3.50 | 4.10 | Variant 2 |

> Ghi chu: baseline trong hai lan chay co dao dong nho (faithfulness/completeness), do LLM-as-Judge va generation co tinh xac suat.

---

## Tom tat hoc duoc

1. **Loi pho bien nhat trong pipeline nay la gi?**
   Retrieval khong phai luc nao cung la diem nghẽn; van de lon la context nao duoc dua vao prompt sau cung. Rerank giup giam noise ro ret.

2. **Bien nao co tac dong lon nhat toi chat luong?**
   Bat `use_rerank=True` tren nen Hybrid cho tac dong lon nhat (dac biet completeness +0.40).

3. **Neu co them 1 gio, nhom se thu gi tiep theo?**
   Tinh chinh prompt abstain + bo sung rule citation bat buoc theo source/section de cai thien tiep q04, q09, q10.
