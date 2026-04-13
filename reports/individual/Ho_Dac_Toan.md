# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Hồ Đắc Toàn - 2A202600057 
**Vai trò trong nhóm:** Eval Owner  
**Ngày nộp:** 2026-04-13  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này?

Tôi phụ trách Sprint 4 — toàn bộ phần evaluation và tài liệu kỹ thuật.

Về code, tôi implement file `eval.py` với bốn scoring function sử dụng LLM-as-Judge (`gpt-4o`): `score_faithfulness`, `score_answer_relevance`, `score_context_recall`, `score_completeness`. Mỗi hàm gọi OpenAI với prompt structured có rubric 1–5, fallback heuristic nếu API lỗi. Tôi cũng viết `run_scorecard()` để chạy toàn bộ 10 câu hỏi qua pipeline, `compare_ab()` để so sánh delta giữa các cấu hình, và `generate_scorecard_summary()` để xuất markdown.

Quan trọng hơn, tôi thiết kế lại luồng `__main__` để chạy **3 bản** (baseline, variant_a, variant_b) và so sánh theo chuỗi: baseline → Variant A (Δ hybrid) rồi Variant A → Variant B (Δ rerank), thay vì so sánh mỗi variant với baseline riêng lẻ. Cách này đảm bảo đúng nguyên tắc A/B: mỗi lần chỉ thay một biến.

Sau khi có số liệu, tôi điền `docs/architecture.md` với thông số thực tế và `docs/tuning-log.md` với phân tích từng lần chạy, rồi viết `reports/group_report.md` tổng hợp kết quả cho nhóm.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

Trước lab, tôi hiểu hybrid retrieval theo nghĩa "kết hợp dense và sparse nên sẽ tốt hơn". Sau khi chạy thực tế, tôi nhận ra điều đó **không tự động đúng**.

Kết quả Variant A (hybrid không rerank) cho Relevance 4.10 và Completeness 3.60 — thấp hơn baseline dense (4.50 và 3.90). Nguyên nhân: BM25 match keyword "IT Admin" từ `access_control_sop.txt` vào câu hỏi về SLA escalation, đưa sai chunk lên candidate pool. Dense search một mình không bị lỗi này vì nó search theo semantic similarity, không theo keyword.

Điều tôi học được: hybrid mở rộng recall nhưng cũng mở rộng noise. Rerank là bước bắt buộc để lọc lại — không phải tùy chọn. Variant B (hybrid + rerank) đẩy Relevance lên 4.80 và Completeness lên 4.00, chứng minh CrossEncoder re-score lại đúng chunk trước khi đưa vào prompt.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

Điều ngạc nhiên nhất là LLM-as-Judge không nhất quán hoàn toàn giữa các lần chạy. Cùng một câu hỏi, cùng một answer, nhưng chạy hai lần có thể cho điểm lệch nhau 1 điểm ở Faithfulness hay Completeness. Ban đầu tôi nghĩ `temperature=0` đảm bảo deterministic — nhưng không, vì model vẫn có variability nhỏ ở tầng sampling.

Hệ quả thực tế: baseline của Experiment A và Experiment B (chạy cùng config `dense`) cho số liệu hơi khác nhau (Relevance 4.70 vs 4.50). Điều này buộc tôi phải chạy baseline mới cho mỗi experiment thay vì tái sử dụng kết quả cũ — và là lý do tôi thiết kế `__main__` chạy 3 bản đầy đủ mỗi lần thay vì cache kết quả baseline.

Khó khăn kỹ thuật khác: parse JSON output từ judge model. Model đôi khi wrap JSON trong markdown code block hoặc thêm text trước, khiến `json.loads()` crash. Tôi phải thêm regex fallback `re.search(r"\{[\s\S]*\}", content)` để extract JSON an toàn.

---

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** `gq05` — *"Contractor từ bên ngoài công ty có thể được cấp quyền Admin Access không? Nếu có, cần bao nhiêu ngày và có yêu cầu đặc biệt gì?"*

**Phân tích:**

Pipeline trả lời:
> *"Có, contractor có thể được cấp quyền Admin Access tạm thời trong trường hợp khẩn cấp, tối đa 24 giờ sau khi được Tech Lead phê duyệt..."*

Câu trả lời mắc lỗi **retrieval nghiêm trọng**: pipeline retrieve đúng `access_control_sop.md` nhưng lấy nhầm **Section 4 (emergency temporary access)** thay vì **Section 2 (Level 4 Admin Access procedure)**.

Lý do: câu hỏi có hai keyword mạnh — "contractor" và "Admin Access". BM25 match "Admin" và "IT Admin" từ Section 4 (emergency escalation) vì đoạn đó xuất hiện nhiều lần hơn trong corpus. Chunk Level 4 Admin Access nằm sâu hơn trong document và ít keyword overlap hơn. Kết quả là model nhận được context về quy trình khẩn cấp 24 giờ, trả lời sai hoàn toàn so với expected answer (quy trình thông thường 5 ngày, approver IT Manager + CISO, yêu cầu training bắt buộc).

**Root cause:** Lỗi ở **retrieval + chunking**. Section 4 và Section 2 của access control doc đều có liên quan đến cấp quyền, nhưng bị lấy nhầm. Nếu chunking dùng heading-based tốt hơn (giữ nguyên section header trong mỗi chunk), CrossEncoder có thể phân biệt được "Emergency access" vs "Level 4 standard procedure" qua heading.

**Đề xuất fix:** Thêm section title vào đầu mỗi chunk khi index (metadata `section` đã có, nhưng không được prepend vào chunk text). Khi rerank có cả "Section 2: Access Level Matrix" và "Section 4: Emergency Access" trong text, CrossEncoder sẽ score đúng hơn với câu hỏi về "Admin Access" quy trình thường.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

**Cải tiến 1 — Sửa q04 (digital products refund):** Faithfulness=1–2 ở cả 3 cấu hình vì LLM tự thêm exception "lỗi nhà sản xuất" không có trong context. Tôi sẽ thêm một câu explicit vào grounded prompt: *"Do not infer exceptions or conditions not stated in the context."* Evidence: q04 là câu duy nhất có Faithfulness <3 ở mọi cấu hình, chứng tỏ đây là lỗi prompt chứ không phải lỗi retrieval.

**Cải tiến 2 — Tune abstain cho q09 (ERR-403-AUTH):** Variant A trả lời sai hoàn toàn (Relevance=1) vì hybrid lấy chunk access control rồi model tự suy diễn quy trình xử lý lỗi. Tôi sẽ thêm rule: *"If the question asks about a specific error code not mentioned in any source, explicitly state that the information is not available."* Evidence từ scorecard: q09 Completeness=2 ở mọi cấu hình, cho thấy abstain logic hiện tại chưa đủ rõ để model áp dụng nhất quán.
