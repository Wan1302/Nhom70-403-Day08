# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Hồ Trần Đình Nguyên  
**Vai trò trong nhóm:** Tech Lead  
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Tôi đảm nhận vai trò Tech Lead và viết toàn bộ `rag_answer.py` cho Sprint 2 và Sprint 3. Cụ thể, tôi implement ba hàm retrieval: `retrieve_dense` dùng embedding OpenAI + ChromaDB, `retrieve_sparse` dùng BM25 (rank-bm25), và `retrieve_hybrid` kết hợp hai phương pháp qua Reciprocal Rank Fusion. Ngoài ra, tôi implement `rerank` bằng cross-encoder (`ms-marco-MiniLM-L-6-v2`), `transform_query` với ba chiến lược expansion/decomposition/HyDE gọi LLM để sinh variant query, và hàm `rag_answer` là pipeline tổng hợp toàn bộ.

Cấu hình pipeline: `TOP_K_SEARCH=10`, `TOP_K_SELECT=3`, `ABSTAIN_SCORE_THRESHOLD=0.25`, LLM mặc định `gpt-4o-mini`. Phần của tôi kết nối với `index.py` qua import `get_embedding` và `CHROMA_DB_DIR` để đảm bảo query embedding dùng cùng model với lúc indexing.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Điều tôi hiểu rõ nhất là **tại sao RRF tốt hơn cộng điểm trực tiếp** khi kết hợp dense và sparse. Dense dùng cosine similarity (0–1), BM25 trả về điểm tuyệt đối phụ thuộc corpus — hai scale hoàn toàn khác nhau, cộng thẳng sẽ khiến một bên lấn át bên kia. RRF chỉ dùng **thứ hạng**: `score = weight × (1 / (60 + rank))`. Hằng số 60 làm phẳng khoảng cách giữa rank 1 và rank 5, tránh overfit vào một kết quả đứng đầu ở một hệ thống nhưng vắng mặt ở hệ thống kia.

Tuy nhiên, qua kết quả thực tế ở q06: hybrid lấy nhầm chunk về Access Control thay vì SLA, khiến Completeness từ 4 (baseline) xuống 1 (hybrid). Điều này cho thấy RRF không phải lúc nào cũng tốt hơn dense — nếu sparse BM25 kéo lên một chunk sai nhưng có keyword khớp, RRF sẽ bị ảnh hưởng.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều khó nhất là thiết kế **abstain logic**. Tôi đặt `ABSTAIN_SCORE_THRESHOLD=0.25` và chỉ áp dụng cho `dense` mode vì hybrid/sparse dùng RRF score (~0.009) — không thể so sánh với cùng ngưỡng cosine similarity.

Kết quả thực tế cho thấy abstain vẫn chưa hoàn hảo: ở q09 (ERR-403-AUTH không có trong docs), baseline vượt ngưỡng 0.25 vì vector gần với chunk về Access Control (có chứa "403"), dẫn đến LLM hallucinate câu trả lời (`Faithfulness=4`, `Completeness=2`). Hybrid mode lại đi quá hướng ngược — abstain hoàn toàn, chỉ trả về "Tôi không biết" mà không có hướng dẫn nào (`Relevance=1`). Cả hai đều sai theo cách khác nhau, và lỗi nằm ở mức retrieval chứ không phải generation.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** *"Sản phẩm kỹ thuật số có được hoàn tiền không?"* (q04 — category: Refund, difficulty: medium)

**Phân tích:**

Đây là câu hỏi mà retrieval hoạt động đúng (Context Recall=5 ở cả baseline và hybrid — chunk từ `policy/refund-v4.pdf` được lấy về đầy đủ), nhưng **generation thất bại**. Baseline cho `Faithfulness=1` vì LLM trả lời: *"không được hoàn tiền, trừ khi có lỗi do nhà sản xuất"* — thông tin về ngoại lệ nhà sản xuất không có trong context. Hybrid cũng tương tự, `Faithfulness=2`.

Lỗi nằm hoàn toàn ở **generation**: LLM thêm thông tin không có trong docs dù prompt đã ép "Answer only from the retrieved context". Đây là hallucination điển hình khi model có prior knowledge về refund policy thực tế và "tự điền" vào.

Cả baseline lẫn hybrid đều thất bại ở câu này với cùng lý do — thay đổi retrieval strategy không giải quyết được vấn đề generation. Fix cần nhắm vào prompt: thêm ràng buộc cứng hơn, ví dụ *"Do NOT add any exceptions or conditions not explicitly stated in the context."*

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Có hai cải tiến cụ thể: Thứ nhất, thêm negative constraint vào `build_grounded_prompt` vì eval cho thấy q04 hallucinate dù retrieval đúng — prompt hiện tại chưa đủ mạnh để ngăn LLM thêm prior knowledge. Thứ hai, tích hợp `transform_query` với strategy `"expansion"` vào `rag_answer` cho q07 (Approval Matrix alias) — cả baseline lẫn hybrid đều cho Completeness=3 vì không nhận ra tên mới "Access Control SOP", query expansion có thể giải quyết trực tiếp vấn đề này.

---

*Lưu file này với tên: `reports/individual/Ho_Tran_Dinh_Nguyen.md`*
