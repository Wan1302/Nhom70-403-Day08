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

Kết quả thực tế cho thấy abstain vẫn chưa hoàn hảo: ở q09 (ERR-403-AUTH không có trong docs), baseline vượt ngưỡng 0.25 vì vector gần với chunk về Access Control, dẫn đến LLM hallucinate câu trả lời (`Faithfulness=4`, `Completeness=2`). Hybrid mode lại đi quá hướng ngược — abstain hoàn toàn, chỉ trả về "Tôi không biết" mà không có hướng dẫn nào (`Relevance=1`). Cả hai đều sai theo cách khác nhau, và lỗi nằm ở mức retrieval chứ không phải generation.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** *"Contractor từ bên ngoài công ty có thể được cấp quyền Admin Access không? Nếu có, cần bao nhiêu ngày và có yêu cầu đặc biệt gì?"* (gq05 — category: Access Control, difficulty: hard)

**Phân tích:**

Pipeline trả lời: *"Có, contractor có thể được cấp Admin Access tạm thời trong trường hợp khẩn cấp, tối đa 24 giờ sau khi được Tech Lead phê duyệt bằng lời..."*

Câu trả lời đúng là: Level 4 Admin Access theo quy trình tiêu chuẩn — cần phê duyệt từ IT Manager + CISO, thời gian xử lý 5 ngày làm việc, bắt buộc training security policy.

Lỗi nằm ở **retrieval**: hybrid lấy nhầm chunk về *emergency temporary access* (24 giờ, Tech Lead phê duyệt bằng lời) thay vì chunk về *Level 4 Admin Access* (5 ngày, IT Manager + CISO). Cả hai chunk đều từ cùng file `access-control-sop.md`, đều chứa keyword "contractor" và "Admin" — cross-encoder rerank cũng không phân biệt được vì cả hai chunk đều có độ liên quan cao với câu hỏi.

Root cause: pipeline không có cơ chế phân biệt section trong cùng một document. Fix cụ thể: thêm metadata filter theo `section` trong retrieval, hoặc expand query thành "Level 4 Admin Access" để bias về đúng section.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Có hai cải tiến cụ thể: Thứ nhất, thêm section-level metadata filter vào `retrieve_dense` và `retrieve_hybrid` — gq05 cho thấy pipeline lấy nhầm section trong cùng một document dù rerank đã bật, và cách fix trực tiếp nhất là filter theo `section` metadata khi query có từ khóa định danh rõ ràng ("Level 4", "Admin Access"). Thứ hai, sửa abstain message trong `rag_answer` từ "Tôi không biết" thành nêu rõ lý do — gq07 chỉ đạt 5/10 vì abstain mơ hồ, trong khi thêm một câu *"Thông tin này không có trong tài liệu nội bộ hiện có"* sẽ đủ điều kiện Full marks theo rubric.

---

*Lưu file này với tên: `reports/individual/Ho_Tran_Dinh_Nguyen.md`*
