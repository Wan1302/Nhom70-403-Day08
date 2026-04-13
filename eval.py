"""
eval.py — Sprint 4: Evaluation & Scorecard
==========================================
Mục tiêu Sprint 4 (60 phút):
  - Chạy 10 test questions qua pipeline
  - Chấm điểm theo 4 metrics: Faithfulness, Relevance, Context Recall, Completeness
  - So sánh baseline vs variant
  - Ghi kết quả ra scorecard

Definition of Done Sprint 4:
  ✓ Demo chạy end-to-end (index → retrieve → answer → score)
  ✓ Scorecard trước và sau tuning
  ✓ A/B comparison: baseline vs variant với giải thích vì sao variant tốt hơn

A/B Rule (từ slide):
  Chỉ đổi MỘT biến mỗi lần để biết điều gì thực sự tạo ra cải thiện.
  Đổi đồng thời chunking + hybrid + rerank + prompt = không biết biến nào có tác dụng.
"""

import os
import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from rag_answer import rag_answer

# =============================================================================
# CẤU HÌNH
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "grading_questions.json"
GRADING_QUESTIONS_PATH = Path(__file__).parent / "data" / "grading_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"
LOGS_DIR = Path(__file__).parent / "logs"
GRADING_LOG_PATH = LOGS_DIR / "grading_run.json"

# Cấu hình baseline (Sprint 2)
BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "baseline_dense",
}

VARIANT_CONFIG_A = {
    "retrieval_mode": "hybrid",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "variant_hybrid",
}

VARIANT_CONFIG_B = {
    "retrieval_mode": "hybrid",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": True,
    "label": "variant_hybrid",
}

# =============================================================================
# SCORING FUNCTIONS
# 4 metrics từ slide: Faithfulness, Answer Relevance, Context Recall, Completeness
# =============================================================================

def score_faithfulness(
    answer: str,
    chunks_used: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Faithfulness: Câu trả lời có bám đúng chứng cứ đã retrieve không?
    Câu hỏi: Model có tự bịa thêm thông tin ngoài retrieved context không?

    Thang điểm 1-5:
      5: Mọi thông tin trong answer đều có trong retrieved chunks
      4: Gần như hoàn toàn grounded, 1 chi tiết nhỏ chưa chắc chắn
      3: Phần lớn grounded, một số thông tin có thể từ model knowledge
      2: Nhiều thông tin không có trong retrieved chunks
      1: Câu trả lời không grounded, phần lớn là model bịa

    TODO Sprint 4 — Có 2 cách chấm:

    Cách 1 — Chấm thủ công (Manual, đơn giản):
        Đọc answer và chunks_used, chấm điểm theo thang trên.
        Ghi lý do ngắn gọn vào "notes".

    Cách 2 — LLM-as-Judge (Tự động, nâng cao):
        Gửi prompt cho LLM:
            "Given these retrieved chunks: {chunks}
             And this answer: {answer}
             Rate the faithfulness on a scale of 1-5.
             5 = completely grounded in the provided context.
             1 = answer contains information not in the context.
             Output JSON: {'score': <int>, 'reason': '<string>'}"

    Trả về dict với: score (1-5) và notes (lý do)
    """
    answer_text = (answer or "").strip()
    answer_lower = answer_text.lower()
    abstain_markers = [
        "i do not know",
        "i don't know",
        "không biết",
        "khong biet",
        "không đủ dữ liệu",
        "khong du du lieu",
        "insufficient context",
    ]
    is_abstain = any(marker in answer_lower for marker in abstain_markers)

    if answer_text in ("", "PIPELINE_NOT_IMPLEMENTED") or answer_text.startswith("ERROR:"):
        return {
            "score": 1,
            "notes": "Pipeline error/empty answer -> cannot verify grounded evidence.",
        }

    if not chunks_used:
        if is_abstain:
            return {
                "score": 5,
                "notes": "No retrieved context and model abstained -> no hallucination.",
            }
        return {
            "score": 1,
            "notes": "No retrieved context but answer still asserted facts -> likely ungrounded.",
        }

    # Build compact context for judge prompt.
    context_parts = []
    for i, chunk in enumerate(chunks_used[:8], 1):
        meta = chunk.get("metadata", {}) or {}
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        text = (chunk.get("text", "") or "").strip()
        if len(text) > 700:
            text = text[:700] + "..."
        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        context_parts.append(f"{header}\n{text}")
    context_block = "\n\n".join(context_parts)

    # Chọn LLM-as-Judge để tự động chấm và tối ưu điểm bonus Sprint 4.
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY for LLM-as-Judge.")

        judge_model = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o")
        client = OpenAI(api_key=api_key)

        judge_prompt = f"""<role>
You are a strict evaluator for RAG faithfulness.
</role>

<task>
Rate only faithfulness: whether the answer is supported by retrieved context.
Do not score relevance, style, or completeness unless it affects grounding.
</task>

<rubric>
5 = fully grounded in context
4 = almost fully grounded, one minor uncertain detail
3 = mostly grounded, some possible model-added info
2 = many claims not supported by context
1 = mostly hallucinated / not grounded
</rubric>

<context>
{context_block}
</context>

<answer>
{answer_text}
</answer>

<constraints>
- Use only information in <context>.
- If answer adds unsupported facts, lower score.
- Keep reason short (max 25 words).
</constraints>

<output_format>
Return strict JSON only (no markdown, no code block):
{{"score": <integer 1-5>, "reason": "<short reason>"}}
</output_format>
"""

        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            max_tokens=220,
        )
        content = (response.choices[0].message.content or "").strip()
        # Parse robustly in case model wraps JSON in extra text/code fences.
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                raise
            parsed = json.loads(match.group(0))

        score = int(parsed.get("score"))
        score = max(1, min(5, score))
        reason = (parsed.get("reason") or "LLM judge").strip()
        return {
            "score": score,
            "notes": f"LLM-as-Judge ({judge_model}): {reason}",
        }
    except Exception as e:
        # Fallback heuristic để tránh crash pipeline nếu judge call lỗi.
        context_text = " ".join((c.get("text", "") or "") for c in chunks_used).lower()
        if is_abstain:
            return {
                "score": 5,
                "notes": f"Fallback abstain (safe): answer abstained. Judge error: {e}",
            }

        def _tokenize(t: str) -> List[str]:
            normalized = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in t)
            return [w for w in normalized.split() if len(w) >= 3]

        ans_tokens = set(_tokenize(answer_text))
        ctx_tokens = set(_tokenize(context_text))
        overlap = (len(ans_tokens & ctx_tokens) / len(ans_tokens)) if ans_tokens else 0.0

        if overlap >= 0.80:
            score = 5
        elif overlap >= 0.60:
            score = 4
        elif overlap >= 0.40:
            score = 3
        elif overlap >= 0.20:
            score = 2
        else:
            score = 1

        return {
            "score": score,
            "notes": f"Fallback heuristic overlap={overlap:.2f}. Judge error: {e}",
        }


def score_answer_relevance(
    query: str,
    answer: str,
) -> Dict[str, Any]:
    """
    Answer Relevance: Answer có trả lời đúng câu hỏi người dùng hỏi không?
    Câu hỏi: Model có bị lạc đề hay trả lời đúng vấn đề cốt lõi không?

    Thang điểm 1-5:
      5: Answer trả lời trực tiếp và đầy đủ câu hỏi
      4: Trả lời đúng nhưng thiếu vài chi tiết phụ
      3: Trả lời có liên quan nhưng chưa đúng trọng tâm
      2: Trả lời lạc đề một phần
      1: Không trả lời câu hỏi

    TODO Sprint 4: Implement tương tự score_faithfulness
    """
    query_text = (query or "").strip()
    answer_text = (answer or "").strip()
    answer_lower = answer_text.lower()

    abstain_markers = [
        "i do not know",
        "i don't know",
        "không biết",
        "khong biet",
        "không đủ dữ liệu",
        "khong du du lieu",
        "insufficient context",
    ]
    is_abstain = any(marker in answer_lower for marker in abstain_markers)

    if answer_text in ("", "PIPELINE_NOT_IMPLEMENTED") or answer_text.startswith("ERROR:"):
        return {
            "score": 1,
            "notes": "Pipeline error/empty answer -> not relevant to question.",
        }

    # LLM-as-Judge for relevance.
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY for LLM-as-Judge.")

        judge_model = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o")
        client = OpenAI(api_key=api_key)

        judge_prompt = f"""<role>
You are a strict evaluator for answer relevance.
</role>

<task>
Rate how directly and correctly the answer addresses the user question.
</task>

<rubric>
5 = directly and fully answers the question
4 = correct answer, minor missing detail
3 = related but misses core focus
2 = partially off-topic
1 = does not answer the question
</rubric>

<question>
{query_text}
</question>

<answer>
{answer_text}
</answer>

<constraints>
- Evaluate relevance only (not faithfulness, not style).
- Keep reason short (max 25 words).
</constraints>

<output_format>
Return strict JSON only (no markdown, no code block):
{{"score": <integer 1-5>, "reason": "<short reason>"}}
</output_format>
"""

        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            max_tokens=220,
        )
        content = (response.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                raise
            parsed = json.loads(match.group(0))

        score = int(parsed.get("score"))
        score = max(1, min(5, score))
        reason = (parsed.get("reason") or "LLM judge").strip()
        return {
            "score": score,
            "notes": f"LLM-as-Judge ({judge_model}): {reason}",
        }
    except Exception as e:
        # Fallback heuristic
        if is_abstain:
            # Abstain có thể relevant nếu query thiếu dữ liệu.
            # Chấm trung bình-khá để tránh ưu ái quá mức khi chưa có judge.
            return {
                "score": 3,
                "notes": f"Fallback abstain relevance=3. Judge error: {e}",
            }

        def _tokenize(t: str) -> List[str]:
            normalized = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in t)
            return [w for w in normalized.split() if len(w) >= 3]

        q_tokens = set(_tokenize(query_text))
        a_tokens = set(_tokenize(answer_text))
        overlap = (len(q_tokens & a_tokens) / len(q_tokens)) if q_tokens else 0.0

        if overlap >= 0.80:
            score = 5
        elif overlap >= 0.60:
            score = 4
        elif overlap >= 0.40:
            score = 3
        elif overlap >= 0.20:
            score = 2
        else:
            score = 1

        return {
            "score": score,
            "notes": f"Fallback heuristic overlap={overlap:.2f}. Judge error: {e}",
        }


def score_context_recall(
    chunks_used: List[Dict[str, Any]],
    expected_sources: List[str],
) -> Dict[str, Any]:
    """
    Context Recall: Retriever có mang về đủ evidence cần thiết không?
    Câu hỏi: Expected source có nằm trong retrieved chunks không?

    Đây là metric đo retrieval quality, không phải generation quality.

    Cách tính đơn giản:
        recall = (số expected source được retrieve) / (tổng số expected sources)

    Ví dụ:
        expected_sources = ["policy/refund-v4.pdf", "sla-p1-2026.pdf"]
        retrieved_sources = ["policy/refund-v4.pdf", "helpdesk-faq.md"]
        recall = 1/2 = 0.5

    TODO Sprint 4:
    1. Lấy danh sách source từ chunks_used
    2. Kiểm tra xem expected_sources có trong retrieved sources không
    3. Tính recall score
    """
    if not expected_sources:
        # Câu hỏi không có expected source (ví dụ: "Không đủ dữ liệu" cases)
        return {"score": None, "recall": None, "notes": "No expected sources"}

    retrieved_sources = {
        c.get("metadata", {}).get("source", "")
        for c in chunks_used
    }

    # TODO: Kiểm tra matching theo partial path (vì source paths có thể khác format)
    found = 0
    missing = []
    for expected in expected_sources:
        # Kiểm tra partial match (tên file)
        expected_name = expected.split("/")[-1].replace(".pdf", "").replace(".md", "")
        matched = any(expected_name.lower() in r.lower() for r in retrieved_sources)
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall = found / len(expected_sources) if expected_sources else 0

    return {
        "score": round(recall * 5),  # Convert to 1-5 scale
        "recall": recall,
        "found": found,
        "missing": missing,
        "notes": f"Retrieved: {found}/{len(expected_sources)} expected sources" +
                 (f". Missing: {missing}" if missing else ""),
    }


def score_completeness(
    query: str,
    answer: str,
    expected_answer: str,
) -> Dict[str, Any]:
    """
    Completeness: Answer có thiếu điều kiện ngoại lệ hoặc bước quan trọng không?
    Câu hỏi: Answer có bao phủ đủ thông tin so với expected_answer không?

    Thang điểm 1-5:
      5: Answer bao gồm đủ tất cả điểm quan trọng trong expected_answer
      4: Thiếu 1 chi tiết nhỏ
      3: Thiếu một số thông tin quan trọng
      2: Thiếu nhiều thông tin quan trọng
      1: Thiếu phần lớn nội dung cốt lõi

    TODO Sprint 4:
    Option 1 — Chấm thủ công: So sánh answer vs expected_answer và chấm.
    Option 2 — LLM-as-Judge:
        "Compare the model answer with the expected answer.
         Rate completeness 1-5. Are all key points covered?
         Output: {'score': int, 'missing_points': [str]}"
    """
    query_text = (query or "").strip()
    answer_text = (answer or "").strip()
    expected_text = (expected_answer or "").strip()

    if answer_text in ("", "PIPELINE_NOT_IMPLEMENTED") or answer_text.startswith("ERROR:"):
        return {
            "score": 1,
            "notes": "Pipeline error/empty answer -> completeness is very low.",
        }

    if not expected_text:
        return {
            "score": None,
            "notes": "No expected_answer provided.",
        }

    # Option 2: LLM-as-Judge
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY for LLM-as-Judge.")

        judge_model = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o")
        client = OpenAI(api_key=api_key)

        judge_prompt = f"""<role>
You are a strict evaluator for answer completeness.
</role>

<task>
Compare model answer with expected answer and score completeness from 1 to 5.
</task>

<rubric>
5 = includes all key points in expected answer
4 = missing one minor detail
3 = missing some important information
2 = missing many important points
1 = misses most core content
</rubric>

<question>
{query_text}
</question>

<expected_answer>
{expected_text}
</expected_answer>

<model_answer>
{answer_text}
</model_answer>

<constraints>
- Evaluate completeness only (not faithfulness, not style).
- Identify concrete missing points vs expected answer.
- Keep reason short (max 25 words).
</constraints>

<output_format>
Return strict JSON only (no markdown, no code block):
{{"score": <integer 1-5>, "reason": "<short reason>", "missing_points": ["..."]}}
</output_format>
"""

        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            max_tokens=260,
        )
        content = (response.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                raise
            parsed = json.loads(match.group(0))

        score = int(parsed.get("score"))
        score = max(1, min(5, score))
        reason = (parsed.get("reason") or "LLM judge").strip()
        missing_points = parsed.get("missing_points", [])
        if not isinstance(missing_points, list):
            missing_points = [str(missing_points)]
        missing_points = [str(x).strip() for x in missing_points if str(x).strip()][:3]

        notes = f"LLM-as-Judge ({judge_model}): {reason}"
        if missing_points:
            notes += f" | missing: {missing_points}"

        return {
            "score": score,
            "notes": notes,
        }
    except Exception as e:
        # Fallback heuristic: token coverage of expected answer by model answer.
        def _tokenize(t: str) -> List[str]:
            normalized = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in t)
            return [w for w in normalized.split() if len(w) >= 3]

        exp_tokens = set(_tokenize(expected_text))
        ans_tokens = set(_tokenize(answer_text))
        coverage = (len(exp_tokens & ans_tokens) / len(exp_tokens)) if exp_tokens else 0.0

        if coverage >= 0.85:
            score = 5
        elif coverage >= 0.70:
            score = 4
        elif coverage >= 0.50:
            score = 3
        elif coverage >= 0.30:
            score = 2
        else:
            score = 1

        missing_tokens = sorted(list(exp_tokens - ans_tokens))[:6]
        return {
            "score": score,
            "notes": (
                f"Fallback heuristic coverage={coverage:.2f}. "
                f"Missing token hints={missing_tokens}. Judge error: {e}"
            ),
        }


# =============================================================================
# SCORECARD RUNNER
# =============================================================================

def run_scorecard(
    config: Dict[str, Any],
    test_questions: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chạy toàn bộ test questions qua pipeline và chấm điểm.

    Args:
        config: Pipeline config (retrieval_mode, top_k, use_rerank, ...)
        test_questions: List câu hỏi (load từ JSON nếu None)
        verbose: In kết quả từng câu

    Returns:
        List scorecard results, mỗi item là một row

    TODO Sprint 4:
    1. Load test_questions từ data/test_questions.json
    2. Với mỗi câu hỏi:
       a. Gọi rag_answer() với config tương ứng
       b. Chấm 4 metrics
       c. Lưu kết quả
    3. Tính average scores
    4. In bảng kết quả
    """
    if test_questions is None:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)

    results = []
    label = config.get("label", "unnamed")

    print(f"\n{'='*70}")
    print(f"Chạy scorecard: {label}")
    print(f"Config: {config}")
    print('='*70)

    for q in test_questions:
        question_id = q["id"]
        query = q["question"]
        expected_answer = q.get("expected_answer", "")
        expected_sources = q.get("expected_sources", [])
        category = q.get("category", "")

        if verbose:
            print(f"\n[{question_id}] {query}")

        # --- Gọi pipeline ---
        try:
            result = rag_answer(
                query=query,
                retrieval_mode=config.get("retrieval_mode", "dense"),
                top_k_search=config.get("top_k_search", 10),
                top_k_select=config.get("top_k_select", 3),
                use_rerank=config.get("use_rerank", False),
                verbose=False,
            )
            answer = result["answer"]
            chunks_used = result["chunks_used"]

        except NotImplementedError:
            answer = "PIPELINE_NOT_IMPLEMENTED"
            chunks_used = []
        except Exception as e:
            answer = f"ERROR: {e}"
            chunks_used = []

        # --- Chấm điểm ---
        faith = score_faithfulness(answer, chunks_used)
        relevance = score_answer_relevance(query, answer)
        recall = score_context_recall(chunks_used, expected_sources)
        complete = score_completeness(query, answer, expected_answer)

        row = {
            "id": question_id,
            "category": category,
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            "faithfulness": faith["score"],
            "faithfulness_notes": faith["notes"],
            "relevance": relevance["score"],
            "relevance_notes": relevance["notes"],
            "context_recall": recall["score"],
            "context_recall_notes": recall["notes"],
            "completeness": complete["score"],
            "completeness_notes": complete["notes"],
            "config_label": label,
        }
        results.append(row)

        if verbose:
            print(f"  Answer: {answer[:100]}...")
            print(f"  Faithful: {faith['score']} | Relevant: {relevance['score']} | "
                  f"Recall: {recall['score']} | Complete: {complete['score']}")

    # Tính averages (bỏ qua None)
    for metric in ["faithfulness", "relevance", "context_recall", "completeness"]:
        scores = [r[metric] for r in results if r[metric] is not None]
        avg = sum(scores) / len(scores) if scores else None
        print(f"\nAverage {metric}: {avg:.2f}" if avg else f"\nAverage {metric}: N/A (chưa chấm)")

    return results


# =============================================================================
# GRADING RUN LOGGER
# =============================================================================

def run_grading_questions_log(
    questions_path: Path = GRADING_QUESTIONS_PATH,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Chạy bộ grading questions theo config được chỉ định và lưu log JSON.
    Mặc định dùng VARIANT_CONFIG_B (hybrid + rerank) nếu không truyền config.
    """
    if config is None:
        config = VARIANT_CONFIG_B
    if output_path is None:
        label = config.get("label", "run")
        output_path = LOGS_DIR / f"grading_run_{label}.json"

    if not questions_path.exists():
        print(f"Không tìm thấy grading questions: {questions_path}")
        return []

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    log = []
    for q in questions:
        result = rag_answer(
            q["question"],
            retrieval_mode=config.get("retrieval_mode", "hybrid"),
            top_k_search=config.get("top_k_search", 10),
            top_k_select=config.get("top_k_select", 3),
            use_rerank=config.get("use_rerank", False),
            verbose=False,
        )
        log.append({
            "id": q["id"],
            "question": q["question"],
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks_retrieved": len(result["chunks_used"]),
            "retrieval_mode": result["config"]["retrieval_mode"],
            "timestamp": datetime.now().isoformat(),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"Đã ghi grading log: {output_path} ({len(log)} câu)")
    return log


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_ab(
    baseline_results: List[Dict],
    variant_results: List[Dict],
    output_csv: Optional[str] = None,
    title: str = "Baseline vs Variant",
) -> None:
    """
    So sánh baseline vs variant theo từng câu hỏi và tổng thể.

    TODO Sprint 4:
    Điền vào bảng sau để trình bày trong báo cáo:

    | Metric          | Baseline | Variant | Delta |
    |-----------------|----------|---------|-------|
    | Faithfulness    |   ?/5    |   ?/5   |  +/?  |
    | Answer Relevance|   ?/5    |   ?/5   |  +/?  |
    | Context Recall  |   ?/5    |   ?/5   |  +/?  |
    | Completeness    |   ?/5    |   ?/5   |  +/?  |

    Câu hỏi cần trả lời:
    - Variant tốt hơn baseline ở câu nào? Vì sao?
    - Biến nào (chunking / hybrid / rerank) đóng góp nhiều nhất?
    - Có câu nào variant lại kém hơn baseline không? Tại sao?
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]

    print(f"\n{'='*70}")
    print(f"A/B Comparison: {title}")
    print('='*70)
    print(f"{'Metric':<20} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
    print("-" * 55)

    for metric in metrics:
        b_scores = [r[metric] for r in baseline_results if r[metric] is not None]
        v_scores = [r[metric] for r in variant_results if r[metric] is not None]

        b_avg = sum(b_scores) / len(b_scores) if b_scores else None
        v_avg = sum(v_scores) / len(v_scores) if v_scores else None
        delta = (v_avg - b_avg) if (b_avg is not None and v_avg is not None) else None

        b_str = f"{b_avg:.2f}" if b_avg is not None else "N/A"
        v_str = f"{v_avg:.2f}" if v_avg is not None else "N/A"
        d_str = f"{delta:+.2f}" if delta is not None else "N/A"

        print(f"{metric:<20} {b_str:>10} {v_str:>10} {d_str:>8}")

    # Per-question comparison
    print(f"\n{'Câu':<6} {'Baseline F/R/Rc/C':<22} {'Variant F/R/Rc/C':<22} {'Better?':<10}")
    print("-" * 65)

    b_by_id = {r["id"]: r for r in baseline_results}
    for v_row in variant_results:
        qid = v_row["id"]
        b_row = b_by_id.get(qid, {})

        b_scores_str = "/".join([
            str(b_row.get(m, "?")) for m in metrics
        ])
        v_scores_str = "/".join([
            str(v_row.get(m, "?")) for m in metrics
        ])

        # So sánh đơn giản
        b_total = sum(b_row.get(m, 0) or 0 for m in metrics)
        v_total = sum(v_row.get(m, 0) or 0 for m in metrics)
        better = "Variant" if v_total > b_total else ("Baseline" if b_total > v_total else "Tie")

        print(f"{qid:<6} {b_scores_str:<22} {v_scores_str:<22} {better:<10}")

    # Export to CSV
    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / output_csv
        combined = baseline_results + variant_results
        if combined:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=combined[0].keys())
                writer.writeheader()
                writer.writerows(combined)
            print(f"\nKết quả đã lưu vào: {csv_path}")


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_scorecard_summary(results: List[Dict], label: str) -> str:
    """
    Tạo báo cáo tóm tắt scorecard dạng markdown.

    TODO Sprint 4: Cập nhật template này theo kết quả thực tế của nhóm.
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    averages = {}
    for metric in metrics:
        scores = [r[metric] for r in results if r[metric] is not None]
        averages[metric] = sum(scores) / len(scores) if scores else None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"""# Scorecard: {label}
Generated: {timestamp}

## Summary

| Metric | Average Score |
|--------|--------------|
"""
    for metric, avg in averages.items():
        avg_str = f"{avg:.2f}/5" if avg else "N/A"
        md += f"| {metric.replace('_', ' ').title()} | {avg_str} |\n"

    md += "\n## Per-Question Results\n\n"
    md += "| ID | Category | Faithful | Relevant | Recall | Complete | Notes |\n"
    md += "|----|----------|----------|----------|--------|----------|-------|\n"

    for r in results:
        md += (f"| {r['id']} | {r['category']} | {r.get('faithfulness', 'N/A')} | "
               f"{r.get('relevance', 'N/A')} | {r.get('context_recall', 'N/A')} | "
               f"{r.get('completeness', 'N/A')} | {r.get('faithfulness_notes', '')} |\n")

    return md


# =============================================================================
# MAIN — Chạy evaluation
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 4: Evaluation & Scorecard")
    print("=" * 60)

    # Load test questions một lần dùng chung cho cả 3 bản
    print(f"\nLoading test questions từ: {TEST_QUESTIONS_PATH}")
    try:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)
        print(f"Tìm thấy {len(test_questions)} câu hỏi")
        for q in test_questions[:3]:
            print(f"  [{q['id']}] {q['question']} ({q['category']})")
        print("  ...")
    except FileNotFoundError:
        print("Không tìm thấy file test_questions.json!")
        test_questions = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Chạy 3 bản: baseline, variant_a, variant_b
    # =========================================================
    print("\n--- [1/3] Chạy Baseline (dense) ---")
    baseline_results = run_scorecard(BASELINE_CONFIG, test_questions, verbose=True)
    baseline_md = generate_scorecard_summary(baseline_results, BASELINE_CONFIG["label"])
    (RESULTS_DIR / "scorecard_baseline.md").write_text(baseline_md, encoding="utf-8")

    print("\n--- [2/3] Chạy Variant A (hybrid, không rerank) ---")
    variant_a_results = run_scorecard(VARIANT_CONFIG_A, test_questions, verbose=True)
    variant_a_md = generate_scorecard_summary(variant_a_results, VARIANT_CONFIG_A["label"])
    (RESULTS_DIR / "scorecard_variant_a.md").write_text(variant_a_md, encoding="utf-8")

    run_grading_questions_log(
        config=VARIANT_CONFIG_A,
        output_path=LOGS_DIR / "grading_run_variant_a.json",
    )

    print("\n--- [3/3] Chạy Variant B (hybrid + rerank) ---")
    variant_b_results = run_scorecard(VARIANT_CONFIG_B, test_questions, verbose=True)
    variant_b_md = generate_scorecard_summary(variant_b_results, VARIANT_CONFIG_B["label"])
    (RESULTS_DIR / "scorecard_variant_b.md").write_text(variant_b_md, encoding="utf-8")

    run_grading_questions_log(
        config=VARIANT_CONFIG_B,
        output_path=LOGS_DIR / "grading_run_variant_b.json",
    )

    # =========================================================
    # So sánh theo chuỗi: baseline -> A -> B
    # =========================================================
    print("\n--- Compare 1: Baseline vs Variant A ---")
    compare_ab(
        baseline_results,
        variant_a_results,
        output_csv="ab_comparison_baseline_vs_a.csv",
        title="Baseline (dense) vs Variant A (hybrid)",
    )

    print("\n--- Compare 2: Variant A vs Variant B ---")
    compare_ab(
        variant_a_results,
        variant_b_results,
        output_csv="ab_comparison_a_vs_b.csv",
        title="Variant A (hybrid) vs Variant B (hybrid + rerank)",
    )

    # =========================================================
    # TỔNG KẾT
    # =========================================================
    print("\n" + "=" * 60)
    print("Hoàn tất! Kết quả lưu tại:")
    print(f"  results/scorecard_baseline.md")
    print(f"  results/scorecard_variant_a.md")
    print(f"  results/scorecard_variant_b.md")
    print(f"  results/ab_comparison_baseline_vs_a.csv  (dense vs hybrid)")
    print(f"  results/ab_comparison_a_vs_b.csv         (hybrid vs hybrid+rerank)")
    print(f"  logs/grading_run_variant_a.json")
    print(f"  logs/grading_run_variant_b.json")
    print("=" * 60)
