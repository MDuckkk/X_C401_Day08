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

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from rag_answer import rag_answer, call_llm
import os

# =============================================================================
# CẤU HÌNH
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "test_questions.json"
GRADING_QUESTIONS_PATH = Path(__file__).parent / "data" / "grading_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Điểm từng câu grading (theo SCORING.md)
GRADING_POINTS = {
    "gq01": 10, "gq02": 10, "gq03": 10, "gq04": 8,  "gq05": 10,
    "gq06": 12, "gq07": 10, "gq08": 10, "gq09": 8,  "gq10": 10,
}
GRADING_TOTAL_RAW = 98  # Tổng điểm raw tối đa

# Cấu hình baseline (Sprint 2)
BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "baseline_dense",
}

# Cấu hình variant (Sprint 3 — điều chỉnh theo lựa chọn của nhóm)
# TODO Sprint 4: Cập nhật VARIANT_CONFIG theo variant nhóm đã implement
VARIANT_CONFIG = {
    "retrieval_mode": "hybrid",   # Biến duy nhất thay đổi so với baseline
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,          # Giữ nguyên như baseline — A/B rule: chỉ đổi 1 biến
    "label": "variant_hybrid",
}


# =============================================================================
# SCORING retrieval_mode 
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
    # LLM-as-Judge
    context = "\n".join([c.get("text", "") for c in chunks_used])
    prompt = f"""Given these retrieved chunks: {context}
And this answer: {answer}

Rate the faithfulness on a scale of 1-5.

IMPORTANT RULES:
- If the answer says "I don't know" / "Tôi không biết" / "Không đủ dữ liệu" / "Không có thông tin", this is PERFECTLY FAITHFUL (score = 5) because it does NOT hallucinate.
- 5 = completely grounded in the provided context OR explicitly states lack of information
- 4 = mostly grounded, one minor uncertain detail
- 3 = partially grounded, some information may be from model knowledge
- 2 = many details not in the context
- 1 = answer fabricates information not in the context (hallucination)

Output JSON format only: {{"score": <int>, "notes": "<string reason>"}}"""

    try:
        response = call_llm(prompt)
        res_json = json.loads(response.replace("```json", "").replace("```", "").strip())
        if "score" not in res_json: res_json["score"] = 3
        if "notes" not in res_json: res_json["notes"] = "No notes"
        return res_json
    except Exception as e:
        return {"score": 3, "notes": f"Lỗi chấm LLM: {str(e)}"}


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
    prompt = f"""Given the user question: "{query}"
And this answer: "{answer}"

Rate the answer relevance on a scale of 1-5.

IMPORTANT RULES:
- If the answer says "I don't know" / "Tôi không biết" / "Không đủ dữ liệu" when the question cannot be answered from available data, this is RELEVANT (score = 5) because it directly addresses the question by stating the limitation.
- 5 = Answer explicitly answers the question directly (including stating "I don't know" when appropriate)
- 4 = Answers correctly but missing minor details
- 3 = Related but not focused on the core issue
- 2 = Partially off-topic
- 1 = Completely irrelevant to the question

Output JSON format only: {{"score": <int>, "notes": "<string reason>"}}"""

    try:
        response = call_llm(prompt)
        res_json = json.loads(response.replace("```json", "").replace("```", "").strip())
        if "score" not in res_json: res_json["score"] = 3
        if "notes" not in res_json: res_json["notes"] = "No notes"
        return res_json
    except Exception as e:
        return {"score": 3, "notes": f"Lỗi chấm LLM: {str(e)}"}


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
    prompt = f"""Compare the model answer with the expected answer.
Question: {query}
Model Answer: {answer}
Expected Answer: {expected_answer}

Rate completeness 1-5. Are all key points from Expected Answer covered in Model Answer?

IMPORTANT RULES:
- If Expected Answer says "Không tìm thấy thông tin" / "không đề cập" and Model Answer says "Tôi không biết" / "Không đủ dữ liệu", they MATCH (score = 5).
- If both answers indicate lack of information, this is COMPLETE (score = 5).
- 5 = Fully covered (including both stating lack of information)
- 4 = Missing one minor detail
- 3 = Missing some important information
- 2 = Missing many important points
- 1 = Completely missed core content

Output JSON format only: {{"score": <int>, "notes": "<string reason>"}}"""

    try:
        response = call_llm(prompt)
        res_json = json.loads(response.replace("```json", "").replace("```", "").strip())
        if "score" not in res_json: res_json["score"] = 3
        if "notes" not in res_json: res_json["notes"] = "No notes"
        return res_json
    except Exception as e:
        return {"score": 3, "notes": f"Lỗi chấm LLM: {str(e)}"}


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
# A/B COMPARISON
# =============================================================================

def compare_ab(
    baseline_results: List[Dict],
    variant_results: List[Dict],
    output_csv: Optional[str] = None,
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
    print("A/B Comparison: Baseline vs Variant")
    print('='*70)
    print(f"{'Metric':<20} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
    print("-" * 55)

    for metric in metrics:
        b_scores = [r[metric] for r in baseline_results if r[metric] is not None]
        v_scores = [r[metric] for r in variant_results if r[metric] is not None]

        b_avg = sum(b_scores) / len(b_scores) if b_scores else None
        v_avg = sum(v_scores) / len(v_scores) if v_scores else None
        delta = (v_avg - b_avg) if (b_avg and v_avg) else None

        b_str = f"{b_avg:.2f}" if b_avg else "N/A"
        v_str = f"{v_avg:.2f}" if v_avg else "N/A"
        d_str = f"{delta:+.2f}" if delta else "N/A"

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
               f"{r.get('completeness', 'N/A')} | {r.get('faithfulness_notes', '')[:50]} |\n")

    return md


# =============================================================================
# GRADING LOG GENERATOR
# =============================================================================

def generate_grading_log(
    config: Dict[str, Any],
    output_filename: str = "grading_run.json"
) -> None:
    """
    Tạo file JSON grading log từ grading_questions.json để nộp bài đúng chuẩn SCORING.md
    """
    log = []
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(GRADING_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            grading_questions = json.load(f)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy {GRADING_QUESTIONS_PATH}. File này được public lúc 17:00!")
        return

    print(f"\n{'='*70}")
    print(f"Tạo Grading Log với config: {config['label']}")
    print(f"Số câu grading: {len(grading_questions)}")
    print(f"{'='*70}")

    best_mode = config.get("retrieval_mode", "dense")
    use_rerank = config.get("use_rerank", False)

    for i, q in enumerate(grading_questions):
        print(f"[{i+1}/{len(grading_questions)}] {q['id']}: {q['question'][:60]}...")
        try:
            result = rag_answer(
                query=q["question"],
                retrieval_mode=best_mode,
                use_rerank=use_rerank,
                verbose=False
            )
            log.append({
                "id": q["id"],
                "question": q["question"],
                "answer": result["answer"],
                "sources": result["sources"],
                "chunks_retrieved": len(result["chunks_used"]),
                "retrieval_mode": result.get("config", {}).get("retrieval_mode", best_mode),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            print(f"  Lỗi câu {q['id']}: {e}")
            log.append({
                "id": q["id"],
                "question": q["question"],
                "answer": f"PIPELINE_ERROR: {str(e)}",
                "sources": [],
                "chunks_retrieved": 0,
                "retrieval_mode": best_mode,
                "timestamp": datetime.now().isoformat()
            })

    output_path = log_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Đã lưu Grading Log tại: {output_path}")


def print_scoring_estimate(results: List[Dict], label: str = "") -> None:
    """
    Ước tính điểm theo rubric SCORING.md dựa trên 4 metrics.
    Quy tắc: Full (avg >= 4) = 100%, Partial (avg >= 2.5) = 50%, Zero = 0%
    Áp dụng cho scorecard chạy trên test_questions (không phải grading chính thức).
    """
    print(f"\n{'='*70}")
    print(f"Ước tính điểm SCORING.md — {label}")
    print(f"(Dựa trên test_questions, không phải grading chính thức)")
    print('='*70)

    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    total_raw_estimate = 0.0
    penalty = 0.0

    print(f"\n{'ID':<6} {'F':>4} {'R':>4} {'Rc':>4} {'C':>4} {'Avg':>6} {'Mức':>10}")
    print("-" * 45)

    for r in results:
        scores = [r[m] for m in metrics if r.get(m) is not None]
        avg = sum(scores) / len(scores) if scores else 0

        if avg >= 4.0:
            level = "Full"
        elif avg >= 2.5:
            level = "Partial"
        else:
            level = "Zero"

        print(f"{r['id']:<6} "
              f"{str(r.get('faithfulness','?')):>4} "
              f"{str(r.get('relevance','?')):>4} "
              f"{str(r.get('context_recall','?')):>4} "
              f"{str(r.get('completeness','?')):>4} "
              f"{avg:>6.2f} {level:>10}")

        # Faithfulness thấp (< 2) → có thể hallucinate → penalty
        faith_score = r.get("faithfulness", 3)
        if faith_score is not None and faith_score < 2:
            penalty += 0.5  # Ước tính -50% điểm câu đó
            print(f"       ⚠️  Faithfulness thấp — nguy cơ hallucination penalty!")

    # Tính tổng ước tính (dùng số câu test thay cho grading)
    n = len(results)
    avg_per_q = GRADING_TOTAL_RAW / 10  # Trung bình điểm mỗi câu grading
    for r in results:
        scores = [r[m] for m in metrics if r.get(m) is not None]
        avg = sum(scores) / len(scores) if scores else 0
        if avg >= 4.0:
            total_raw_estimate += avg_per_q
        elif avg >= 2.5:
            total_raw_estimate += avg_per_q * 0.5

    total_raw_estimate -= penalty * avg_per_q
    scoring_30 = max(0, (total_raw_estimate / GRADING_TOTAL_RAW) * 30)

    print(f"\n{'─'*45}")
    print(f"Ước tính raw score : {total_raw_estimate:.1f} / {GRADING_TOTAL_RAW}")
    print(f"Ước tính điểm nhóm (Grading Questions 30đ): {scoring_30:.1f} / 30")
    print(f"⚠️  Đây chỉ là ước tính từ test_questions — điểm thực tế từ grading_questions.json")


# =============================================================================
# MAIN — Chạy evaluation
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 4: Evaluation & Scorecard")
    print("=" * 60)

    # --- Load test_questions (dùng cho scorecard) ---
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

    # --- Chạy Baseline Scorecard (dùng test_questions) ---
    print("\n--- Chạy Baseline Scorecard (test_questions) ---")
    try:
        baseline_results = run_scorecard(
            config=BASELINE_CONFIG,
            test_questions=test_questions,
            verbose=True,
        )
        baseline_md = generate_scorecard_summary(baseline_results, "baseline_dense")
        (RESULTS_DIR / "scorecard_baseline.md").write_text(baseline_md, encoding="utf-8")
        print(f"\nScorecard lưu tại: {RESULTS_DIR / 'scorecard_baseline.md'}")
        print_scoring_estimate(baseline_results, label="Baseline")
    except NotImplementedError:
        print("Pipeline chưa implement. Hoàn thành Sprint 2 trước.")
        baseline_results = []

    # --- Chạy Variant Scorecard (dùng test_questions) ---
    print("\n--- Chạy Variant Scorecard (test_questions) ---")
    try:
        variant_results = run_scorecard(
            config=VARIANT_CONFIG,
            test_questions=test_questions,
            verbose=True,
        )
        variant_md = generate_scorecard_summary(variant_results, VARIANT_CONFIG["label"])
        (RESULTS_DIR / "scorecard_variant.md").write_text(variant_md, encoding="utf-8")
        print(f"\nScorecard lưu tại: {RESULTS_DIR / 'scorecard_variant.md'}")
        print_scoring_estimate(variant_results, label="Variant")
    except NotImplementedError:
        print("Variant chưa implement. Hoàn thành Sprint 3 trước.")
        variant_results = []

    # --- A/B Comparison ---
    if baseline_results and variant_results:
        compare_ab(
            baseline_results,
            variant_results,
            output_csv="ab_comparison.csv"
        )

    # --- Grading Log (dùng grading_questions.json — public lúc 17:00) ---
    print("\n--- Ghi Grading Log (grading_questions.json) ---")
    generate_grading_log(
        config=BASELINE_CONFIG,
        output_filename="grading_run.json"
    )

    print("\n\nChecklist Sprint 4:")
    print("  ✓ scorecard_baseline.md và scorecard_variant.md → từ test_questions.json")
    print("  ✓ logs/grading_run.json → từ grading_questions.json (public lúc 17:00)")
    print("  → Nếu grading_questions.json chưa có, chạy lại sau 17:00 để tạo grading log")
