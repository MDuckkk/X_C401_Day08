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
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import OpenAI
from rag_answer import rag_answer

# =============================================================================
# CẤU HÌNH
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "test_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "baseline_dense",
}

VARIANT_CONFIG = {
    "retrieval_mode": "hybrid",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": True,
    "label": "variant_hybrid_rerank",
}


def _get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def score_faithfulness(
    answer: str,
    chunks_used: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if answer in ("PIPELINE_NOT_IMPLEMENTED",) or answer.startswith("ERROR:"):
        return {"score": None, "notes": "Pipeline error"}

    context = "\n\n".join([c.get("text", "") for c in chunks_used])

    prompt = f"""You are an evaluation judge. Rate the faithfulness of an answer given the retrieved context.

Faithfulness means: every claim in the answer is supported by the provided context. The answer does not fabricate information.

Scale 1-5:
5: Every claim is fully supported by the context
4: Nearly all grounded, one minor uncertain detail
3: Mostly grounded, some information may come from outside the context
2: Many claims not supported by the context
1: Answer is largely fabricated, not grounded in context

Retrieved context:
{context[:3000]}

Answer:
{answer}

Output ONLY valid JSON: {{"score": <int 1-5>, "reason": "<brief explanation>"}}"""

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        result = json.loads(response.choices[0].message.content.strip())
        return {"score": result["score"], "notes": result.get("reason", "")}
    except Exception:
        return {"score": None, "notes": "LLM judge error"}


def score_answer_relevance(
    query: str,
    answer: str,
) -> Dict[str, Any]:
    if answer in ("PIPELINE_NOT_IMPLEMENTED",) or answer.startswith("ERROR:"):
        return {"score": None, "notes": "Pipeline error"}

    prompt = f"""You are an evaluation judge. Rate how relevant the answer is to the user's question.

Scale 1-5:
5: Directly and fully answers the question
4: Answers correctly but misses minor details
3: Related but doesn't address the core question
2: Partially off-topic
1: Does not answer the question at all

Question: {query}

Answer: {answer}

Output ONLY valid JSON: {{"score": <int 1-5>, "reason": "<brief explanation>"}}"""

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        result = json.loads(response.choices[0].message.content.strip())
        return {"score": result["score"], "notes": result.get("reason", "")}
    except Exception:
        return {"score": None, "notes": "LLM judge error"}


def score_context_recall(
    chunks_used: List[Dict[str, Any]],
    expected_sources: List[str],
) -> Dict[str, Any]:
    if not expected_sources:
        return {"score": None, "recall": None, "notes": "No expected sources"}

    retrieved_sources = {
        c.get("metadata", {}).get("source", "")
        for c in chunks_used
    }

    found = 0
    missing = []
    for expected in expected_sources:
        expected_name = expected.split("/")[-1].replace(".pdf", "").replace(".md", "").replace(".txt", "")
        matched = any(expected_name.lower() in r.lower() for r in retrieved_sources)
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall = found / len(expected_sources)

    return {
        "score": round(recall * 5),
        "recall": recall,
        "found": found,
        "missing": missing,
        "notes": f"Retrieved: {found}/{len(expected_sources)} expected sources"
                 + (f". Missing: {missing}" if missing else ""),
    }


def score_completeness(
    query: str,
    answer: str,
    expected_answer: str,
) -> Dict[str, Any]:
    if answer in ("PIPELINE_NOT_IMPLEMENTED",) or answer.startswith("ERROR:"):
        return {"score": None, "notes": "Pipeline error"}

    if not expected_answer:
        return {"score": None, "notes": "No expected answer provided"}

    prompt = f"""You are an evaluation judge. Compare the model's answer against the expected answer and rate completeness.

Scale 1-5:
5: Covers all key points from the expected answer
4: Misses one minor detail
3: Misses some important information
2: Misses many important points
1: Misses most core content

Question: {query}

Expected answer: {expected_answer}

Model answer: {answer}

Output ONLY valid JSON: {{"score": <int 1-5>, "missing_points": ["<point1>", ...], "reason": "<brief explanation>"}}"""

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        result = json.loads(response.choices[0].message.content.strip())
        return {
            "score": result["score"],
            "notes": result.get("reason", "")
                     + (f" Missing: {result['missing_points']}" if result.get("missing_points") else ""),
        }
    except Exception:
        return {"score": None, "notes": "LLM judge error"}


# =============================================================================
# SCORECARD RUNNER
# =============================================================================

def run_scorecard(
    config: Dict[str, Any],
    test_questions: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
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
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]

    print(f"\n{'='*70}")
    print("A/B Comparison: Baseline vs Variant")
    print('='*70)
    print(f"{'Metric':<20} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
    print("-" * 55)

    summary_data = {}
    for metric in metrics:
        b_scores = [r[metric] for r in baseline_results if r[metric] is not None]
        v_scores = [r[metric] for r in variant_results if r[metric] is not None]

        b_avg = sum(b_scores) / len(b_scores) if b_scores else None
        v_avg = sum(v_scores) / len(v_scores) if v_scores else None
        delta = (v_avg - b_avg) if (b_avg is not None and v_avg is not None) else None

        summary_data[metric] = {"baseline": b_avg, "variant": v_avg, "delta": delta}

        b_str = f"{b_avg:.2f}" if b_avg is not None else "N/A"
        v_str = f"{v_avg:.2f}" if v_avg is not None else "N/A"
        d_str = f"{delta:+.2f}" if delta is not None else "N/A"

        print(f"{metric:<20} {b_str:>10} {v_str:>10} {d_str:>8}")

    # Per-question comparison
    print(f"\n{'Câu':<6} {'Baseline F/R/Rc/C':<22} {'Variant F/R/Rc/C':<22} {'Better?':<10}")
    print("-" * 65)

    b_by_id = {r["id"]: r for r in baseline_results}
    per_question_rows = []

    for v_row in variant_results:
        qid = v_row["id"]
        b_row = b_by_id.get(qid, {})

        b_scores_str = "/".join([str(b_row.get(m, "?")) for m in metrics])
        v_scores_str = "/".join([str(v_row.get(m, "?")) for m in metrics])

        b_total = sum(b_row.get(m, 0) or 0 for m in metrics)
        v_total = sum(v_row.get(m, 0) or 0 for m in metrics)
        better = "Variant" if v_total > b_total else ("Baseline" if b_total > v_total else "Tie")

        print(f"{qid:<6} {b_scores_str:<22} {v_scores_str:<22} {better:<10}")

        per_question_rows.append({
            "id": qid,
            "query": v_row.get("query", ""),
            "baseline_answer": b_row.get("answer", ""),
            "variant_answer": v_row.get("answer", ""),
            **{f"baseline_{m}": b_row.get(m, "") for m in metrics},
            **{f"variant_{m}": v_row.get(m, "") for m in metrics},
            **{f"delta_{m}": (v_row.get(m, 0) or 0) - (b_row.get(m, 0) or 0) for m in metrics},
            "better": better,
        })

    # --- LLM analysis ---
    b_label = baseline_results[0].get("config_label", "baseline") if baseline_results else "baseline"
    v_label = variant_results[0].get("config_label", "variant") if variant_results else "variant"

    summary_text = "\n".join([
        f"{m}: baseline={d['baseline']:.2f}, variant={d['variant']:.2f}, delta={d['delta']:+.2f}"
        for m, d in summary_data.items() if d["delta"] is not None
    ])

    per_q_text = "\n".join([
        f"Q{r['id']}: baseline={r['baseline_faithfulness']}/{r['baseline_relevance']}/{r['baseline_context_recall']}/{r['baseline_completeness']} "
        f"variant={r['variant_faithfulness']}/{r['variant_relevance']}/{r['variant_context_recall']}/{r['variant_completeness']} → {r['better']}"
        for r in per_question_rows
    ])

    analysis_prompt = f"""You are a RAG evaluation analyst. Analyze the A/B comparison results and provide a clear explanation.

Baseline config: {b_label}
Variant config: {v_label}

Overall metrics (scale 1-5):
{summary_text}

Per-question results (Faithfulness/Relevance/ContextRecall/Completeness):
{per_q_text}

Write a concise analysis in Vietnamese covering:
1. Tóm tắt: variant tốt hơn hay kém hơn baseline? Ở metric nào?
2. Phân tích câu hỏi: câu nào variant thắng, câu nào thua, vì sao?
3. Giải thích biến đã thay đổi (hybrid/rerank/query transform) đóng góp gì?
4. Kết luận: nên dùng config nào cho production?

Keep it under 300 words. Be specific with numbers."""

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0,
            max_tokens=600,
        )
        analysis = response.choices[0].message.content.strip()
    except Exception:
        analysis = "Không thể tạo phân tích tự động."

    print(f"\n{'='*70}")
    print("ANALYSIS")
    print('='*70)
    print(analysis)

    # Export to CSV
    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / output_csv

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(["=== SUMMARY ==="])
            writer.writerow(["Metric", "Baseline", "Variant", "Delta"])
            for m, d in summary_data.items():
                writer.writerow([
                    m,
                    f"{d['baseline']:.2f}" if d["baseline"] is not None else "N/A",
                    f"{d['variant']:.2f}" if d["variant"] is not None else "N/A",
                    f"{d['delta']:+.2f}" if d["delta"] is not None else "N/A",
                ])
            writer.writerow([])

            writer.writerow(["=== PER-QUESTION ==="])
            if per_question_rows:
                headers = list(per_question_rows[0].keys())
                writer.writerow(headers)
                for row in per_question_rows:
                    writer.writerow([row[h] for h in headers])
            writer.writerow([])

            writer.writerow(["=== ANALYSIS ==="])
            writer.writerow(["explanation"])
            for line in analysis.split("\n"):
                writer.writerow([line])

        print(f"\nKết quả đã lưu vào: {csv_path}")


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_scorecard_summary(results: List[Dict], label: str) -> str:
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
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 4: Evaluation & Scorecard")
    print("=" * 60)

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

    # --- Chạy Baseline ---
    print("\n--- Chạy Baseline ---")
    try:
        baseline_results = run_scorecard(
            config=BASELINE_CONFIG,
            test_questions=test_questions,
            verbose=True,
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        baseline_md = generate_scorecard_summary(baseline_results, "baseline_dense")
        (RESULTS_DIR / "scorecard_baseline.md").write_text(baseline_md, encoding="utf-8")
    except NotImplementedError:
        print("Pipeline chưa implement.")
        baseline_results = []

    # --- Chạy Variant ---
    print("\n--- Chạy Variant ---")
    try:
        variant_results = run_scorecard(
            config=VARIANT_CONFIG,
            test_questions=test_questions,
            verbose=True,
        )
        variant_md = generate_scorecard_summary(variant_results, VARIANT_CONFIG["label"])
        (RESULTS_DIR / "scorecard_variant.md").write_text(variant_md, encoding="utf-8")
    except NotImplementedError:
        print("Variant chưa implement.")
        variant_results = []

    # --- A/B Comparison ---
    if baseline_results and variant_results:
        compare_ab(
            baseline_results,
            variant_results,
            output_csv="ab_comparison.csv",
        )