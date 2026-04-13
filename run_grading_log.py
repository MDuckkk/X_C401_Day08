import json
from datetime import datetime
from pathlib import Path
from rag_answer import rag_answer

GRADING_QUESTIONS_PATH = 'data/test_questions.json'
LOGS_DIR = Path(__file__).parent / "logs"


def save_log(all_logs, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, ensure_ascii=False, indent=2)


def run_all():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOGS_DIR / "grading_run.json"

    # Load existing progress if interrupted before
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            all_logs = json.load(f)
        print(f"Resumed: found {len(all_logs)} existing entries")
    else:
        all_logs = []

    with open(GRADING_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    configs = [
        {"retrieval_mode": "hybrid", "use_rerank": True},
        {"retrieval_mode": "hybrid", "use_rerank": False},
        {"retrieval_mode": "dense", "use_rerank": False},
        {"retrieval_mode": "dense", "use_rerank": True},
        {"retrieval_mode": "sparse", "use_rerank": False},
        {"retrieval_mode": "sparse", "use_rerank": True},
    ]

    # Figure out where we left off
    done = len(all_logs)
    total = len(configs) * len(questions)

    if done >= total:
        print(f"Already complete: {done}/{total} entries. Delete grading_run.json to re-run.")
        return

    idx = 0
    for cfg in configs:
        for q in questions:
            if idx < done:
                idx += 1
                continue

            mode = cfg["retrieval_mode"]
            rerank = cfg["use_rerank"]
            print(f"[{idx+1}/{total}] {mode}|rerank={rerank} | [{q['id']}] {q['question']}")

            try:
                result = rag_answer(
                    q["question"],
                    retrieval_mode=mode,
                    use_rerank=rerank,
                    verbose=False,
                )
                all_logs.append({
                    "id": q["id"],
                    "question": q["question"],
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "chunks_retrieved": len(result["chunks_used"]),
                    "retrieval_mode": result["config"]["retrieval_mode"],
                    "use_rerank": result["config"]["use_rerank"],
                    "timestamp": datetime.now().isoformat(),
                })
                print(f"  → {result['answer'][:100]}...")
            except Exception as e:
                all_logs.append({
                    "id": q["id"],
                    "question": q["question"],
                    "answer": f"ERROR: {e}",
                    "sources": [],
                    "chunks_retrieved": 0,
                    "retrieval_mode": mode,
                    "use_rerank": rerank,
                    "timestamp": datetime.now().isoformat(),
                })
                print(f"  → ERROR: {e}")

            save_log(all_logs, output_path)
            idx += 1

    print(f"\nDone! {len(all_logs)} total entries in {output_path}")


if __name__ == "__main__":
    run_all()