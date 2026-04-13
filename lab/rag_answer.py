"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
Sprint 2 (60 phút): Baseline RAG
  - Dense retrieval từ ChromaDB
  - Grounded answer function với prompt ép citation
  - Trả lời được ít nhất 3 câu hỏi mẫu, output có source

Sprint 3 (60 phút): Tuning tối thiểu
  - Thêm hybrid retrieval (dense + sparse/BM25)
  - Hoặc thêm rerank (cross-encoder)
  - Hoặc thử query transformation (expansion, decomposition, HyDE)
  - Tạo bảng so sánh baseline vs variant

Definition of Done Sprint 2:
  ✓ rag_answer("SLA ticket P1?") trả về câu trả lời có citation
  ✓ rag_answer("Câu hỏi không có trong docs") trả về "Không đủ dữ liệu"

Definition of Done Sprint 3:
  ✓ Có ít nhất 1 variant (hybrid / rerank / query transform) chạy được
  ✓ Giải thích được tại sao chọn biến đó để tune
"""

import os
import re
import sys
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10    # Số chunk lấy từ vector store trước rerank (search rộng)
TOP_K_SELECT = 3     # Số chunk gửi vào prompt sau rerank/select (top-3 sweet spot)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_MAX_TOKENS = 512

try:
    from index import CHROMA_DB_DIR, DOCS_DIR, get_embedding, preprocess_document, chunk_document
except Exception:
    CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"
    DOCS_DIR = Path(__file__).parent / "data" / "docs"
    get_embedding = None
    preprocess_document = None
    chunk_document = None

_ALL_CHUNKS_CACHE: Optional[List[Dict[str, Any]]] = None
_BM25_CACHE = None
_CROSS_ENCODER_MODEL = None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ỹ0-9]+", text.lower())


def _keyword_overlap_score(query: str, text: str) -> float:
    query_tokens = set(_tokenize(query))
    text_tokens = set(_tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = query_tokens & text_tokens
    return len(overlap) / len(query_tokens)


def _boost_score_for_exact_terms(query: str, text: str) -> float:
    score = 0.0
    lowered_query = query.lower()
    lowered_text = text.lower()

    for term in re.findall(r"[A-Za-z0-9\-]{2,}", lowered_query):
        if term in lowered_text:
            if any(ch.isdigit() for ch in term) or "-" in term or term.isupper():
                score += 0.25
            else:
                score += 0.08
    return score


def _lexical_score(query: str, text: str) -> float:
    return _keyword_overlap_score(query, text) + _boost_score_for_exact_terms(query, text)


def _load_all_chunks_from_docs() -> List[Dict[str, Any]]:
    global _ALL_CHUNKS_CACHE
    if _ALL_CHUNKS_CACHE is not None:
        return _ALL_CHUNKS_CACHE

    chunks: List[Dict[str, Any]] = []
    if preprocess_document is None or chunk_document is None:
        return chunks

    for filepath in sorted(DOCS_DIR.glob("*.txt")):
        raw_text = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw_text, str(filepath))
        doc_chunks = chunk_document(doc)
        for chunk in doc_chunks:
            meta = chunk.setdefault("metadata", {})
            meta.setdefault("source", filepath.name)
        chunks.extend(doc_chunks)

    _ALL_CHUNKS_CACHE = chunks
    return chunks


def _collection_available() -> Tuple[Optional[Any], Optional[str]]:
    try:
        import chromadb
    except ModuleNotFoundError:
        return None, "Thiếu package 'chromadb'"

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = client.get_collection("rag_lab")
    except Exception:
        return None, "Chưa có collection 'rag_lab' trong ChromaDB"
    return collection, None


def _fallback_retrieve_lexical(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    all_chunks = _load_all_chunks_from_docs()
    scored = []
    for chunk in all_chunks:
        text = chunk.get("text", "")
        score = _lexical_score(query, text)
        if score <= 0:
            continue
        scored.append({
            "text": text,
            "metadata": chunk.get("metadata", {}),
            "score": score,
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _is_context_sufficient(query: str, candidates: List[Dict[str, Any]]) -> bool:
    if not candidates:
        return False

    top = candidates[0]
    top_text = top.get("text", "")
    lexical = _keyword_overlap_score(query, top_text)
    score = float(top.get("score", 0) or 0)

    if lexical >= 0.25:
        return True
    if score >= 0.55:
        return True
    if any(token in top_text.lower() for token in _tokenize(query) if len(token) >= 4):
        return True
    return False


def _extract_candidate_sentences(query: str, chunks: List[Dict[str, Any]]) -> List[Tuple[str, int, float]]:
    query_tokens = set(_tokenize(query))
    results: List[Tuple[str, int, float]] = []

    for idx, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "")
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[\.\?!])\s+|\n+", text)
            if sentence.strip()
        ]
        for sentence in sentences:
            overlap = len(query_tokens & set(_tokenize(sentence)))
            score = overlap + _boost_score_for_exact_terms(query, sentence)
            if any(ch.isdigit() for ch in sentence):
                score += 0.5
            results.append((sentence, idx, score))

    results.sort(key=lambda item: item[2], reverse=True)
    return results


def _fallback_generate_answer(query: str, chunks: List[Dict[str, Any]]) -> str:
    if not _is_context_sufficient(query, chunks):
        return "Không đủ dữ liệu trong tài liệu hiện có để trả lời câu hỏi này."

    candidate_sentences = _extract_candidate_sentences(query, chunks)
    selected: List[Tuple[str, int]] = []
    seen = set()

    for sentence, ref_idx, score in candidate_sentences:
        normalized = _normalize_text(sentence)
        if len(normalized) < 20:
            continue
        if normalized.lower() in seen:
            continue
        selected.append((normalized, ref_idx))
        seen.add(normalized.lower())
        if len(selected) >= 2:
            break

    if not selected:
        return "Không đủ dữ liệu trong tài liệu hiện có để trả lời câu hỏi này."

    parts = [f"{sentence} [{ref_idx}]" for sentence, ref_idx in selected]
    return " ".join(parts)


def _get_transformed_queries(query: str, retrieval_mode: str) -> List[str]:
    if retrieval_mode == "hybrid":
        return transform_query(query, strategy="expansion")
    return [query]


# =============================================================================
# RETRIEVAL — DENSE (Vector Search)
# =============================================================================

def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Dense retrieval: tìm kiếm theo embedding similarity trong ChromaDB.

    Args:
        query: Câu hỏi của người dùng
        top_k: Số chunk tối đa trả về

    Returns:
        List các dict, mỗi dict là một chunk với:
          - "text": nội dung chunk
          - "metadata": metadata (source, section, effective_date, ...)
          - "score": cosine similarity score

    TODO Sprint 2:
    1. Embed query bằng cùng model đã dùng khi index (xem index.py)
    2. Query ChromaDB với embedding đó
    3. Trả về kết quả kèm score

    Gợi ý:
        import chromadb
        from index import get_embedding, CHROMA_DB_DIR

        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection("rag_lab")

        query_embedding = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        # Lưu ý: distances trong ChromaDB cosine = 1 - similarity
        # Score = 1 - distance
    """
    collection, error = _collection_available()
    if collection is None or get_embedding is None:
        return _fallback_retrieve_lexical(query, top_k=top_k)

    try:
        query_embedding = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return _fallback_retrieve_lexical(query, top_k=top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    dense_results = []

    for doc, metadata, distance in zip(documents, metadatas, distances):
        score = 1 - float(distance)
        dense_results.append({
            "text": doc,
            "metadata": metadata or {},
            "score": score,
        })

    return dense_results


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Keyword Search)
# Dùng cho Sprint 3 Variant hoặc kết hợp Hybrid
# =============================================================================

def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Sparse retrieval: tìm kiếm theo keyword (BM25).

    Mạnh ở: exact term, mã lỗi, tên riêng (ví dụ: "ERR-403", "P1", "refund")
    Hay hụt: câu hỏi paraphrase, đồng nghĩa

    TODO Sprint 3 (nếu chọn hybrid):
    1. Cài rank_bm25: pip install rank-bm25
    2. Load tất cả chunks từ ChromaDB (hoặc rebuild từ docs)
    3. Tokenize và tạo BM25Index
    4. Query và trả về top_k kết quả

    Gợi ý:
        from rank_bm25 import BM25Okapi
        corpus = [chunk["text"] for chunk in all_chunks]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    """
    all_chunks = _load_all_chunks_from_docs()
    if not all_chunks:
        return []

    try:
        from rank_bm25 import BM25Okapi

        global _BM25_CACHE
        if _BM25_CACHE is None:
            tokenized_corpus = [_tokenize(chunk["text"]) for chunk in all_chunks]
            _BM25_CACHE = (BM25Okapi(tokenized_corpus), tokenized_corpus)

        bm25, _ = _BM25_CACHE
        tokenized_query = _tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        max_score = max(float(scores[i]) for i in ranked_indices) if ranked_indices else 0.0
        sparse_results = []
        for i in ranked_indices:
            raw_score = float(scores[i])
            normalized_score = (raw_score / max_score) if max_score > 0 else 0.0
            sparse_results.append({
                "text": all_chunks[i]["text"],
                "metadata": all_chunks[i]["metadata"],
                "score": normalized_score,
            })
        return sparse_results
    except ModuleNotFoundError:
        pass

    return _fallback_retrieve_lexical(query, top_k=top_k)


# =============================================================================
# RETRIEVAL — HYBRID (Dense + Sparse với Reciprocal Rank Fusion)
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: kết hợp dense và sparse bằng Reciprocal Rank Fusion (RRF).

    Mạnh ở: giữ được cả nghĩa (dense) lẫn keyword chính xác (sparse)
    Phù hợp khi: corpus lẫn lộn ngôn ngữ tự nhiên và tên riêng/mã lỗi/điều khoản

    Args:
        dense_weight: Trọng số cho dense score (0-1)
        sparse_weight: Trọng số cho sparse score (0-1)

    TODO Sprint 3 (nếu chọn hybrid):
    1. Chạy retrieve_dense() → dense_results
    2. Chạy retrieve_sparse() → sparse_results
    3. Merge bằng RRF:
       RRF_score(doc) = dense_weight * (1 / (60 + dense_rank)) +
                        sparse_weight * (1 / (60 + sparse_rank))
       60 là hằng số RRF tiêu chuẩn
    4. Sort theo RRF score giảm dần, trả về top_k

    Khi nào dùng hybrid (từ slide):
    - Corpus có cả câu tự nhiên VÀ tên riêng, mã lỗi, điều khoản
    - Query như "Approval Matrix" khi doc đổi tên thành "Access Control SOP"
    """
    dense_results = retrieve_dense(query, top_k=top_k)
    sparse_results = retrieve_sparse(query, top_k=top_k)

    fused: Dict[str, Dict[str, Any]] = {}

    for rank, chunk in enumerate(dense_results, 1):
        key = f"{chunk['metadata'].get('source', '')}::{chunk['metadata'].get('section', '')}::{_normalize_text(chunk['text'])[:120]}"
        score = dense_weight * (1 / (60 + rank))
        existing = fused.get(key, {
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "score": 0.0,
        })
        existing["score"] += score
        fused[key] = existing

    for rank, chunk in enumerate(sparse_results, 1):
        key = f"{chunk['metadata'].get('source', '')}::{chunk['metadata'].get('section', '')}::{_normalize_text(chunk['text'])[:120]}"
        score = sparse_weight * (1 / (60 + rank))
        existing = fused.get(key, {
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "score": 0.0,
        })
        existing["score"] += score
        fused[key] = existing

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


# =============================================================================
# RERANK (Sprint 3 alternative)
# Cross-encoder để chấm lại relevance sau search rộng
# =============================================================================

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    """
    Rerank các candidate chunks bằng cross-encoder.

    Cross-encoder: chấm lại "chunk nào thực sự trả lời câu hỏi này?"
    MMR (Maximal Marginal Relevance): giữ relevance nhưng giảm trùng lặp

    Funnel logic (từ slide):
      Search rộng (top-20) → Rerank (top-6) → Select (top-3)

    TODO Sprint 3 (nếu chọn rerank):
    Option A — Cross-encoder:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, chunk["text"]] for chunk in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_k]]

    Option B — Rerank bằng LLM (đơn giản hơn nhưng tốn token):
        Gửi list chunks cho LLM, yêu cầu chọn top_k relevant nhất

    Khi nào dùng rerank:
    - Dense/hybrid trả về nhiều chunk nhưng có noise
    - Muốn chắc chắn chỉ 3-5 chunk tốt nhất vào prompt
    """
    if not candidates:
        return []

    try:
        from sentence_transformers import CrossEncoder

        global _CROSS_ENCODER_MODEL
        if _CROSS_ENCODER_MODEL is None:
            _CROSS_ENCODER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, chunk["text"]] for chunk in candidates]
        scores = _CROSS_ENCODER_MODEL.predict(pairs)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            {**chunk, "score": float(score)}
            for chunk, score in ranked[:top_k]
        ]
    except Exception:
        ranked = []
        for chunk in candidates:
            lexical = _lexical_score(query, chunk["text"])
            combined = (float(chunk.get("score", 0) or 0) * 0.6) + (lexical * 0.4)
            ranked.append({**chunk, "score": combined})
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]


# =============================================================================
# QUERY TRANSFORMATION (Sprint 3 alternative)
# =============================================================================

def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    """
    Biến đổi query để tăng recall.

    Strategies:
      - "expansion": Thêm từ đồng nghĩa, alias, tên cũ
      - "decomposition": Tách query phức tạp thành 2-3 sub-queries
      - "hyde": Sinh câu trả lời giả (hypothetical document) để embed thay query

    TODO Sprint 3 (nếu chọn query transformation):
    Gọi LLM với prompt phù hợp với từng strategy.

    Ví dụ expansion prompt:
        "Given the query: '{query}'
         Generate 2-3 alternative phrasings or related terms in Vietnamese.
         Output as JSON array of strings."

    Ví dụ decomposition:
        "Break down this complex query into 2-3 simpler sub-queries: '{query}'
         Output as JSON array."

    Khi nào dùng:
    - Expansion: query dùng alias/tên cũ (ví dụ: "Approval Matrix" → "Access Control SOP")
    - Decomposition: query hỏi nhiều thứ một lúc
    - HyDE: query mơ hồ, search theo nghĩa không hiệu quả
    """
    if strategy != "expansion":
        return [query]

    variants = [query]
    lowered = query.lower()

    alias_rules = {
        "approval matrix": ["Access Control SOP", "Approval Matrix for System Access", "cấp quyền hệ thống"],
        "p1": ["ticket P1", "incident P1", "critical incident"],
        "hoàn tiền": ["refund", "chính sách hoàn tiền"],
        "remote": ["remote work", "làm việc từ xa"],
        "vpn": ["Cisco AnyConnect", "kết nối từ xa"],
    }

    for alias, expansions in alias_rules.items():
        if alias in lowered:
            variants.extend([query] + expansions)

    if "err-" in lowered:
        variants.append(query.replace("-", " "))

    deduped = []
    seen = set()
    for item in variants:
        normalized = item.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(item.strip())
    return deduped


# =============================================================================
# GENERATION — GROUNDED ANSWER FUNCTION
# =============================================================================

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Đóng gói danh sách chunks thành context block để đưa vào prompt.

    Format: structured snippets với source, section, score (từ slide).
    Mỗi chunk có số thứ tự [1], [2], ... để model dễ trích dẫn.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        effective_date = meta.get("effective_date", "")
        score = chunk.get("score", 0)
        text = chunk.get("text", "")

        # TODO: Tùy chỉnh format nếu muốn (thêm effective_date, department, ...)
        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if effective_date:
            header += f" | effective_date={effective_date}"
        if score > 0:
            header += f" | score={score:.2f}"

        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Xây dựng grounded prompt theo 4 quy tắc từ slide:
    1. Evidence-only: Chỉ trả lời từ retrieved context
    2. Abstain: Thiếu context thì nói không đủ dữ liệu
    3. Citation: Gắn source/section khi có thể
    4. Short, clear, stable: Output ngắn, rõ, nhất quán

    TODO Sprint 2:
    Đây là prompt baseline. Trong Sprint 3, bạn có thể:
    - Thêm hướng dẫn về format output (JSON, bullet points)
    - Thêm ngôn ngữ phản hồi (tiếng Việt vs tiếng Anh)
    - Điều chỉnh tone phù hợp với use case (CS helpdesk, IT support)
    """
    prompt = f"""Answer only from the retrieved context below.
If the context is insufficient to answer the question, say you do not know and do not make up information.
Cite the source field (in brackets like [1]) when possible.
Keep your answer short, clear, and factual.
Respond in the same language as the question.

Question: {query}

Context:
{context_block}

Answer:"""
    return prompt


def call_llm(prompt: str) -> str:
    """
    Gọi LLM để sinh câu trả lời.

    TODO Sprint 2:
    Chọn một trong hai:

    Option A — OpenAI (cần OPENAI_API_KEY):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,     # temperature=0 để output ổn định, dễ đánh giá
            max_tokens=512,
        )
        return response.choices[0].message.content

    Option B — Google Gemini (cần GOOGLE_API_KEY):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    Lưu ý: Dùng temperature=0 hoặc thấp để output ổn định cho evaluation.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if openai_api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=LLM_MAX_TOKENS,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            pass

    if google_api_key:
        try:
            import google.generativeai as genai

            genai.configure(api_key=google_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return (response.text or "").strip()
        except Exception:
            pass

    question_match = re.search(r"Question:\s*(.+?)\n\nContext:", prompt, re.DOTALL)
    query = question_match.group(1).strip() if question_match else ""
    context_chunks = []
    for block in re.split(r"\n(?=\[\d+\]\s)", prompt):
        match = re.match(r"\[(\d+)\]\s+([^\n]+)\n(.+)", block.strip(), re.DOTALL)
        if not match:
            continue
        idx, header, text = match.groups()
        source = header.split("|")[0].strip()
        context_chunks.append({
            "text": text.strip(),
            "metadata": {"source": source},
            "score": 0.0,
        })
    return _fallback_generate_answer(query, context_chunks)


def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline RAG hoàn chỉnh: query → retrieve → (rerank) → generate.

    Args:
        query: Câu hỏi
        retrieval_mode: "dense" | "sparse" | "hybrid"
        top_k_search: Số chunk lấy từ vector store (search rộng)
        top_k_select: Số chunk đưa vào prompt (sau rerank/select)
        use_rerank: Có dùng cross-encoder rerank không
        verbose: In thêm thông tin debug

    Returns:
        Dict với:
          - "answer": câu trả lời grounded
          - "sources": list source names trích dẫn
          - "chunks_used": list chunks đã dùng
          - "query": query gốc
          - "config": cấu hình pipeline đã dùng

    TODO Sprint 2 — Implement pipeline cơ bản:
    1. Chọn retrieval function dựa theo retrieval_mode
    2. Gọi rerank() nếu use_rerank=True
    3. Truncate về top_k_select chunks
    4. Build context block và grounded prompt
    5. Gọi call_llm() để sinh câu trả lời
    6. Trả về kết quả kèm metadata

    TODO Sprint 3 — Thử các variant:
    - Variant A: đổi retrieval_mode="hybrid"
    - Variant B: bật use_rerank=True
    - Variant C: thêm query transformation trước khi retrieve
    """
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
    }

    # --- Bước 1: Retrieve ---
    transformed_queries = _get_transformed_queries(query, retrieval_mode)
    merged_candidates: List[Dict[str, Any]] = []

    for query_variant in transformed_queries:
        if retrieval_mode == "dense":
            partial_candidates = retrieve_dense(query_variant, top_k=top_k_search)
        elif retrieval_mode == "sparse":
            partial_candidates = retrieve_sparse(query_variant, top_k=top_k_search)
        elif retrieval_mode == "hybrid":
            partial_candidates = retrieve_hybrid(query_variant, top_k=top_k_search)
        else:
            raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")
        merged_candidates.extend(partial_candidates)

    deduped: Dict[str, Dict[str, Any]] = {}
    for candidate in merged_candidates:
        key = f"{candidate.get('metadata', {}).get('source', '')}::{candidate.get('metadata', {}).get('section', '')}::{_normalize_text(candidate.get('text', ''))[:120]}"
        existing = deduped.get(key)
        if existing is None or float(candidate.get("score", 0) or 0) > float(existing.get("score", 0) or 0):
            deduped[key] = candidate
    candidates = sorted(deduped.values(), key=lambda item: float(item.get("score", 0) or 0), reverse=True)[:top_k_search]

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:3]):
            print(f"  [{i+1}] score={c.get('score', 0):.3f} | {c['metadata'].get('source', '?')}")

    # --- Bước 2: Rerank (optional) ---
    if use_rerank:
        candidates = rerank(query, candidates, top_k=top_k_select)
    else:
        candidates = candidates[:top_k_select]

    if verbose:
        print(f"[RAG] After select: {len(candidates)} chunks")

    # --- Bước 3: Build context và prompt ---
    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"\n[RAG] Prompt:\n{prompt[:500]}...\n")

    # --- Bước 4: Generate ---
    if not _is_context_sufficient(query, candidates):
        answer = "Không đủ dữ liệu trong tài liệu hiện có để trả lời câu hỏi này."
    else:
        answer = call_llm(prompt)

    # --- Bước 5: Extract sources ---
    sources = sorted({
        c["metadata"].get("source", "unknown")
        for c in candidates
    })

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": candidates,
        "config": config,
    }


# =============================================================================
# SPRINT 3: SO SÁNH BASELINE VS VARIANT
# =============================================================================

def compare_retrieval_strategies(query: str) -> None:
    """
    So sánh các retrieval strategies với cùng một query.

    TODO Sprint 3:
    Chạy hàm này để thấy sự khác biệt giữa dense, sparse, hybrid.
    Dùng để justify tại sao chọn variant đó cho Sprint 3.

    A/B Rule (từ slide): Chỉ đổi MỘT biến mỗi lần.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    strategies = ["dense", "hybrid"]  # Thêm "sparse" sau khi implement

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        try:
            result = rag_answer(query, retrieval_mode=strategy, verbose=False)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except NotImplementedError as e:
            print(f"Chưa implement: {e}")
        except Exception as e:
            print(f"Lỗi: {e}")


# =============================================================================
# MAIN — Demo và Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 + 3: RAG Answer Pipeline")
    print("=" * 60)

    # Test queries từ data/test_questions.json
    test_queries = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
        "Ai phải phê duyệt để cấp quyền Level 3?",
        "ERR-403-AUTH là lỗi gì?",  # Query không có trong docs → kiểm tra abstain
    ]

    print("\n--- Sprint 2: Test Baseline (Dense) ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except NotImplementedError:
            print("Chưa implement — hoàn thành TODO trong retrieve_dense() và call_llm() trước.")
        except Exception as e:
            print(f"Lỗi: {e}")

    # Uncomment sau khi Sprint 3 hoàn thành:
    # print("\n--- Sprint 3: So sánh strategies ---")
    # compare_retrieval_strategies("Approval Matrix để cấp quyền là tài liệu nào?")
    # compare_retrieval_strategies("ERR-403-AUTH")

    print("\n\nViệc cần làm Sprint 2:")
    print("  1. Implement retrieve_dense() — query ChromaDB")
    print("  2. Implement call_llm() — gọi OpenAI hoặc Gemini")
    print("  3. Chạy rag_answer() với 3+ test queries")
    print("  4. Verify: output có citation không? Câu không có docs → abstain không?")

    print("\nViệc cần làm Sprint 3:")
    print("  1. Chọn 1 trong 3 variants: hybrid, rerank, hoặc query transformation")
    print("  2. Implement variant đó")
    print("  3. Chạy compare_retrieval_strategies() để thấy sự khác biệt")
    print("  4. Ghi lý do chọn biến đó vào docs/tuning-log.md")
