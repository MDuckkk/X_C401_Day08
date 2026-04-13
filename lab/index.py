"""
index.py — Sprint 1: Build RAG Index
====================================
Mục tiêu Sprint 1 (60 phút):
  - Đọc và preprocess tài liệu từ data/docs/
  - Chunk tài liệu theo cấu trúc tự nhiên (heading/section)
  - Gắn metadata: source, section, department, effective_date, access
  - Embed và lưu vào vector store (ChromaDB)

Definition of Done Sprint 1:
  ✓ Script chạy được và index đủ docs
  ✓ Có ít nhất 3 metadata fields hữu ích cho retrieval
  ✓ Có thể kiểm tra chunk bằng list_chunks()
"""

import os
import json
import re
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
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

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

# TODO Sprint 1: Điều chỉnh chunk size và overlap theo quyết định của nhóm
# Gợi ý từ slide: chunk 300-500 tokens, overlap 50-80 tokens
DEFAULT_CHUNK_SIZE = 360
DEFAULT_CHUNK_OVERLAP = 60
EMBEDDING_MODEL = "text-embedding-3-small"
_OPENAI_CLIENT = None
_SENTENCE_TRANSFORMER_MODEL = None


def _estimate_chunk_settings(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    """
    Ước lượng chunk size/overlap dựa trên dữ liệu thật trong data/docs.

    Heuristic:
    - Bộ dữ liệu hiện tại gồm tài liệu ngắn, chia section rõ ràng bằng heading === ... ===
    - Ưu tiên chunk nhỏ-vừa để retrieval chính xác hơn cho fact lookup
    - Chunk size được giữ trong khoảng 320-480 tokens, overlap khoảng 15-20%
    """
    doc_files = sorted(docs_dir.glob("*.txt"))
    if not doc_files:
        return DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

    section_lengths = []
    paragraph_lengths = []
    for filepath in doc_files:
        try:
            raw_text = filepath.read_text(encoding="utf-8")
        except OSError:
            continue

        sections = re.split(r"===.*?===", raw_text)
        for section in sections:
            cleaned = section.strip()
            if not cleaned:
                continue
            section_lengths.append(len(cleaned))
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
            paragraph_lengths.extend(len(p) for p in paragraphs)

    if not section_lengths:
        return DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

    section_lengths.sort()
    paragraph_lengths.sort()
    median_section_chars = section_lengths[len(section_lengths) // 2]
    median_paragraph_chars = (
        paragraph_lengths[len(paragraph_lengths) // 2]
        if paragraph_lengths
        else median_section_chars // 2
    )

    estimated_tokens = max(320, min(480, round((median_section_chars / 4) * 1.35 / 20) * 20))
    overlap_tokens = max(50, min(80, round(max(median_paragraph_chars / 4, estimated_tokens * 0.18) / 10) * 10))
    overlap_tokens = min(overlap_tokens, estimated_tokens // 3)
    return int(estimated_tokens), int(overlap_tokens)


CHUNK_SIZE, CHUNK_OVERLAP = _estimate_chunk_settings()


# =============================================================================
# STEP 1: PREPROCESS
# Làm sạch text trước khi chunk và embed
# =============================================================================

def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    """
    Preprocess một tài liệu: extract metadata từ header và làm sạch nội dung.

    Args:
        raw_text: Toàn bộ nội dung file text
        filepath: Đường dẫn file để làm source mặc định

    Returns:
        Dict chứa:
          - "text": nội dung đã clean
          - "metadata": dict với source, department, effective_date, access

    TODO Sprint 1:
    - Extract metadata từ dòng đầu file (Source, Department, Effective Date, Access)
    - Bỏ các dòng header metadata khỏi nội dung chính
    - Normalize khoảng trắng, xóa ký tự rác

    Gợi ý: dùng regex để parse dòng "Key: Value" ở đầu file.
    """
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")
    metadata = {
        "source": filepath,
        "section": "",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    content_lines = []
    header_done = False

    metadata_key_map = {
        "source": "source",
        "department": "department",
        "effective date": "effective_date",
        "access": "access",
    }

    for line in lines:
        stripped = line.strip()
        if not header_done:
            # TODO: Parse metadata từ các dòng "Key: Value"
            # Ví dụ: "Source: policy/refund-v4.pdf" → metadata["source"] = "policy/refund-v4.pdf"
            metadata_match = re.match(r"^([A-Za-z ]+):\s*(.+?)\s*$", stripped)
            if metadata_match:
                raw_key, raw_value = metadata_match.groups()
                metadata_key = metadata_key_map.get(raw_key.lower().strip())
                if metadata_key:
                    metadata[metadata_key] = raw_value.strip()
                    continue

            if stripped.startswith("==="):
                # Gặp section heading đầu tiên → kết thúc header
                header_done = True
                content_lines.append(stripped)
            elif stripped == "" or stripped.isupper():
                # Dòng tên tài liệu (toàn chữ hoa) hoặc dòng trống
                continue
            else:
                # Ví dụ: "Ghi chú:" trước section đầu tiên vẫn nên giữ lại.
                header_done = True
                content_lines.append(stripped)
        else:
            content_lines.append(line.rstrip())

    cleaned_text = "\n".join(content_lines)

    # TODO: Thêm bước normalize text nếu cần
    # Gợi ý: bỏ ký tự đặc biệt thừa, chuẩn hóa dấu câu
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = re.sub(r" *\n *", "\n", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)  # max 2 dòng trống liên tiếp
    cleaned_text = cleaned_text.strip()

    return {
        "text": cleaned_text,
        "metadata": metadata,
    }


# =============================================================================
# STEP 2: CHUNK
# Chia tài liệu thành các đoạn nhỏ theo cấu trúc tự nhiên
# =============================================================================

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk một tài liệu đã preprocess thành danh sách các chunk nhỏ.

    Args:
        doc: Dict với "text" và "metadata" (output của preprocess_document)

    Returns:
        List các Dict, mỗi dict là một chunk với:
          - "text": nội dung chunk
          - "metadata": metadata gốc + "section" của chunk đó

    TODO Sprint 1:
    1. Split theo heading "=== Section ... ===" hoặc "=== Phần ... ===" trước
    2. Nếu section quá dài (> CHUNK_SIZE * 4 ký tự), split tiếp theo paragraph
    3. Thêm overlap: lấy đoạn cuối của chunk trước vào đầu chunk tiếp theo
    4. Mỗi chunk PHẢI giữ metadata đầy đủ từ tài liệu gốc

    Gợi ý: Ưu tiên cắt tại ranh giới tự nhiên (section, paragraph)
    thay vì cắt theo token count cứng.
    """
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    # TODO: Implement chunking theo section heading
    # Bước 1: Split theo heading pattern "=== ... ==="
    heading_pattern = re.compile(r"^(===.*?===)\s*$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))

    if not matches:
        chunks.extend(
            _split_by_size(
                text.strip(),
                base_metadata=base_metadata,
                section="General",
            )
        )
    else:
        leading_text = text[:matches[0].start()].strip()
        if leading_text:
            chunks.extend(
                _split_by_size(
                    leading_text,
                    base_metadata=base_metadata,
                    section="General",
                )
            )

        for index, match in enumerate(matches):
            current_section = match.group(1).strip("= ").strip()
            section_start = match.end()
            section_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section_body = text[section_start:section_end].strip()
            if not section_body:
                continue
            section_chunks = _split_by_size(
                section_body,
                base_metadata=base_metadata,
                section=current_section,
            )
            chunks.extend(section_chunks)

    for idx, chunk in enumerate(chunks):
        chunk["metadata"]["chunk_index"] = idx
        chunk["metadata"]["chunk_chars"] = len(chunk["text"])

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict,
    section: str,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """
    Helper: Split text dài thành chunks với overlap.

    TODO Sprint 1:
    Hiện tại dùng split đơn giản theo ký tự.
    Cải thiện: split theo paragraph (\n\n) trước, rồi mới ghép đến khi đủ size.
    """
    if len(text) <= chunk_chars:
        # Toàn bộ section vừa một chunk
        return [{
            "text": text,
            "metadata": {**base_metadata, "section": section},
        }]

    # TODO: Implement split theo paragraph với overlap
    # Gợi ý:
    # paragraphs = text.split("\n\n")
    # Ghép paragraphs lại cho đến khi gần đủ chunk_chars
    # Lấy overlap từ đoạn cuối chunk trước
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    segments: List[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= chunk_chars:
            segments.append(paragraph)
            continue
        segments.extend(_split_large_paragraph(paragraph, chunk_chars))

    chunks = []
    current_parts: List[str] = []
    current_length = 0

    for segment in segments:
        segment_length = len(segment) + (2 if current_parts else 0)
        if current_parts and current_length + segment_length > chunk_chars:
            chunk_text = "\n\n".join(current_parts).strip()
            chunks.append({
                "text": chunk_text,
                "metadata": {**base_metadata, "section": section},
            })

            overlap_text = _build_overlap_text(chunk_text, overlap_chars)
            current_parts = [overlap_text] if overlap_text else []
            current_length = len(overlap_text) if overlap_text else 0

        current_parts.append(segment)
        current_length += len(segment) + (2 if len(current_parts) > 1 else 0)

    if current_parts:
        chunk_text = "\n\n".join(current_parts).strip()
        if not chunks or chunk_text != chunks[-1]["text"]:
            chunks.append({
                "text": chunk_text,
                "metadata": {**base_metadata, "section": section},
            })

    return chunks


def _split_large_paragraph(paragraph: str, chunk_chars: int) -> List[str]:
    """
    Tách paragraph quá dài ở ranh giới tự nhiên: câu, bullet, rồi mới fallback theo ký tự.
    """
    sentence_like_parts = [
        part.strip()
        for part in re.split(r"(?<=[\.\?!:;])\s+|\n(?=[\-0-9A-Za-z])", paragraph)
        if part.strip()
    ]
    if len(sentence_like_parts) == 1:
        sentence_like_parts = [paragraph]

    pieces: List[str] = []
    buffer = ""

    for part in sentence_like_parts:
        candidate = f"{buffer} {part}".strip() if buffer else part
        if len(candidate) <= chunk_chars:
            buffer = candidate
            continue

        if buffer:
            pieces.append(buffer.strip())
            buffer = ""

        if len(part) <= chunk_chars:
            buffer = part
            continue

        start = 0
        while start < len(part):
            end = min(start + chunk_chars, len(part))
            natural_cut = max(
                part.rfind(". ", start, end),
                part.rfind("; ", start, end),
                part.rfind(", ", start, end),
                part.rfind(" ", start, end),
            )
            if natural_cut <= start:
                natural_cut = end
            else:
                natural_cut += 1
            pieces.append(part[start:natural_cut].strip())
            start = natural_cut

    if buffer:
        pieces.append(buffer.strip())

    return [piece for piece in pieces if piece]


def _build_overlap_text(chunk_text: str, overlap_chars: int) -> str:
    """
    Lấy đoạn cuối chunk trước để giữ ngữ cảnh cho chunk sau.
    Ưu tiên lấy theo paragraph, fallback theo ký tự nếu cần.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", chunk_text) if p.strip()]
    selected: List[str] = []
    total = 0

    for paragraph in reversed(paragraphs):
        addition = len(paragraph) + (2 if selected else 0)
        if selected and total + addition > overlap_chars:
            break
        if not selected and len(paragraph) > overlap_chars:
            return paragraph[-overlap_chars:].strip()
        selected.insert(0, paragraph)
        total += addition
        if total >= overlap_chars:
            break

    return "\n\n".join(selected).strip()


# =============================================================================
# STEP 3: EMBED + STORE
# Embed các chunk và lưu vào ChromaDB
# =============================================================================

def get_embedding(text: str) -> List[float]:
    """
    Tạo embedding vector cho một đoạn text.

    TODO Sprint 1:
    Chọn một trong hai:

    Option A — OpenAI Embeddings (cần OPENAI_API_KEY):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    Option B — Sentence Transformers (chạy local, không cần API key):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return model.encode(text).tolist()
    """
    global _OPENAI_CLIENT, _SENTENCE_TRANSFORMER_MODEL

    normalized_text = re.sub(r"\s+", " ", text).strip()
    if not normalized_text:
        normalized_text = " "

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Thiếu package 'openai'. Hãy chạy: pip install -r requirements.txt"
            ) from exc

        if _OPENAI_CLIENT is None:
            _OPENAI_CLIENT = OpenAI(api_key=openai_api_key)
        response = _OPENAI_CLIENT.embeddings.create(
            input=normalized_text,
            model=EMBEDDING_MODEL,
        )
        return response.data[0].embedding

    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Không tìm thấy model embedding khả dụng. "
            "Hãy cài dependencies bằng `pip install -r requirements.txt` "
            "hoặc cung cấp OPENAI_API_KEY."
        ) from exc

    if _SENTENCE_TRANSFORMER_MODEL is None:
        _SENTENCE_TRANSFORMER_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _SENTENCE_TRANSFORMER_MODEL.encode(normalized_text).tolist()


def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> bool:
    """
    Pipeline hoàn chỉnh: đọc docs → preprocess → chunk → embed → store.

    TODO Sprint 1:
    1. Cài thư viện: pip install chromadb
    2. Khởi tạo ChromaDB client và collection
    3. Với mỗi file trong docs_dir:
       a. Đọc nội dung
       b. Gọi preprocess_document()
       c. Gọi chunk_document()
       d. Với mỗi chunk: gọi get_embedding() và upsert vào ChromaDB
    4. In số lượng chunk đã index

    Gợi ý khởi tạo ChromaDB:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_or_create_collection(
            name="rag_lab",
            metadata={"hnsw:space": "cosine"}
        )
    """
    try:
        import chromadb
    except ModuleNotFoundError:
        print("Thiếu package 'chromadb'. Hãy chạy: pip install -r requirements.txt")
        return False

    print(f"Đang build index từ: {docs_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Khởi tạo ChromaDB
    # client = chromadb.PersistentClient(path=str(db_dir))
    # collection = client.get_or_create_collection(...)
    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        client.delete_collection("rag_lab")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name="rag_lab",
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    doc_files = list(docs_dir.glob("*.txt"))

    if not doc_files:
        print(f"Không tìm thấy file .txt trong {docs_dir}")
        return False

    for filepath in doc_files:
        print(f"  Processing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")

        # TODO: Gọi preprocess_document
        # doc = preprocess_document(raw_text, str(filepath))
        doc = preprocess_document(raw_text, str(filepath))

        # TODO: Gọi chunk_document
        # chunks = chunk_document(doc)
        chunks = chunk_document(doc)

        # TODO: Embed và lưu từng chunk vào ChromaDB
        # for i, chunk in enumerate(chunks):
        #     chunk_id = f"{filepath.stem}_{i}"
        #     embedding = get_embedding(chunk["text"])
        #     collection.upsert(
        #         ids=[chunk_id],
        #         embeddings=[embedding],
        #         documents=[chunk["text"]],
        #         metadatas=[chunk["metadata"]],
        #     )
        # total_chunks += len(chunks)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filepath.stem}_{i}_{hashlib.md5(chunk['text'].encode('utf-8')).hexdigest()[:8]}"
            embedding = get_embedding(chunk["text"])
            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[chunk["metadata"]],
            )
        print(f"    → {len(chunks)} chunks indexed")
        total_chunks += len(chunks)

    print(f"\nHoàn thành! Tổng số chunks: {total_chunks}")
    print(f"Chunk config đang dùng: size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP} tokens")
    return True


# =============================================================================
# STEP 4: INSPECT / KIỂM TRA
# Dùng để debug và kiểm tra chất lượng index
# =============================================================================

def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 5) -> None:
    """
    In ra n chunk đầu tiên trong ChromaDB để kiểm tra chất lượng index.

    TODO Sprint 1:
    Implement sau khi hoàn thành build_index().
    Kiểm tra:
    - Chunk có giữ đủ metadata không? (source, section, effective_date)
    - Chunk có bị cắt giữa điều khoản không?
    - Metadata effective_date có đúng không?
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(limit=n, include=["documents", "metadatas"])

        print(f"\n=== Top {n} chunks trong index ===\n")
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            print(f"[Chunk {i+1}]")
            print(f"  Source: {meta.get('source', 'N/A')}")
            print(f"  Section: {meta.get('section', 'N/A')}")
            print(f"  Effective Date: {meta.get('effective_date', 'N/A')}")
            print(f"  Text preview: {doc[:120]}...")
            print()
    except Exception as e:
        print(f"Lỗi khi đọc index: {e}")
        print("Hãy chạy build_index() trước.")


def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Kiểm tra phân phối metadata trong toàn bộ index.

    Checklist Sprint 1:
    - Mọi chunk đều có source?
    - Có bao nhiêu chunk từ mỗi department?
    - Chunk nào thiếu effective_date?

    TODO: Implement sau khi build_index() hoàn thành.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(include=["metadatas"])

        print(f"\nTổng chunks: {len(results['metadatas'])}")

        # TODO: Phân tích metadata
        # Đếm theo department, kiểm tra effective_date missing, v.v.
        departments = {}
        missing_date = 0
        for meta in results["metadatas"]:
            dept = meta.get("department", "unknown")
            departments[dept] = departments.get(dept, 0) + 1
            if meta.get("effective_date") in ("unknown", "", None):
                missing_date += 1

        print("Phân bố theo department:")
        for dept, count in departments.items():
            print(f"  {dept}: {count} chunks")
        print(f"Chunks thiếu effective_date: {missing_date}")

    except Exception as e:
        print(f"Lỗi: {e}. Hãy chạy build_index() trước.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 1: Build RAG Index")
    print("=" * 60)

    # Bước 1: Kiểm tra docs
    doc_files = list(DOCS_DIR.glob("*.txt"))
    print(f"\nChunk config tự động theo dữ liệu: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"\nTìm thấy {len(doc_files)} tài liệu:")
    for f in doc_files:
        print(f"  - {f.name}")

    # Bước 2: Test preprocess và chunking (không cần API key)
    print("\n--- Test preprocess + chunking ---")
    for filepath in doc_files[:1]:  # Test với 1 file đầu
        raw = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(filepath))
        chunks = chunk_document(doc)
        print(f"\nFile: {filepath.name}")
        print(f"  Metadata: {doc['metadata']}")
        print(f"  Số chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n  [Chunk {i+1}] Section: {chunk['metadata']['section']}")
            print(f"  Text: {chunk['text'][:150]}...")

    # Bước 3: Build index (yêu cầu implement get_embedding)
    print("\n--- Build Full Index ---")
    index_built = build_index()

    # Bước 4: Kiểm tra index
    if index_built:
        list_chunks()
        inspect_metadata_coverage()
    else:
        print("Bỏ qua bước inspect vì index chưa được tạo trong môi trường hiện tại.")

    print("\nSprint 1 setup hoàn thành!")
    if index_built:
        print("Index đã được build và kiểm tra metadata cơ bản.")
    else:
        print("Logic Sprint 1 đã hoàn thiện; chỉ còn cần cài dependencies để build index thật.")
