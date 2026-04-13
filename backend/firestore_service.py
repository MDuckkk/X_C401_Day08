"""
Firestore persistence layer for RAG system.
Stores: document chunks, chat history, and user memories.
"""
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime


# ─── Initialize Firebase ──────────────────────────────────────────────────────

_db = None

def _get_db():
    """Lazy-initialize Firestore client."""
    global _db
    if _db is not None:
        return _db

    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS env var not set. "
            "Set it to the path of your Firebase service account JSON key."
        )

    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    _db = firestore.client()
    return _db


# ─── Document Chunks ──────────────────────────────────────────────────────────

CHUNKS_COLLECTION = "document_chunks"

def save_chunks_to_firestore(source: str, chunks: list[dict]):
    """
    Save document chunks to Firestore.
    Each chunk: {"content": str, "source": str, "chunk_index": int, "total_chunks": int}
    """
    db = _get_db()
    batch = db.batch()

    for chunk in chunks:
        doc_id = f"{source}_chunk_{chunk['chunk_index']}"
        ref = db.collection(CHUNKS_COLLECTION).document(doc_id)
        batch.set(ref, {
            "source": source,
            "content": chunk["content"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "department": chunk.get("department", ""),
            "created_at": datetime.utcnow().isoformat(),
        })

    batch.commit()


def save_single_doc_to_firestore(source: str, content: str):
    """Save a single (non-chunked) document to Firestore."""
    db = _get_db()
    ref = db.collection(CHUNKS_COLLECTION).document(source)
    ref.set({
        "source": source,
        "content": content,
        "chunk_index": 0,
        "total_chunks": 1,
        "created_at": datetime.utcnow().isoformat(),
    })


def load_all_chunks_from_firestore() -> list[dict]:
    """Load all document chunks from Firestore."""
    db = _get_db()
    docs = db.collection(CHUNKS_COLLECTION).stream()
    return [doc.to_dict() for doc in docs]


def delete_chunks_from_firestore(source: str):
    """Delete all chunks for a given source from Firestore."""
    db = _get_db()
    # Query all chunks with this source
    query = db.collection(CHUNKS_COLLECTION).where("source", "==", source)
    docs = query.stream()
    batch = db.batch()
    for doc in docs:
        batch.delete(doc.reference)
    batch.commit()


# ─── Chat History ─────────────────────────────────────────────────────────────

CHAT_COLLECTION = "chat_history"

def save_chat_message(thread_id: str, role: str, content: str):
    """Save a single chat message to Firestore."""
    db = _get_db()
    db.collection(CHAT_COLLECTION).add({
        "thread_id": thread_id,
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
    })


def load_chat_history(thread_id: str) -> list[dict]:
    """Load chat history for a thread, ordered by timestamp."""
    db = _get_db()
    query = (
        db.collection(CHAT_COLLECTION)
        .where("thread_id", "==", thread_id)
        .order_by("timestamp")
    )
    docs = query.stream()
    return [{"role": d.to_dict()["role"], "content": d.to_dict()["content"]} for d in docs]


def delete_chat_history(thread_id: str):
    """Delete all messages for a thread."""
    db = _get_db()
    query = db.collection(CHAT_COLLECTION).where("thread_id", "==", thread_id)
    docs = query.stream()
    batch = db.batch()
    for doc in docs:
        batch.delete(doc.reference)
    batch.commit()


# ─── User Memories ────────────────────────────────────────────────────────────

MEMORIES_COLLECTION = "user_memories"

def save_user_memory_firestore(user_id: str, facts: list[str]):
    """Save user memory facts to Firestore."""
    db = _get_db()
    ref = db.collection(MEMORIES_COLLECTION).document(user_id)
    ref.set({
        "user_id": user_id,
        "facts": facts,
        "updated_at": datetime.utcnow().isoformat(),
    })


def load_user_memory_firestore(user_id: str) -> list[str]:
    """Load user memory facts from Firestore."""
    db = _get_db()
    ref = db.collection(MEMORIES_COLLECTION).document(user_id)
    doc = ref.get()
    if doc.exists:
        return doc.to_dict().get("facts", [])
    return []
