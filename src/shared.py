# shared.py
"""Single place for clients and models so they're loaded once per process."""

import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

LLM_MODEL = "openai/gpt-oss-20b"

# New table so the original `documents` table stays untouched (easy rollback).
# The SQL functions in supabase_migration.sql are built against this table.
DOCUMENTS_TABLE = "documents_v2"

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
groq_client = Groq()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed texts with normalized vectors (used for both docs and queries)."""
    vectors = embed_model.encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.tolist()


def list_session_files(session_id: str) -> list[dict]:
    """Returns [{'source_file': str, 'chunk_count': int}, ...] in upload order."""
    if not session_id:
        return []
    resp = supabase.rpc(
        "list_session_files", {"filter_session_id": session_id}
    ).execute()
    return resp.data or []
