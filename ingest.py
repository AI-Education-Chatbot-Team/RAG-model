import os
import tempfile

os.environ["RAPIDOCR_MODEL_DIR"] = os.path.join(tempfile.gettempdir(), "rapidocr_models")

import io
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

doc_converter = DocumentConverter()

SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".docx", ".csv"]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", "#", ". ", " ", ""]
)

def extract_and_chunk(file_path: str, source_file_name: str) -> list[dict]:
    
    result = doc_converter.convert(file_path)
    markdown_text = result.document.export_to_markdown()

    if not markdown_text.strip():
        return []

    
    if not markdown_text.strip():
        return []

    raw_chunks = text_splitter.split_text(markdown_text)

    return [
        {
            "text": chunk,
            "source_file": source_file_name,
            "page_number": 1,
        }
        for chunk in raw_chunks
    ]

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def embed_and_store(chunks_with_meta: list[dict], session_id: str | None = None) -> list[int]:
    inserted_ids = []
    for item in chunks_with_meta:
        embedding = model.encode(item["text"]).tolist()

        response = supabase.table("documents").insert({
            "content": item["text"],
            "embedding": embedding,
            "source_file": item["source_file"],
            "page_number": item["page_number"],
            "session_id": session_id,
        }).execute()

        if response.data:
            inserted_ids.append(response.data[0]["id"])

    print(f"Successfully ingested {len(chunks_with_meta)} chunks into Supabase!")
    return inserted_ids


def delete_documents(ids: list[int]) -> None:
    if not ids:
        return
    supabase.table("documents").delete().in_("id", ids).execute()
    print(f"Deleted {len(ids)} chunk(s) by ID.")


def delete_session_documents(session_id: str) -> int:
    if not session_id:
        return 0
    response = supabase.table("documents").delete().eq("session_id", session_id).execute()
    deleted = len(response.data) if response.data else 0
    print(f"Deleted {deleted} chunk(s) for session {session_id}.")
    return deleted