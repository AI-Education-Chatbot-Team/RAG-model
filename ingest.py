import os
import tempfile
import io
from dotenv import load_dotenv

os.environ["RAPIDOCR_MODEL_DIR"] = os.path.join(tempfile.gettempdir(), "rapidocr_models")

from docling.datamodel.base_models import DocumentStream
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

def extract_and_chunk(uploaded_file: str) -> list[dict]:
    buf = io.BytesIO(uploaded_file.getvalue())
    doc_stream = DocumentStream(name=uploaded_file.name, filename=uploaded_file.name, stream=buf)
    
    
    result = doc_converter.convert(doc_stream)
    markdown_text = result.document.export_to_markdown()

    if not markdown_text.strip():
        return []

    
    raw_chunks = text_splitter.split_text(markdown_text)
    
    return [
        {
            "text": chunk,
            "source_file": uploaded_file.name,
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