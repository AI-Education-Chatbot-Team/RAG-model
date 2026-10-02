import os
import io
import csv
import glob
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client
from pypdf import PdfReader
from docx import Document

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".docx", ".csv"]


def extract_pages_from_pdf(pdf_source) -> list[dict]:
    reader = PdfReader(pdf_source)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            pages.append({"page_number": i, "text": page_text})
    return pages


def _read_raw_text(file_source) -> str:
    if hasattr(file_source, "read"):
        data = file_source.read()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore")
        return data
    with open(file_source, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_pages_from_txt(file_source) -> list[dict]:
    text = _read_raw_text(file_source)
    if not text.strip():
        return []
    return [{"page_number": 1, "text": text}]


def extract_pages_from_md(file_source) -> list[dict]:
    text = _read_raw_text(file_source)
    if not text.strip():
        return []
    return [{"page_number": 1, "text": text}]


def extract_pages_from_docx(file_source) -> list[dict]:
    doc = Document(file_source)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if not text.strip():
        return []
    return [{"page_number": 1, "text": text}]


def extract_pages_from_csv(file_source, rows_per_block: int = 50) -> list[dict]:
    if hasattr(file_source, "read"):
        raw = file_source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        stream = io.StringIO(raw)
    else:
        stream = open(file_source, "r", encoding="utf-8", errors="ignore")

    reader = csv.DictReader(stream)
    pages = []
    block_lines = []
    block_number = 1

    for i, row in enumerate(reader, start=1):
        line = ", ".join(f"{k}: {v}" for k, v in row.items())
        block_lines.append(line)
        if i % rows_per_block == 0:
            pages.append({"page_number": block_number, "text": "\n".join(block_lines)})
            block_lines = []
            block_number += 1

    if block_lines:
        pages.append({"page_number": block_number, "text": "\n".join(block_lines)})

    if not hasattr(file_source, "read"):
        stream.close()

    return pages


def extract_pages(file_source, filename: str) -> list[dict]:
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return extract_pages_from_pdf(file_source)
    elif ext == ".txt":
        return extract_pages_from_txt(file_source)
    elif ext == ".md":
        return extract_pages_from_md(file_source)
    elif ext == ".docx":
        return extract_pages_from_docx(file_source)
    elif ext == ".csv":
        return extract_pages_from_csv(file_source)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}"
        )


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
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


def chunk_file_with_metadata(
    file_source, source_file: str, chunk_size: int = 100, overlap: int = 20
) -> list[dict]:
    pages = extract_pages(file_source, source_file)
    chunks_with_meta = []
    for page in pages:
        page_chunks = chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap)
        for chunk in page_chunks:
            chunks_with_meta.append({
                "text": chunk,
                "source_file": source_file,
                "page_number": page["page_number"],
            })
    return chunks_with_meta


def load_files_from_folder(folder_path: str) -> list[dict]:
    all_chunks = []
    files_found = []
    for ext in SUPPORTED_EXTENSIONS:
        files_found.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))

    if not files_found:
        print(f"No supported files found in {folder_path}")
        return all_chunks

    for file_path in files_found:
        print(f"Reading {file_path}...")
        source_file = os.path.basename(file_path)
        try:
            chunks = chunk_file_with_metadata(file_path, source_file)
        except ValueError as e:
            print(f"  -> Skipped: {e}")
            continue
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks extracted")

    embed_and_store(all_chunks, session_id=None)
    return all_chunks


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

if __name__ == "__main__":
    source_folder = "./pdfs"
    load_files_from_folder(source_folder)
