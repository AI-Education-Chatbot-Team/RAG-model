import os
import io
import csv
import glob
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client
from pypdf import PdfReader
from docx import Document

from main import summarize_document

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
    # Split text into word based chunks
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
    """
    Extract + chunk a PDF, tagging every chunk with its source filename
    and the page it came from.
    Returns a list of {"text": str, "source_file": str, "page_number": int}.
    """
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
    # Extract and chunk text from every PDF in a folder, with metadata
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

    embed_and_store(all_chunks)
    embed_and_store(all_chunks, session_id=None)
    return all_chunks


def embed_and_store(chunks_with_meta: list[dict], session_id: str | None = None):
    """
    chunks_with_meta: list of {"text": str, "source_file": str, "page_number": int}
    """
    for item in chunks_with_meta:
        embedding = model.encode(item["text"]).tolist()

        supabase.table("documents").insert({
            "content": item["text"],
            "embedding": embedding,
            "source_file": item["source_file"],
            "page_number": item["page_number"],
            "session_id": session_id,
        }).execute()

    print(f"Successfully ingested {len(chunks_with_meta)} chunks into Supabase!")

def process_uploaded_files(file: str, source_file: str, session_id: str):
    chunks_with_meta = chunk_file_with_metadata(file, source_file)
    embed_and_store(chunks_with_meta, session_id)

    final_summary = summarize_document(chunks_with_meta)

    supabase.table("document_summaries").insert({
        "session_id": session_id,
        "source_file": source_file,
        "summary": final_summary,
    }).execute()


def cleanup_session_data(session_id: str):
    supabase.table("documents").delete().eq(
        "session_id", session_id
    ).execute()
    supabase.table("document_summaries").delete().eq(
            "session_id", session_id
        ).execute()


if __name__ == "__main__":
    # Pointing to folder containing PDFs
    pdf_folder = "./pdfs"
    load_pdfs_from_folder(pdf_folder)