import os
import glob
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client
from pypdf import PdfReader

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_pages_from_pdf(pdf_source) -> list[dict]:
    #Extract all text from PDF
    reader = PdfReader(pdf_source)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            pages.append({"page_number": i, "text": page_text})
    return pages


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


def chunk_pdf_with_metadata(
    pdf_source, source_file: str, chunk_size: int = 100, overlap: int = 20
) -> list[dict]:
    """
    Extract + chunk a PDF, tagging every chunk with its source filename
    and the page it came from.
    Returns a list of {"text": str, "source_file": str, "page_number": int}.
    """
    pages = extract_pages_from_pdf(pdf_source)
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


def load_pdfs_from_folder(folder_path: str) -> list[dict]:
    # Extract and chunk text from every PDF in a folder, with metadata
    all_chunks = []
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return all_chunks

    for pdf_path in pdf_files:
        print(f"Reading {pdf_path}...")
        source_file = os.path.basename(pdf_path)
        chunks = chunk_pdf_with_metadata(pdf_path, source_file)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks extracted")

    embed_and_store(all_chunks)
    return all_chunks


def embed_and_store(chunks_with_meta: list[dict]):
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
        }).execute()

    print(f"Successfully ingested {len(chunks_with_meta)} chunks into Supabase!")


if __name__ == "__main__":
    # Pointing to folder containing PDFs
    pdf_folder = "./pdfs"
    load_pdfs_from_folder(pdf_folder)