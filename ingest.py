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

def extract_text_from_pdf(pdf_path: str) -> str:
    #Extract all text from PDF
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    #Split text into word based chunks
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

def load_pdfs_from_folder(folder_path: str) -> list[str]
    #Extract and chunk text from every PDF in a folder
    all_chunks = []
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return all_chunks

    for pdf_path in pdf_files:
        print(f"Reading {pdf_path}...")
        raw_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(raw_text)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks extracted")

    return all_chunks
        

def embed_and_store(texts: list[str]):
    for text in texts:
        # Generate vector embedding
        embedding = model.encode(text).tolist()
        
        # Insert into Supabase
        supabase.table("documents").insert({
            "content": text,
            "embedding": embedding
        }).execute()
        
    print(f"Successfully ingested {len(texts)} chunks into Supabase!")

if __name__ == "__main__":
    #Pointing to folder containing PDFs
    pdf_folder = "./pdfs"

    # Test "Documents"
    knowledge_base = [
        "Groq uses Language Processing Units (LPUs) to deliver extremely high-speed LLM inference.",
        "Supabase provides open-source PostgreSQL database services with native pgvector support.",
        "RAG enhances language models by retrieving relevant external data before generating responses."
    ]
        if knowledge_base:
            embed_and_store(knowledge_base)
        else:
            print("No text extracted — nothing to ingest.")
