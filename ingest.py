import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

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
    # Test "Documents"
    knowledge_base = [
        "Groq uses Language Processing Units (LPUs) to deliver extremely high-speed LLM inference.",
        "Supabase provides open-source PostgreSQL database services with native pgvector support.",
        "RAG enhances language models by retrieving relevant external data before generating responses."
    ]
    embed_and_store(knowledge_base)