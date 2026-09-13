# main.py
import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
client = Groq()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(query: str, match_count: int = 3) -> str:
    # 1. Embed user query
    query_vector = embed_model.encode(query).tolist()
    
    # 2. Match documents in Supabase
    rpc_response = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_threshold": 0.3,
        "match_count": match_count
    }).execute()
    
    # 3. Concatenate matched contents
    matches = rpc_response.data
    context = "\n".join([item["content"] for item in matches])
    return context

def ask_rag(question: str):
    context = retrieve_context(question)
    
    system_prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer isn't in the context, state that you don't know.

Context:
{context}"""

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None
    )

    print(f"\nQ: {question}\nA: ", end="")
    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    print()

if __name__ == "__main__":
    user_query = "What hardware does Groq use for speed?"
    ask_rag(user_query)