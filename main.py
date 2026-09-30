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


def retrieve_context(query: str, session_id: str | None = None, match_count: int = 3) -> list[dict]:
    query_vector = embed_model.encode(query).tolist()

    rpc_response = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_threshold": 0.3,
        "match_count": match_count,
        "filter_session_id": session_id,
    }).execute()

    return rpc_response.data


def _build_context_str(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[Source: {c.get('source_file', 'Unknown')}, p.{c.get('page_number', 'N/A')}]\n{c.get('content', '')}"
        for c in chunks
    )


BROAD_QUERY_KEYWORDS = [
    "summarize", "summary", "overview", "main points",
    "tl;dr", "tldr", "what is this document about",
    "what's this document about", "what is this about",
]


def is_summary_request(query: str) -> bool:
    q = query.lower()
    return any(keyword in q for keyword in BROAD_QUERY_KEYWORDS)


def get_most_recent_source_file(session_id: str) -> str | None:
    response = supabase.table("documents") \
        .select("source_file") \
        .eq("session_id", session_id) \
        .order("id", desc=True) \
        .limit(1) \
        .execute()

    if response.data:
        return response.data[0]["source_file"]
    return None


def get_full_document_text(session_id: str, source_file: str) -> str:
    response = supabase.table("documents") \
        .select("content, page_number, id") \
        .eq("session_id", session_id) \
        .eq("source_file", source_file) \
        .order("page_number") \
        .order("id") \
        .execute()

    rows = response.data or []
    return "\n".join(r["content"] for r in rows)


def generate_summary(session_id: str):
    source_file = get_most_recent_source_file(session_id)

    if not source_file:
        yield "You haven't uploaded a document in this session yet — upload one first, then ask me to summarize it."
        return

    full_text = get_full_document_text(session_id, source_file)

    if not full_text.strip():
        yield f"I couldn't find any readable content for {source_file}."
        return

    system_prompt = f"""You are a helpful assistant. Provide a clear, concise summary of the following document. Cover the main points and overall purpose. Do not mention that you were given chunks or excerpts — write as if you read the whole document.

Document: {source_file}

Content:
{full_text}"""

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Please summarize this document."}
        ],
        temperature=0.3,
        max_completion_tokens=2048,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None,
    )

    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            yield content


def generate_answer(question: str, chunks: list[dict]):
    context = _build_context_str(chunks)

    system_prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer isn't in the context, state that it is not in the database.

Context:
{context}"""

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.3,
        max_completion_tokens=2048,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None,
    )

    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            yield content


def ask_rag(question: str):
    if is_summary_request(question):
        print(f"\nQ: {question}\nA: ", end="")
        print("Summarization requires a session (use the Streamlit app to upload + summarize a document).")
        return

    chunks = retrieve_context(question, session_id=None)
    print(f"\nQ: {question}\nA: ", end="")
    for token in generate_answer(question, chunks):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    user_query = input("Ask a Question: ")
    ask_rag(user_query)
