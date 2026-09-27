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


def retrieve_context(query: str, session_id: str, match_count: int = 3) -> list[dict]:
    # 1. Embed user query
    query_vector = embed_model.encode(query).tolist()

    # 2. Match documents in Supabase
    rpc_response = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_threshold": 0.3,
        "match_count": match_count,
        "p_session_id": session_id,
    }).execute()

    return rpc_response.data


def _build_context_str(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[Source: {c.get('source_file', 'Unknown')}, p.{c.get('page_number', 'N/A')}]\n{c.get('content', '')}"
        for c in chunks
    )

def is_broad_query(query: str) -> bool:
    """Detects if the query asks for a high-level summary or overview."""
    broad_keywords = [
        "summarize",
        "summary",
        "about",
        "overview",
        "main points",
        "tl;dr",
        "what is this",
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in broad_keywords)

def generate_answer(question: str, chunks: list[dict], session_id: str):
    # Broad Queries: Summaries
    if is_broad_query(question):
        latest_doc = get_latest_summary(session_id)

        if not latest_doc:
            return "No document context available for this session."

        system_prompt = f"""You are a helpful assistant. Use the pre-computed document summary of '{latest_doc['source_file']}' to answer the user's broad question. If you cannot generate a summary reply "Cannot generate a summary try uploading another document."

Document Summary:
{latest_doc['summary']}"""

    # Specific Queries: document facts
    else:
        #Yields the answer text token-by-token

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
        stop=None
    )

    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            yield content


def score_faithfulness(answer: str, chunks: list[dict]) -> dict:
    #Returns a 0-1 faithfulness score
    context = _build_context_str(chunks)

    judge_prompt = f"""You are evaluating whether an AI-generated answer is
faithful to its source context (i.e., not hallucinating).

Context:
{context}

Answer to evaluate:
{answer}

Break the answer into individual factual claims. For each claim, state
whether it is:
- SUPPORTED (directly backed by the context)
- UNSUPPORTED (not found in the context, possibly hallucinated)

Format each line as: CLAIM: <claim text> | VERDICT: <SUPPORTED/UNSUPPORTED>
If the answer makes no factual claims (e.g. "I don't know"), respond with
NO_CLAIMS.
"""

    judge_response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
        max_completion_tokens=1024,
    ).choices[0].message.content

    lines = [l for l in judge_response.split("\n") if "VERDICT:" in l]
    total = len(lines)
    supported = sum(1 for l in lines if "SUPPORTED" in l and "UNSUPPORTED" not in l)

    faithfulness_score = supported / total if total > 0 else None

    return {
        "faithfulness_score": faithfulness_score,
        "total_claims": total,
        "supported_claims": supported,
        "raw_judge_output": judge_response,
    }

def get_latest_summary(session_id: str) -> dict | None:
    response = (
        supabase.table("document_summaries")
        .select("source_file, summary, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if response.data:
        return response.data[0]
    return None

def ask_rag(question: str):
    """
    CLI convenience wrapper: retrieve, generate, print. Kept for
    `python3 main.py` terminal usage.
    """
    chunks = retrieve_context(question)
    print(f"\nQ: {question}\nA: ", end="")
    tokens = []
    for token in generate_answer(question, chunks):
        print(token, end="", flush=True)
        tokens.append(token)
    print()

    full_answer = "".join(tokens)
    result = score_faithfulness(full_answer, chunks)
    if result["faithfulness_score"] is not None:
        print(f"\n[Faithfulness: {result['faithfulness_score']:.0%}]")

def summarize_document(full_text: str, chunk_size: int = 10000) -> str:
    # Split raw text into large chunks
    blocks = [
        full_text[i : i + chunk_size]
        for i in range(0, len(full_text), chunk_size)
    ]

    partial_summaries = []
    for block in blocks[:5]:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize key points from this section:\n\n{block}. Be short and concise with your summaries",
                }
            ],
            temperature=0.2,
        )
        partial_summaries.append(response.choices[0].message.content)

    combined = "\n\n".join(partial_summaries)
    final_response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": f"Synthesize these section summaries into an overall executive summary:\n\n{combined}. Be short and concise. If no summary can be retreived ",
            }
        ],
        temperature=0.3,
    )
    return final_response.choices[0].message.content

if __name__ == "__main__":
    user_query = input("Ask a Question: ")
    ask_rag(user_query)