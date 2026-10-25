# main.py
import math
import os
import re

import groq
from sentence_transformers import CrossEncoder

from shared import (
    LLM_MODEL,
    DOCUMENTS_TABLE,
    supabase,
    groq_client,
    embed_texts,
    list_session_files,
)

# Config
TOP_K = 6
RERANK_CANDIDATES = 20
USE_RERANKER = os.getenv("USE_RERANKER", "0") == "1"  # set USE_RERANKER=1 to enable

DIRECT_SUMMARY_CHAR_LIMIT = 16_000
MAP_GROUP_CHARS = 10_000
MAX_MAP_GROUPS = 10

HISTORY_TURNS = 6
HISTORY_MAX_CHARS = 1500

RATE_LIMIT_MSG = "I'm being rate limited by the model provider right now. Please wait a few seconds and try again."
API_ERROR_MSG = (
    "Something went wrong while contacting the language model. Please try again."
)

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


# LLM helpers
def _stream(messages: list[dict], max_tokens: int = 3000, temperature: float = 0.2):
    """Streams visible content; turns provider errors into friendly messages."""
    try:
        completion = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            reasoning_effort="low",
            stream=True,
        )
        for chunk in completion:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except groq.RateLimitError:
        yield RATE_LIMIT_MSG
    except groq.APIError:
        yield API_ERROR_MSG


def _complete(
    messages: list[dict], max_tokens: int = 1500, temperature: float = 0.2
) -> str:
    """Non-streaming call. Raises on provider errors (callers handle)."""
    r = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        reasoning_effort="low",
    )
    return (r.choices[0].message.content or "").strip()


def _history_messages(history: list[dict] | None) -> list[dict]:
    out = []
    for m in (history or [])[-HISTORY_TURNS:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            out.append({"role": m["role"], "content": m["content"][:HISTORY_MAX_CHARS]})
    return out


# Retrieval
def condense_question(question: str, history: list[dict] | None) -> str:
    """Rewrites a follow-up into a standalone search query using recent history."""
    hist = _history_messages(history)
    if not hist:
        return question
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in hist)
    try:
        rewritten = _complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's last message as a standalone search query, using the "
                        "conversation for context. Resolve pronouns and references. "
                        "Output only the query, nothing else."
                    ),
                },
                {"role": "user", "content": f"{convo}\n\nLast message: {question}"},
            ],
            max_tokens=400,
            temperature=0,
        )
        return rewritten or question
    except Exception:
        return question


def retrieve_context(
    query: str, session_id: str | None, top_k: int = TOP_K
) -> list[dict]:
    """Hybrid (vector + full-text) search scoped to ONE session. No session -> no results."""
    if not session_id:
        return []

    fetch = RERANK_CANDIDATES if USE_RERANKER else top_k
    rows = (
        supabase.rpc(
            "hybrid_search",
            {
                "query_text": query,
                "query_embedding": embed_texts([query])[0],
                "filter_session_id": session_id,
                "match_count": fetch,
            },
        )
        .execute()
        .data
        or []
    )

    if USE_RERANKER and rows:
        scores = _get_reranker().predict([(query, r["content"]) for r in rows])
        for r, s in zip(rows, scores):
            r["score"] = float(s)
        rows.sort(key=lambda r: r["score"], reverse=True)
        rows = rows[:top_k]

    return rows


def _build_context_str(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[Source: {c.get('source_file', 'Unknown')}]\n{c.get('content', '')}"
        for c in chunks
    )


# Formatting
DOCUMENT_EXTS = {".pdf", ".docx", ".md", ".txt"}
TABULAR_EXTS = {".csv", ".xlsx"}
SLIDE_EXTS = {".pptx"}

_DOC_STYLE = (
    "Prose documents (reports, resumes, articles): structure the answer like a document. "
    "For multi-part answers use short markdown headers followed by body text or bullet points; "
    "use bullets for lists of items (skills, requirements, steps, findings); "
    "for a simple factual question, answer in one or two plain sentences with no headers."
)
_TABLE_STYLE = (
    "Spreadsheet/table data: keep tabular answers as a markdown table that reuses the source's "
    "exact column names, includes only the relevant rows and columns, and never converts rows "
    "into bullet points. For a single value, answer in a sentence and mention the row/column it "
    "came from. Do not invent or alter cell values."
)
_SLIDE_STYLE = (
    "Slide decks: organize by slide or topic using short headers, with concise bullet points "
    "under each, mirroring how the deck presents the material."
)


def _format_guidance(source_files: list[str]) -> str:
    """Builds formatting instructions from the file types being drawn on."""
    exts = {os.path.splitext(f)[1].lower() for f in source_files if f}
    parts = []
    if exts & TABULAR_EXTS:
        parts.append(_TABLE_STYLE)
    if exts & DOCUMENT_EXTS:
        parts.append(_DOC_STYLE)
    if exts & SLIDE_EXTS:
        parts.append(_SLIDE_STYLE)
    if not parts:
        return ""
    return (
        "Formatting (match the format of the source material; use markdown):\n- "
        + "\n- ".join(parts)
        + "\n- If the user explicitly asks for a particular format (table, bullets, paragraph, etc.), "
        "follow their request instead.\n- When sources of different types are mixed, format each "
        "part of the answer according to the type of source it came from."
    )


# Answers
def generate_answer(
    question: str,
    chunks: list[dict],
    history: list[dict] | None = None,
    has_documents: bool = False,
):
    context = _build_context_str(chunks)
    format_rules = _format_guidance(list({c.get("source_file") for c in chunks}))

    if context.strip():
        system_prompt = f"""You are a helpful assistant that answers questions using ONLY the document context below. Do not use outside knowledge.

Rules:
- If the answer is not in the context, say so plainly. Do not guess or fill gaps.
- Mention the source file name when it helps the user know where something came from.
- Use the earlier conversation only to understand what the user is referring to, not as a source of facts.

{format_rules}

Context:
{context}"""
    elif has_documents:
        system_prompt = """You are a helpful assistant that answers questions about the user's uploaded documents.
No relevant passages were found for this question. Tell the user you couldn't find that in their uploaded documents and suggest rephrasing or asking about something the documents cover. Do not answer from outside knowledge."""
    else:
        system_prompt = """You are a friendly assistant whose purpose is to answer questions about documents the user uploads.
You can chat casually (greetings, pleasantries), but you have NO access to outside facts or general knowledge.
If the user asks a knowledge question and no documents are uploaded, politely explain you can't answer without an uploaded document and invite them to upload one."""

    messages = [{"role": "system", "content": system_prompt}]
    # `history` should NOT include the current question
    messages += _history_messages(history)
    messages.append({"role": "user", "content": question})

    yield from _stream(messages, temperature=0.1)


# Summaries
SUMMARY_KEYWORDS = [
    "summarize",
    "summarise",
    "summary",
    "overview",
    "main points",
    "tl;dr",
    "tldr",
    "what is this document about",
    "what's this document about",
    "what is this about",
    "key takeaways",
]
SCOPED_WORDS = re.compile(
    r"\b(section|chapter|page|paragraph|table|figure|part|slide|row|column)\b"
)


def is_summary_request(query: str) -> bool:
    """Only whole-document summary asks: keyword present, short, and not scoped to a sub-part."""
    q = query.lower().strip()
    if len(q.split()) > 14 or SCOPED_WORDS.search(q):
        return False
    return any(k in q for k in SUMMARY_KEYWORDS)


def resolve_summary_files(session_id: str, query: str) -> list[str]:
    """Files named in the query; otherwise every file in the session."""
    files = [f["source_file"] for f in list_session_files(session_id)]
    q = query.lower()
    named = [f for f in files if f.lower() in q or os.path.splitext(f)[0].lower() in q]
    return named or files


def get_document_chunks(session_id: str, source_file: str) -> list[str]:
    """All chunk texts for a file in document order (paginated past the 1000-row API cap)."""
    texts, start, page = [], 0, 1000
    while True:
        resp = (
            supabase.table(DOCUMENTS_TABLE)
            .select("content")
            .eq("session_id", session_id)
            .eq("source_file", source_file)
            .order("chunk_index")
            .order("id")
            .range(start, start + page - 1)
            .execute()
        )
        rows = resp.data or []
        texts.extend(r["content"] for r in rows)
        if len(rows) < page:
            break
        start += page
    return texts


def _group_chunks(chunks: list[str], max_chars: int) -> list[str]:
    groups, current, size = [], [], 0
    for c in chunks:
        if current and size + len(c) > max_chars:
            groups.append("\n".join(current))
            current, size = [], 0
        current.append(c)
        size += len(c) + 1
    if current:
        groups.append("\n".join(current))
    return groups


_SUMMARY_RULES = (
    "Cover the main points and overall purpose. Do not infer anything outside of the "
    "document. Do not mention chunks or excerpts; write as if you read the whole document."
)


def _summary_format(source_file: str) -> str:
    ext = os.path.splitext(source_file)[1].lower()
    if ext in TABULAR_EXTS:
        return (
            "Format: one or two sentences on what the dataset contains, then a markdown table "
            "of the columns (name and what it holds) or key figures, then a few bullets on notable patterns. "
            "Only state what is visible in the data."
        )
    if ext in SLIDE_EXTS:
        return "Format: a one-line overview, then a short header per slide or topic with concise bullets."
    return (
        "Format: a one-sentence overview, then short markdown headers for the main sections "
        "with bullet points (or brief body text) under each."
    )


def _summarize_one(session_id: str, source_file: str):
    chunks = get_document_chunks(session_id, source_file)
    full_text = "\n".join(chunks)

    if not full_text.strip():
        yield f"I couldn't find any readable content for {source_file}."
        return

    # Small enough: single pass
    if len(full_text) <= DIRECT_SUMMARY_CHAR_LIMIT:
        yield from _stream(
            [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. Provide a clear, concise summary of the following document. {_SUMMARY_RULES} {_summary_format(source_file)}\n\nDocument: {source_file}\n\nContent:\n{full_text}",
                },
                {"role": "user", "content": "Please summarize this document."},
            ]
        )
        return

    # Large: map-reduce
    groups = _group_chunks(chunks, MAP_GROUP_CHARS)
    if len(groups) > MAX_MAP_GROUPS:
        step = math.ceil(len(groups) / MAX_MAP_GROUPS)
        groups = groups[::step]

    partials = []
    for i, g in enumerate(groups, 1):
        partials.append(
            f"Part {i}:\n"
            + _complete(
                [
                    {
                        "role": "system",
                        "content": "Summarize this portion of a larger document in a short paragraph of key points. Do not infer anything outside the text.",
                    },
                    {"role": "user", "content": g},
                ]
            )
        )

    yield from _stream(
        [
            {
                "role": "system",
                "content": f"You are a helpful assistant. Below are summaries of consecutive parts of one document. Combine them into one clear, concise summary of the whole document. {_SUMMARY_RULES} {_summary_format(source_file)}\n\nDocument: {source_file}\n\n"
                + "\n\n".join(partials),
            },
            {"role": "user", "content": "Please summarize this document."},
        ]
    )


def generate_summary(session_id: str, source_files: list[str]):
    if not source_files:
        yield "You haven't uploaded a document in this session yet. Upload one first, then ask me to summarize it."
        return

    try:
        for idx, source_file in enumerate(source_files):
            if len(source_files) > 1:
                yield ("\n\n" if idx else "") + f"### {source_file}\n\n"
            yield from _summarize_one(session_id, source_file)
    except groq.RateLimitError:
        yield "\n\n" + RATE_LIMIT_MSG
    except groq.APIError:
        yield "\n\n" + API_ERROR_MSG
