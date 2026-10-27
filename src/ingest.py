# ingest.py
import ipaddress
import os
import socket
import tempfile
from urllib.parse import urljoin, urlparse

import requests
from markitdown import MarkItDown
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from shared import supabase, embed_texts, DOCUMENTS_TABLE

md = MarkItDown()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MIN_CHUNK_CHARS = 20
INSERT_BATCH = 100
TABULAR_EXTENSIONS = {".csv"}

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def extract_to_markdown(file_path: str) -> str:
    """Converts PDF, DOCX, PPTX, TXT, MD, XLSX, CSV, HTML into Markdown via MarkItDown."""
    return md.convert(file_path).text_content


def _chunk_table(markdown_text: str, max_chars: int = CHUNK_SIZE) -> list[str]:
    """Chunk a markdown table by groups of rows, repeating the header row in every chunk."""
    lines = [l for l in markdown_text.splitlines() if l.strip()]
    if len(lines) < 3:
        return [markdown_text]

    header = "\n".join(lines[:2])  # column names + separator row
    chunks, current, size = [], [], len(header)
    for row in lines[2:]:
        if current and size + len(row) + 1 > max_chars:
            chunks.append(header + "\n" + "\n".join(current))
            current, size = [], len(header)
        current.append(row)
        size += len(row) + 1
    if current:
        chunks.append(header + "\n" + "\n".join(current))
    return chunks


def extract_and_chunk(file_path: str, source_file_name: str) -> list[dict]:
    markdown_text = extract_to_markdown(file_path)
    if not markdown_text.strip():
        return []

    ext = os.path.splitext(source_file_name)[1].lower()
    texts: list[str] = []

    if ext in TABULAR_EXTENSIONS:
        texts = _chunk_table(markdown_text)
    else:
        header_splits = _header_splitter.split_text(markdown_text)
        for split in _text_splitter.split_documents(header_splits):
            header_context = " > ".join(
                v for k, v in split.metadata.items() if k.startswith("Header")
            )
            body = split.page_content
            texts.append(f"[{header_context}]\n{body}" if header_context else body)

    chunks = []
    for text in texts:
        if len(text.strip()) < MIN_CHUNK_CHARS:
            continue
        chunks.append({"text": text, "source_file": source_file_name})
    return chunks


def embed_and_store(
    chunks_with_meta: list[dict], session_id: str | None = None
) -> list[int]:
    """Embeds in one batch, replaces any previous upload of the same file, inserts in batches."""
    if not chunks_with_meta:
        return []

    # Dedupe: re-uploading a file replaces its old chunks instead of doubling them
    if session_id:
        for name in {c["source_file"] for c in chunks_with_meta}:
            supabase.table(DOCUMENTS_TABLE).delete().eq("session_id", session_id).eq(
                "source_file", name
            ).execute()

    embeddings = embed_texts([c["text"] for c in chunks_with_meta])

    rows = [
        {
            "content": c["text"],
            "embedding": emb,
            "source_file": c["source_file"],
            "session_id": session_id,
            "chunk_index": i,
        }
        for i, (c, emb) in enumerate(zip(chunks_with_meta, embeddings))
    ]

    inserted_ids: list[int] = []
    for start in range(0, len(rows), INSERT_BATCH):
        resp = (
            supabase.table(DOCUMENTS_TABLE)
            .insert(rows[start : start + INSERT_BATCH])
            .execute()
        )
        inserted_ids.extend(r["id"] for r in (resp.data or []))

    print(f"Ingested {len(rows)} chunks into Supabase.")
    return inserted_ids

MAX_URL_BYTES = 5_000_000
URL_HEADERS = {
    "User-Agent": "RAG-Knowledge-Assistant/1.0 (student project; https://github.com/AI-Education-Chatbot-Team/RAG-model)"
}


def _check_public_url(url: str) -> str:
    """Allow only http(s) URLs that resolve to public IPs (blocks SSRF)."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Please enter a valid http(s) URL.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        raise ValueError("Couldn't resolve that hostname.")
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise ValueError("That URL points to a non-public address.")
    return parsed.geturl()


def fetch_url_to_file(url: str) -> str:
    """Download a public web page to a temp .html file and return its path."""
    for _ in range(4):  # follow a few redirects, re-checking every hop
        url = _check_public_url(url)
        resp = requests.get(
            url, headers=URL_HEADERS, timeout=10, allow_redirects=False, stream=True
        )
        if not resp.is_redirect:
            break
        url = urljoin(url, resp.headers.get("Location", ""))
    else:
        raise ValueError("Too many redirects.")
    resp.raise_for_status()
    if "html" not in resp.headers.get("Content-Type", ""):
        raise ValueError("That URL doesn't look like a web page.")
    data = resp.raw.read(MAX_URL_BYTES + 1, decode_content=True)
    if len(data) > MAX_URL_BYTES:
        raise ValueError("That page is too large to ingest.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        f.write(data)
    return f.name


def ingest_url(url: str, session_id: str | None = None) -> int:
    """Fetch a web page, chunk it like an uploaded file, and store it."""
    url = url.strip()
    path = fetch_url_to_file(url)
    try:
        chunks = extract_and_chunk(path, url)
    finally:
        os.remove(path)
    if not chunks:
        raise ValueError("No readable text found on that page.")
    embed_and_store(chunks, session_id=session_id)
    return len(chunks)


def delete_documents(ids: list[int]) -> None:
    if not ids:
        return
    supabase.table(DOCUMENTS_TABLE).delete().in_("id", ids).execute()


def delete_file_documents(session_id: str, source_file: str) -> None:
    if not session_id or not source_file:
        return
    supabase.table(DOCUMENTS_TABLE).delete().eq("session_id", session_id).eq(
        "source_file", source_file
    ).execute()


def delete_session_documents(session_id: str) -> int:
    if not session_id:
        return 0
    response = (
        supabase.table(DOCUMENTS_TABLE).delete().eq("session_id", session_id).execute()
    )
    deleted = len(response.data) if response.data else 0
    print(f"Deleted {deleted} chunk(s) for session {session_id}.")
    return deleted
