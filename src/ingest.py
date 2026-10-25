# ingest.py
import os
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
