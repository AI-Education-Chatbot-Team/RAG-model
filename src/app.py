import os
import tempfile
import uuid

import streamlit as st

from shared import list_session_files
from main import (
    retrieve_context,
    condense_question,
    generate_answer,
    generate_summary,
    is_summary_request,
    resolve_summary_files,
)
from ingest import (
    extract_and_chunk,
    embed_and_store,
    delete_session_documents,
    delete_file_documents,
    ingest_url,
)

st.title("🤖 RAG Knowledge Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

SESSION_ID = st.session_state.session_id


def clear_chat():
    st.session_state.messages = []


def remove_file(source_file: str):
    delete_file_documents(SESSION_ID, source_file)


def remove_all_documents():
    delete_session_documents(SESSION_ID)


# Sidebar
with st.sidebar:
    st.title("Chat Management")
    st.caption("AI can make mistakes. Verify important information.")

    st.subheader("Documents")
    session_files = list_session_files(SESSION_ID)
    if not session_files:
        st.caption("No documents uploaded yet.")
    for i, f in enumerate(session_files):
        c1, c2 = st.columns([5, 1])
        c1.write(f"📄 {f['source_file']}")
        c1.caption(f"{f['chunk_count']} chunks")
        c2.button(
            "✕",
            key=f"rm_{i}",
            on_click=remove_file,
            args=(f["source_file"],),
            help=f"Remove {f['source_file']}",
        )

    st.divider()
    st.button("Clear chat", on_click=clear_chat)
    if session_files:
        st.button("Remove all documents", on_click=remove_all_documents)

    st.divider()
    with st.form("add_url_form", clear_on_submit=True):
        url = st.text_input("Add a web page", placeholder="https://example.com/article")
        add_url = st.form_submit_button("Add URL")
    if add_url and url.strip():
        try:
            with st.spinner("Reading page..."):
                ingest_url(url, session_id=SESSION_ID)
        except ValueError as e:
            st.error(str(e))
        except Exception:
            st.error("Couldn't read that page.")
        else:
            st.rerun()


# Helper Functions
def render_sources(sources_list):
    if not sources_list:
        return
    with st.expander("Sources"):
        for s in sources_list:
            name = s.get("source_file", "Unknown")
            if s.get("type") == "Full Document Summary":
                st.markdown(f"- **{name}** (full document summary)")
            else:
                score = s.get("score")
                score_str = f" (score: {score:.3f})" if score is not None else ""
                st.markdown(f"- **{name}**{score_str}")
                content = s.get("content") or ""
                if content:
                    st.caption(content[:300] + ("…" if len(content) > 300 else ""))


def process_upload(file) -> int:
    ext = os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file.getbuffer())
        temp_path = tmp.name
    try:
        chunks = extract_and_chunk(temp_path, file.name)
        if not chunks:
            return 0
        embed_and_store(chunks, session_id=SESSION_ID)
        return len(chunks)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

prompt_data = st.chat_input(
    "Ask a question or upload a document...",
    accept_file="multiple",
    file_type=["pdf", "docx", "pptx", "txt", "md", "csv", "xlsx"],
)

if prompt_data:
    user_text = prompt_data.text
    uploaded_files = prompt_data.files
    ingested_something = False

    # Uploads
    for file in uploaded_files or []:
        with st.status(f"Processing {file.name}...", expanded=False) as status:
            try:
                n = process_upload(file)
                if n:
                    status.update(
                        label=f"Ingested {n} chunks from {file.name}", state="complete"
                    )
                    ingested_something = True
                else:
                    status.update(
                        label=f"No extractable text found in {file.name}", state="error"
                    )
            except Exception as e:
                status.update(
                    label=f"Failed to process {file.name}: {e}", state="error"
                )

    # Question
    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)

        history = list(st.session_state.messages)  # prior turns only
        st.session_state.messages.append({"role": "user", "content": user_text})

        with st.chat_message("assistant"):
            try:
                if is_summary_request(user_text):
                    files_to_summarize = resolve_summary_files(SESSION_ID, user_text)
                    response = st.write_stream(
                        generate_summary(SESSION_ID, files_to_summarize)
                    )
                    sources_to_save = [
                        {"source_file": f, "type": "Full Document Summary"}
                        for f in files_to_summarize
                    ]
                else:
                    search_query = condense_question(user_text, history)
                    chunks = retrieve_context(search_query, session_id=SESSION_ID)
                    has_documents = bool(list_session_files(SESSION_ID))

                    response = st.write_stream(
                        generate_answer(
                            user_text,
                            chunks,
                            history=history,
                            has_documents=has_documents,
                        )
                    )
                    sources_to_save = [
                        {
                            "source_file": c.get("source_file"),
                            "score": c.get("score"),
                            "content": (c.get("content") or "")[:300],
                        }
                        for c in chunks
                    ]

                render_sources(sources_to_save)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "sources": sources_to_save,
                    }
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.session_state.messages.pop()

    # Refresh
    if ingested_something:
        st.rerun()
