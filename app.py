import time
import uuid
import streamlit as st

from main import retrieve_context, generate_answer, generate_summary, is_summary_request
from ingest import (
    chunk_file_with_metadata,
    embed_and_store,
    delete_session_documents,
    cleanup_expired_sessions,
)

st.title("🤖 RAG Knowledge Assistant")
st.caption("AI can make mistakes. Verify important information.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Chat Management")
    st.caption(f"Session: {st.session_state.session_id}")

    if st.button("Clear Chat"):
        delete_session_documents(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for s in message["sources"]:
                    st.markdown(
                        f"- **{s.get('source_file', 'Unknown')}**, p.{s.get('page_number', 'N/A')} "
                        f"(similarity: {s['similarity']:.2f})"
                    )

prompt_data = st.chat_input(
    "Ask a question or upload a document...",
    accept_file=True,
    file_type=["pdf", "txt", "md", "docx", "csv"],
)

if prompt_data:
    user_text = prompt_data.text
    uploaded_files = prompt_data.files

    if uploaded_files:
        for file in uploaded_files:
            st.info(f"Processing uploaded file: {file.name}")
            try:
                chunks_with_meta = chunk_file_with_metadata(file, source_file=file.name)
            except ValueError as e:
                st.error(str(e))
                continue
            embed_and_store(chunks_with_meta, session_id=st.session_state.session_id)
            st.success(f"Ingested {len(chunks_with_meta)} chunks from {file.name}")

    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)
        st.session_state.messages.append({"role": "user", "content": user_text})

        is_summary = is_summary_request(user_text)

        with st.chat_message("assistant"):
            if is_summary:
                response = st.write_stream(generate_summary(st.session_state.session_id))
                chunks = []
            else:
                chunks = retrieve_context(user_text, session_id=st.session_state.session_id)
                if not chunks:
                    chunks = retrieve_context(user_text, session_id=None)

                response = st.write_stream(generate_answer(user_text, chunks))

                if chunks:
                    with st.expander("Sources"):
                        for c in chunks:
                            st.markdown(
                                f"- **{c.get('source_file', 'Unknown')}**, p.{c.get('page_number', 'N/A')} "
                                f"(similarity: {c['similarity']:.2f})"
                            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": chunks if not is_summary else None,
        })
