import os
import tempfile

os.environ["RAPIDOCR_MODEL_DIR"] = os.path.join(tempfile.gettempdir(), "rapidocr_models")

import uuid
import streamlit as st

from main import (
    retrieve_context, 
    generate_answer, 
    generate_summary, 
    is_summary_request,
    get_most_recent_source_file
)
from ingest import (
    extract_and_chunk,
    embed_and_store,
    delete_session_documents,
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

def render_sources(sources_list):
    if not sources_list:
        return
    with st.expander("Sources"):
        for s in sources_list:
            if s.get("type") == "Full Document Summary":
                st.markdown(
                    f"- **{s.get('source_file', 'Unknown')}** "
                    f"(type: Full Document Summary)"
                )
            else:
                sim = s.get("similarity")
                sim_str = f"{sim:.2f}" if sim is not None else "1.00"
                st.markdown(
                    f"- **{s.get('source_file', 'Unknown')}**, p.{s.get('page_number', 'N/A')} "
                    f"(similarity: {sim_str})"
                )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

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

            # 1. Save uploaded file buffer to a temporary file in /tmp
            file_ext = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(file.getbuffer())
                temp_path = tmp_file.name

            try:
                # 2. Pass the local file path from /tmp to ingest.py
                chunks_with_meta = extract_and_chunk(
                    file_path=temp_path, 
                    source_file_name=file.name
                )

                if chunks_with_meta:
                    embed_and_store(chunks_with_meta, session_id=st.session_state.session_id)
                    st.success(f"Ingested {len(chunks_with_meta)} chunks from {file.name}")
                else:
                    st.warning("No extractable text found in file.")

            finally:
                # 3. Clean up the temporary file from /tmp
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)
        st.session_state.messages.append({"role": "user", "content": user_text})

        is_summary = is_summary_request(user_text)

        with st.chat_message("assistant"):
            if is_summary:
                source_file = get_most_recent_source_file(st.session_state.session_id)
                response = st.write_stream(generate_summary(st.session_state.session_id))

                sources_to_save = (
                    [{"source_file": source_file, "type": "Full Document Summary"}]
                    if source_file
                    else []
                )
            else:
                chunks = retrieve_context(user_text, session_id=st.session_state.session_id)
                if not chunks:
                    chunks = retrieve_context(user_text, session_id=None)

                response = st.write_stream(generate_answer(user_text, chunks))
                sources_to_save = chunks

            render_sources(sources_to_save)
        
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "sources": sources_to_save,
                }
            )
