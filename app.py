import streamlit as st
import uuid

from main import retrieve_context, generate_answer
from ingest import cleanup_session_data, process_uploaded_files

st.title("🤖 RAG Knowledge Assistant")
st.caption("AI can make mistakes")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Chat Management")
    st.caption("Session: " + st.session_state.session_id)

    if st.button("Clear Chat"):
        cleanup_session_data(st.session_state.session_id)
        st.session_state.messages = []
        st.success("Session data deleted from database.")
        st.rerun()

# Display previous chat messages
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
        if message.get("faithfulness_score") is not None:
            st.caption(f"Faithfulness score: {message['faithfulness_score']:.0%}")

# Native chat input with built-in attachment button
prompt_data = st.chat_input(
    "Ask a question or upload a document...",
    accept_file=True,
    file_type=["pdf", "txt", "md", "docx", "csv"],
)

if prompt_data:
    user_text = prompt_data.text
    uploaded_files = prompt_data.files

    # 1. Handle uploaded files (if any were attached)
    if uploaded_files:
        for file in uploaded_files:
            st.info(f"Processing uploaded file: {file.name}")
            try:
                process_uploaded_files(file, file.name, st.session_state.session_id)
            except ValueError as e:
                st.error(str(e))
                continue
            # chunks_with_meta = chunk_pdf_with_metadata(file, source_file=file.name)
            # embed_and_store(chunks_with_meta, st.session_state.session_id)
            st.success(f"Ingested {file.name}")

    # 2. Process chat prompt if text was provided
    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)
        st.session_state.messages.append({"role": "user", "content": user_text})

        # Retrieve chunks up front so we have sources + context for both
        # the answer and the faithfulness check
        chunks = retrieve_context(user_text, st.session_state.session_id)

        with st.chat_message("assistant"):
            response = st.write_stream(generate_answer(user_text, chunks, st.session_state.session_id))

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
            "sources": chunks,
        })