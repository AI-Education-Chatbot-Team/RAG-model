import streamlit as st

from main import ask_rag
from ingest import extract_text_from_pdf, chunk_text, embed_and_store

st.title("🤖 RAG Knowledge Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Native chat input with built-in attachment button
prompt_data = st.chat_input(
    "Ask a question or upload a document...",
    accept_file=True,
    file_type=["pdf", "txt"],  # Restrict allowed formats
)

if prompt_data:
    user_text = prompt_data.text
    uploaded_files = prompt_data.files

    # 1. Handle uploaded files (if any were attached)
    if uploaded_files:
        for file in uploaded_files:
            st.info(f"Processing uploaded file: {file.name}")
            all_chunks = []
            raw_text = extract_text_from_pdf(file)
            chunks = chunk_text(raw_text)
            all_chunks.extend(chunks)
            embed_and_store(all_chunks)

    # 2. Process chat prompt if text was provided
    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)
        st.session_state.messages.append(
            {"role": "user", "content": user_text}
        )

        with st.chat_message("assistant"):
            response = st.write_stream(ask_rag(user_text))

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )