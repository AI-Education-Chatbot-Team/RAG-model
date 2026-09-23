import streamlit as st

from main import retrieve_context, generate_answer, score_faithfulness
from ingest import chunk_pdf_with_metadata, embed_and_store

st.title("🤖 RAG Knowledge Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for s in message["sources"]:
                    st.markdown(
                        f"- **{s['source_file']}**, p.{s['page_number']} "
                        f"(similarity: {s['similarity']:.2f})"
                    )
        if message.get("faithfulness_score") is not None:
            st.caption(f"Faithfulness score: {message['faithfulness_score']:.0%}")

# Native chat input with built-in attachment button
prompt_data = st.chat_input(
    "Ask a question or upload a document...",
    accept_file=True,
    file_type=["pdf"],  # Only PDFs supported for now
)

if prompt_data:
    user_text = prompt_data.text
    uploaded_files = prompt_data.files

    # 1. Handle uploaded files (if any were attached)
    if uploaded_files:
        for file in uploaded_files:
            st.info(f"Processing uploaded file: {file.name}")
            chunks_with_meta = chunk_pdf_with_metadata(file, source_file=file.name)
            embed_and_store(chunks_with_meta)
            st.success(f"Ingested {len(chunks_with_meta)} chunks from {file.name}")

    # 2. Process chat prompt if text was provided
    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)
        st.session_state.messages.append({"role": "user", "content": user_text})

        # Retrieve chunks up front so we have sources + context for both
        # the answer and the faithfulness check
        chunks = retrieve_context(user_text)

        with st.chat_message("assistant"):
            response = st.write_stream(generate_answer(user_text, chunks))

            with st.expander("Sources"):
                for c in chunks:
                    st.markdown(
                        f"- **{c['source_file']}**, p.{c['page_number']} "
                        f"(similarity: {c['similarity']:.2f})"
                    )

            with st.spinner("Scoring faithfulness..."):
                faithfulness = score_faithfulness(response, chunks)
            st.caption(f"Faithfulness score: {faithfulness['faithfulness_score']:.0%}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": chunks,
            "faithfulness_score": faithfulness["faithfulness_score"],
        })