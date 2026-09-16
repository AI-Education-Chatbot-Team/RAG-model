import streamlit as st

# Import your existing pipeline logic
from main import ask_rag

# Set page title and layout
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Knowledge Assistant")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input from the chat input box
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response with streaming
    with st.chat_message("assistant"):
        # st.write_stream automatically handles Groq's generator/streaming tokens!
        response = st.write_stream(ask_rag(prompt))
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})