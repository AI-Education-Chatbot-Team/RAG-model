# Python RAG Chatbot
This is a RAG Chatbot made in Python for the AI Education Team

## Dependencies
- Supabase - Database/Document storage
- Groq - LLM for core RAG model
- sentence-transformers - Generate embeddings through HuggingFace models
- python-dotenv - Loads environment variables from .env
- streamlit - frontend webview
- pypdf - Read and extract text from pdfs

Set up venv:<br>
`python3 -m venv myenv`<br>
`source myenv/bin/activate`

Install dependencies:<br>
`pip install supabase groq sentence-transformers python-dotenv`

To install dependencies run:<br>
`pip install -r requirements.txt`

## Loading New Documents
To load new documents into Supabase run:<br>
`python3 ingest.py`

To run chatbot in terminal run:<br>
`python3 main.py`

To run localhost run:<br>
`streamlit run app.py`