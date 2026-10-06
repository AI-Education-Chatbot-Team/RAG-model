# Python RAG Chatbot
This is a RAG Chatbot made in Python for the AI Education Team
`https://senior-design-chatbot.streamlit.app/`

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

## Running Tests
Install pytest:<br>
`pip install pytest`

Run the fast tests from the repo root:<br>
`pytest`

Run the slower integration tests, which use the real docling parser and embedding model on real files (needs `pip install reportlab python-docx`):<br>
`pytest -m integration`

Tests marked `xfail` are known bugs. When a bug is fixed, its test reports `XPASS` and fails; remove the `xfail` marker from that test.

The same tests run automatically on every pull request into `main` (see `.github/workflows/streamlit-ci.yml`).
