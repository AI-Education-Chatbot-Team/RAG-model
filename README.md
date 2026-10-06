# RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) chatbot built in Python for the AI Education Chatbot Team.
Users upload documents, the text is chunked and embedded into Supabase, and questions are answered
by a Groq-hosted LLM using only the retrieved document context.

- **Live app:** https://senior-design-chatbot.streamlit.app/
- **Setup, running, and deployment:** see [DEPLOYMENT.md](DEPLOYMENT.md)

This README is the project's documentation and change log. Every meaningful change should be
recorded below with **what** changed, **when**, **why**, and **how**.

---

## How It Works

| File | Purpose |
|------|---------|
| `ingest.py` | Extracts text from PDFs (`pypdf`), splits it into overlapping word chunks (100 words, 20 overlap), embeds each chunk with `all-MiniLM-L6-v2`, and stores it in the Supabase `documents` table with its source file and page number. |
| `main.py` | Embeds the user's question, retrieves the closest chunks via the Supabase `match_documents` RPC, and streams an answer from Groq (`openai/gpt-oss-20b`) grounded only in that context. Also runs from the terminal (`python3 main.py`). |
| `app.py` | Streamlit web UI: chat input with file upload, streamed answers, and a "Sources" list for each answer. |
| `pdfs/` | Sample documents for bulk ingestion with `python3 ingest.py`. |
| `tests/` | Automated `pytest` tests for `main.py`, `ingest.py`, and the Streamlit app. Supabase, Groq, and the ML models are faked, so no API keys are needed. |
| `.github/workflows/streamlit-ci.yml` | CI pipeline: installs dependencies and runs the tests on every pull request into `main`. |

---

## Contribution Guidelines

1. **Keep changes minimal.** Only change what the task requires. A single-line fix should be a single-line diff, not a refactor of the codebase.
2. **One issue per branch/PR.** Link the issue in the PR description (e.g. `Closes #13`).
3. **Log your change.** Add an entry to the [Change Log](#change-log) in the same PR, using the template below.
4. **Write descriptive commit messages.** Avoid messages like `Fix` or `Dev`; say what changed.

### Change Log Entry Template

```markdown
### YYYY-MM-DD: Short title (PR #, Issue #)
**Author:** name
- **What:** What changed, from the user's or developer's point of view.
- **Why:** The problem or requirement that motivated it.
- **How:** Files touched and the approach taken.
```

---

## Change Log

Newest first. Dates are the merge date for PRs or the commit date on `main`.

### 2026-10-06: Streamlit CI pipeline and automated tests (Issue #10)
**Author:** Rett Wilson
- **What:** Every pull request into `main` now runs a GitHub Actions pipeline that installs the dependencies, checks the Python files compile, and runs automated tests. Results show as a check on the PR.
- **Why:** Broken code and dependency problems were only found after merging and deploying (see Issue #28). The pipeline catches them before the merge.
- **How:** Added `.github/workflows/streamlit-ci.yml`, `pytest.ini`, and `tests/`. The tests cover `main.py`, `ingest.py`, and the app itself using Streamlit's `AppTest`. `tests/conftest.py` replaces Supabase, Groq, SentenceTransformer, and docling with fakes so tests run offline without secrets.

### 2026-09-30: Documentation restructure (Issue #13)
**Author:** Rett Wilson
- **What:** Renamed the old `README.md` to `DEPLOYMENT.md` and created this README as the project's documentation and change log.
- **Why:** The previous README only covered setup, so there was no record of what changed, when, or why.
- **How:** `git mv README.md DEPLOYMENT.md` (contents unchanged) and a new `README.md` built from the commit and PR history.

### 2026-09-23: Deployment URL renamed
**Author:** Robbie Lee
- **What:** The Streamlit deployment URL changed from `ai-education-chatbot.streamlit.app` to `senior-design-chatbot.streamlit.app`.
- **Why:** The Streamlit deployment was renamed. (Note: the `dev` branch still lists the old `ai-education-chatbot` URL.)
- **How:** Updated the URL in the README (commit `15cccdb`).

### 2026-09-23: Dev container added, then removed from `main` (PR #8)
**Author:** Robbie Lee
- **What:** A `.devcontainer/devcontainer.json` was merged into `main` and then deleted from `main` in a follow-up commit.
- **Why:** The dev container was meant to give everyone the same development environment. The reason for removing it from `main` was not recorded (commit message: "Fix"). It still exists on the `dev` branch.
- **How:** PR #8 added the file; commit `ff91c9f` removed it.

### 2026-09-23: Source citations, faithfulness scoring, PDF metadata (PRs #6, #7)
**Author:** Dylan Freeman
- **What:** Each answer now lists its sources (file name, page number, similarity score). A "faithfulness score" was added that uses a second LLM call to judge how well the answer is supported by the context.
- **Why:** Users could not see where an answer came from or how reliable it was.
- **How:** `ingest.py` now stores `source_file` and `page_number` per chunk (`chunk_pdf_with_metadata`). `main.py` split into `retrieve_context` / `generate_answer` / `score_faithfulness`. `app.py` shows a "Sources" expander and the score caption.
- **Note:** The faithfulness score is being removed in Issue #12 (see [Pending Changes](#pending-changes)).

### 2026-09-22: Removed committed `__pycache__` (PR #5)
**Author:** Robbie Lee
- **What:** Deleted compiled Python files from the repo.
- **Why:** Build artifacts should not be version controlled.
- **How:** Removed `__pycache__/` and added it and `*.pyc` to `.gitignore` (commit `27a8134`).

### 2026-09-22: Deployment-ready release (PR #4)
**Author:** Robbie Lee
- **What:** PDF upload from the web UI, a smaller LLM, and lower temperature.
- **Why:** Prepare the app for Streamlit Cloud deployment. A smaller model is faster and cheaper, and lower temperature keeps answers closer to the source text.
- **How:** `app.py` chat input accepts PDF attachments. `main.py` model changed `openai/gpt-oss-120b` → `openai/gpt-oss-20b`, temperature `1` → `0.3`. `.gitignore` updated to exclude `.env` and `myenv`.

### 2026-09-16: Streamlit web UI (PR #3)
**Author:** Robbie Lee
- **What:** A browser-based chat interface, and a clearer reply when the answer isn't in the documents.
- **Why:** The terminal-only interface was hard to demo and use.
- **How:** New `app.py` using Streamlit. The system prompt in `main.py` now tells the model to say the answer "is not in the database". The sample PDF was replaced with `Star Wars Characters.pdf`.

### 2026-09-14: PDF ingestion (PR #2)
**Authors:** Dylan Freeman, Robbie Lee
- **What:** Documents can be loaded from PDFs instead of hard-coded text.
- **Why:** Course material is distributed as PDFs.
- **How:** `ingest.py` reads every PDF in `./pdfs` with `pypdf` and chunks the text. `pypdf` added to `requirements.txt`.

### 2026-09-13: Base RAG model (PR #1)
**Author:** Robbie Lee
- **What:** First working RAG pipeline in Python.
- **Why:** Establish the core retrieve-then-generate loop for the project.
- **How:** Supabase (pgvector) for storage and similarity search, `sentence-transformers` for embeddings, Groq for generation. Setup instructions added to the README.

### 2026-09-09 to 2026-09-10: Repository created
**Author:** Robbie Lee
- **What:** Initial commit and directory layout.

---

## Pending Changes

Open PRs that are not yet merged into `main`. Move each one into the Change Log when it merges.

| PR | Branch | Issues | Summary |
|----|--------|--------|---------|
| #15 | `session_memory` → `dev` | #9, #14 | Per-user session IDs for uploaded documents. Documents are cleared on "Clear chat" or after 30 minutes. Supports summary-style questions ("What is this document about?") against the most recently uploaded document. |
| #16 | `dev` → `main` | #11, #12 | Upload support for PDF, TXT, MD, DOCX, and CSV. Removes the faithfulness score. (`.pptx` support and docling-based parsing from #11 are not included.) |

## Open Issues

- **#28:** Live website has permission issues when deployed.
