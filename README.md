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

### URL Ingestion Testing – PR #41
**What:** Manually tested URL ingestion locally using Streamlit. **13 tests completed.**
**Why:** To verify website uploads, summaries, removal, and error handling.
**How:** Uploaded the Auburn University website and tested the chatbot's responses.
**Passed:**
- Website upload, question answering, and summarization.
- Website removal from Documents.
- Invalid and inaccessible URL handling.
- Fresh session correctly showed no documents.
- Failed uploads were not added to Documents.
**Issues Found:**
- After removing the website and clearing chat, the chatbot still claimed it had access to it.
- Incorrectly identified the website as a PDF and couldn't provide its URL.

  
### Chatbot Testing 
**What:** Manually tested the updated chatbot locally using TXT, Markdown, and PDF files. **12 tests: 10 passed, 2 failed.**
**Why:** To verify answer accuracy, document summarization, and source references.
**How:** Uploaded documents, asked questions, and compared responses with the original files.
**Passed (10):**
1. `Luna.txt` – Correct favorite food.
2. Reworded question – Consistent answer.
3. Missing information – Correctly handled.
4. Full TXT summary – Accurate.
5. Sunny question – Correct answer.
6. `Rocky.md` – Correct favorite toy.
7. Multiple documents – Correct answers.
8. PDF question – Correct SHA-256 answer.
9. PDF summary – Accurate main features.
10. Security Log Analyzer – Correct explanation.
**Failed (2):**
1. **TXT:** Focused summary about Sunny included unrelated information.
2. **PDF:** Focused summary about the Security Log Analyzer included unrelated sections.


### Testing Similarity Score (PR #37)
**Author:** Shamyla Taylor
- **What:** Tested the chatbot's similarity scores, document retrieval, and duplicate-source handling.
- **Why:** To verify that the chatbot displays similarity scores, retrieves accurate information, and avoids listing the same document multiple times.
- **How:** Ran the chatbot locally using different documents and questions. Verified that similarity scores appeared, answers were accurate, and duplicate sources were not displayed during testing.
  
### Chatbot Testing and Bug Identification (PR #38)
**Author:** Shamyla Taylor
- **What:** Tested the chatbot's accuracy, document retrieval, and ability to understand different question phrasing.
- **Why:** To verify that the chatbot answers questions correctly and identify issues before changes are merged.
- **How:** Tested TXT, MD, and DOCX files locally using different questions and unusual wording. Found an issue where asking for a summary about Sandy Cheeks returned a summary of the entire document instead.
  
### 2026-10-06: Fix Permission Issue with Streamlit Cloud
**Author:** Robbie Lee
- **What:** Moved from Docling to MarkItDown because Docling was causing a permission error trying to write to streamlit cloud.
- **Why:** Docling was trying to create models and documents in streamlit cloud, however, the streamlit cloud container is write only.
- **How:** To get around Streamlit's write only, we moved to MarkItDown. Using this streamlit does not throw a permissions error and parses many more document types to markdown.

### 2026-10-05: Refactor Ingestion
**Author:** Robbie Lee
- **What:** Updated file ingestion to use Docling for better parsing and LLM reading. Also refactored much of the code base removing unused functions
- **Why:** Before our file parsing was grabbing raw text which resulted in a lot of data loss.
- **How:** Imported docling and used built in converter to convert file types to markdown.

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

## Open Issues

- **#28:** Live website has permission issues when deployed.
