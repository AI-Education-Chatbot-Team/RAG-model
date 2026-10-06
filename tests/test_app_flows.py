"""
App flows the basic tests don't reach: file uploads, summaries, and the
session fallback. AppTest can't attach files to st.chat_input, so the upload
tests swap st.chat_input for a fake that returns a file.
"""
import os
from types import SimpleNamespace

import pytest
import streamlit
from streamlit.testing.v1 import AppTest

APP_PATH = "../app.py"


class FakeUpload:
    def __init__(self, name, data=b"file contents"):
        self.name = name
        self._data = data

    def getbuffer(self):
        return memoryview(self._data)


@pytest.fixture
def submit(monkeypatch):
    """Run the app as if the user submitted text and/or files in the chat box."""
    def _submit(text=None, files=None):
        value = SimpleNamespace(text=text, files=files or [])
        monkeypatch.setattr(streamlit, "chat_input", lambda *a, **k: value)
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        return at
    return _submit


def test_upload_ingests_file_into_session(fakes, submit):
    fakes.converter.convert.return_value.document.export_to_markdown.return_value = "Notes text"
    fakes.supabase.table.return_value.insert.return_value.execute.return_value.data = [{"id": 1}]

    at = submit(files=[FakeUpload("notes.pdf")])

    assert not at.exception
    assert any("Ingested 1 chunks from notes.pdf" in s.value for s in at.success)
    row = fakes.supabase.table.return_value.insert.call_args.args[0]
    assert row["session_id"] == at.session_state.session_id
    assert row["source_file"] == "notes.pdf"


def test_upload_passes_a_real_temp_file_with_extension(fakes, submit):
    seen = {}

    def fake_convert(path):
        with open(path, "rb") as f:
            seen["data"] = f.read()
        seen["path"] = path
        result = SimpleNamespace(document=SimpleNamespace(export_to_markdown=lambda: "x"))
        return result

    fakes.converter.convert.side_effect = fake_convert

    submit(files=[FakeUpload("lecture.docx", b"docx bytes")])

    assert seen["path"].endswith(".docx")
    assert seen["data"] == b"docx bytes"


def test_upload_removes_temp_file_afterwards(fakes, submit):
    paths = []
    fakes.converter.convert.side_effect = lambda p: paths.append(p) or SimpleNamespace(
        document=SimpleNamespace(export_to_markdown=lambda: "x")
    )

    submit(files=[FakeUpload("a.txt")])

    assert paths and not os.path.exists(paths[0])


def test_upload_with_no_text_shows_warning(fakes, submit):
    fakes.converter.convert.return_value.document.export_to_markdown.return_value = "  "

    at = submit(files=[FakeUpload("blank.pdf")])

    assert not at.exception
    assert any("No extractable text" in w.value for w in at.warning)
    fakes.supabase.table.return_value.insert.assert_not_called()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="BUG: a file docling can't read crashes the app instead of showing an error")
def test_unreadable_upload_shows_error_instead_of_crashing(fakes, submit):
    fakes.converter.convert.side_effect = RuntimeError("Conversion failed")

    at = submit(files=[FakeUpload("broken.pdf")])

    assert not at.exception


def test_summary_request_uses_most_recent_document(fakes, submit):
    fakes.supabase.table.return_value.select.return_value.eq.return_value \
        .order.return_value.limit.return_value.execute.return_value.data = [{"source_file": "ch3.pdf"}]
    fakes.supabase.table.return_value.select.return_value.eq.return_value \
        .eq.return_value.order.return_value.order.return_value.execute.return_value.data = [
            {"content": "Chapter three text"},
        ]

    at = submit(text="Summarize this document")

    assert not at.exception
    last = at.session_state.messages[-1]
    assert last["content"] == "Hello world"
    assert last["sources"] == [{"source_file": "ch3.pdf", "type": "Full Document Summary"}]
    prompt = fakes.groq.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Chapter three text" in prompt
    fakes.supabase.rpc.assert_not_called()


def test_question_searches_current_session_first(fakes, submit):
    fakes.supabase.rpc.return_value.execute.return_value.data = [
        {"content": "hit", "source_file": "a.pdf", "page_number": 1, "similarity": 0.8},
    ]

    at = submit(text="What is ATP?")

    assert fakes.supabase.rpc.call_count == 1
    params = fakes.supabase.rpc.call_args.args[1]
    assert params["filter_session_id"] == at.session_state.session_id


def test_question_with_no_session_matches_falls_back_to_unfiltered_search(fakes, submit):
    # Documents what the app does today. Whether filter_session_id=None searches
    # every user's documents depends on the match_documents SQL function in
    # Supabase. If it does, this conflicts with Issue #9.
    fakes.supabase.rpc.return_value.execute.return_value.data = []

    submit(text="What is ATP?")

    assert fakes.supabase.rpc.call_count == 2
    assert fakes.supabase.rpc.call_args_list[1].args[1]["filter_session_id"] is None
