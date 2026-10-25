"""
App flows the basic tests don't reach: file uploads, summaries, and the
"nothing found" fallback. File uploads go through the `submit` fixture in
conftest.py because AppTest can't attach files to st.chat_input.
"""
import os
from types import SimpleNamespace

from conftest import FakeUpload

# Chunks shorter than ingest.MIN_CHUNK_CHARS (20) are dropped, so keep test text longer
LONG_TEXT = "Notes text that is long enough to survive the minimum chunk size."


def test_upload_ingests_file_into_session(fakes, submit):
    fakes.converter.convert.return_value.text_content = LONG_TEXT
    insert = fakes.supabase.table.return_value.insert
    insert.return_value.execute.return_value.data = [{"id": 1}]

    at = submit(files=[FakeUpload("notes.pdf")])

    assert not at.exception
    rows = insert.call_args.args[0]  # embed_and_store inserts a list of rows
    assert len(rows) == 1
    assert rows[0]["session_id"] == at.session_state.session_id
    assert rows[0]["source_file"] == "notes.pdf"


def test_upload_passes_a_real_temp_file_with_extension(fakes, submit):
    seen = {}

    def fake_convert(path):
        with open(path, "rb") as f:
            seen["data"] = f.read()
        seen["path"] = path
        return SimpleNamespace(text_content=LONG_TEXT)

    fakes.converter.convert.side_effect = fake_convert

    submit(files=[FakeUpload("lecture.docx", b"docx bytes")])

    assert seen["path"].endswith(".docx")
    assert seen["data"] == b"docx bytes"


def test_upload_removes_temp_file_afterwards(fakes, submit):
    paths = []
    fakes.converter.convert.side_effect = lambda p: (
        paths.append(p) or SimpleNamespace(text_content=LONG_TEXT)
    )

    submit(files=[FakeUpload("a.txt")])

    assert paths and not os.path.exists(paths[0])


def test_upload_with_no_text_stores_nothing(fakes, submit):
    fakes.converter.convert.return_value.text_content = "  "

    at = submit(files=[FakeUpload("blank.pdf")])

    assert not at.exception
    fakes.supabase.table.return_value.insert.assert_not_called()


def test_unreadable_upload_shows_error_instead_of_crashing(fakes, submit):
    fakes.converter.convert.side_effect = RuntimeError("Conversion failed")

    at = submit(files=[FakeUpload("broken.pdf")])

    assert not at.exception
    fakes.supabase.table.return_value.insert.assert_not_called()


def test_summary_request_summarizes_the_session_document(fakes, submit):
    fakes.set_rpc(list_session_files=[{"source_file": "ch3.pdf", "chunk_count": 1}])
    chunk_query = (
        fakes.supabase.table.return_value.select.return_value.eq.return_value
        .eq.return_value.order.return_value.order.return_value.range.return_value
    )
    chunk_query.execute.return_value.data = [{"content": "Chapter three text"}]

    at = submit(text="Summarize this document")

    assert not at.exception
    last = at.session_state.messages[-1]
    assert last["content"] == "Hello world"
    assert last["sources"] == [{"source_file": "ch3.pdf", "type": "Full Document Summary"}]
    prompt = fakes.groq.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Chapter three text" in prompt
    # a summary reads the whole file; it must not run a similarity search
    assert all(c.args[0] != "hybrid_search" for c in fakes.supabase.rpc.call_args_list)


def test_summary_request_without_documents_asks_for_upload(fakes, submit):
    at = submit(text="Summarize this document")

    assert not at.exception
    assert "Upload one first" in at.session_state.messages[-1]["content"]
    fakes.groq.chat.completions.create.assert_not_called()


def test_question_searches_current_session_only(fakes, submit):
    fakes.set_rpc(
        hybrid_search=[{"content": "hit", "source_file": "a.pdf", "score": 0.8}],
        list_session_files=[{"source_file": "a.pdf", "chunk_count": 1}],
    )

    at = submit(text="What is ATP?")

    searches = [c for c in fakes.supabase.rpc.call_args_list if c.args[0] == "hybrid_search"]
    assert len(searches) == 1
    assert searches[0].args[1]["filter_session_id"] == at.session_state.session_id


def test_question_with_no_matches_tells_the_model_nothing_was_found(fakes, submit):
    fakes.set_rpc(
        hybrid_search=[],
        list_session_files=[{"source_file": "a.pdf", "chunk_count": 1}],
    )

    submit(text="What is ATP?")

    prompt = fakes.groq.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "couldn't find" in prompt