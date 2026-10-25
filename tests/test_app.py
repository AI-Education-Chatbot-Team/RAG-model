def _button(at, label):
    return next(b for b in at.sidebar.button if b.label == label)


def test_app_loads_without_errors(fakes, run_app):
    at = run_app()

    assert not at.exception
    assert at.main.title[0].value == "🤖 RAG Knowledge Assistant"
    assert "session_id" in at.session_state


def test_empty_session_shows_no_documents_message(fakes, run_app):
    at = run_app()

    assert any("No documents uploaded yet" in c.value for c in at.sidebar.caption)


def test_question_shows_streamed_answer(fakes, submit):
    fakes.set_rpc(
        hybrid_search=[{"content": "context", "source_file": "notes.pdf", "score": 0.9}],
        list_session_files=[{"source_file": "notes.pdf", "chunk_count": 3}],
    )

    at = submit(text="Who is the main character?")

    assert not at.exception
    last = at.session_state.messages[-1]
    assert last["role"] == "assistant"
    assert last["content"] == "Hello world"
    assert last["sources"][0]["source_file"] == "notes.pdf"


def test_clear_chat_empties_history_but_keeps_session(fakes, run_app):
    at = run_app()
    session_id = at.session_state.session_id
    at.session_state["messages"] = [{"role": "user", "content": "hi"}]

    _button(at, "Clear chat").click().run()

    assert not at.exception
    assert at.session_state.messages == []
    assert at.session_state.session_id == session_id


def test_remove_all_documents_deletes_the_session(fakes, run_app):
    fakes.set_rpc(list_session_files=[{"source_file": "notes.pdf", "chunk_count": 3}])
    at = run_app()

    _button(at, "Remove all documents").click().run()

    assert not at.exception
    fakes.supabase.table.return_value.delete.return_value.eq.assert_called_with(
        "session_id", at.session_state.session_id
    )


def test_remove_single_document_deletes_only_that_file(fakes, run_app):
    fakes.set_rpc(list_session_files=[{"source_file": "notes.pdf", "chunk_count": 3}])
    at = run_app()

    at.button(key="rm_0").click().run()

    assert not at.exception
    fakes.supabase.table.return_value.delete.return_value.eq.return_value.eq.assert_called_with(
        "source_file", "notes.pdf"
    )