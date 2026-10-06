from streamlit.testing.v1 import AppTest

APP_PATH = "../app.py"


def run_app():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    return at


def test_app_loads_without_errors(fakes):
    at = run_app()

    assert not at.exception
    assert at.title[0].value == "🤖 RAG Knowledge Assistant"
    assert "session_id" in at.session_state


def test_question_shows_streamed_answer(fakes):
    fakes.supabase.rpc.return_value.execute.return_value.data = [
        {"content": "context", "source_file": "notes.pdf", "page_number": 1, "similarity": 0.9},
    ]
    at = run_app()

    at.chat_input[0].set_value("Who is the main character?").run()

    assert not at.exception
    assert at.session_state.messages[-1]["role"] == "assistant"
    assert at.session_state.messages[-1]["content"] == "Hello world"


def test_clear_chat_deletes_session_documents(fakes):
    at = run_app()
    old_session = at.session_state.session_id

    at.sidebar.button[0].click().run()

    assert not at.exception
    fakes.supabase.table.return_value.delete.return_value.eq.assert_called_with(
        "session_id", old_session
    )
    assert at.session_state.session_id != old_session
