import main
from conftest import make_stream


def test_is_summary_request_detects_broad_questions():
    assert main.is_summary_request("Can you summarize this?")
    assert main.is_summary_request("What is this document about")
    assert main.is_summary_request("TL;DR please")


def test_is_summary_request_ignores_specific_questions():
    assert not main.is_summary_request("Who is Luke Skywalker's father?")


def test_is_summary_request_ignores_scoped_requests():
    assert not main.is_summary_request("Summarize chapter 3")
    assert not main.is_summary_request("Give me an overview of the table on page 2")


def test_build_context_str_includes_source():
    context = main._build_context_str(
        [{"source_file": "notes.pdf", "content": "Chunk text"}]
    )
    assert "[Source: notes.pdf]" in context
    assert "Chunk text" in context


def test_build_context_str_handles_missing_fields():
    assert "[Source: Unknown]" in main._build_context_str([{}])


def test_retrieve_context_filters_by_session(fakes):
    fakes.set_rpc(hybrid_search=[{"content": "hit"}])

    result = main.retrieve_context("question", session_id="abc")

    assert result == [{"content": "hit"}]
    name, params = fakes.supabase.rpc.call_args.args
    assert name == "hybrid_search"
    assert params["filter_session_id"] == "abc"
    assert params["query_text"] == "question"
    assert params["query_embedding"] == [0.1, 0.2, 0.3]
    assert params["match_count"] == main.TOP_K


def test_retrieve_context_without_session_returns_nothing(fakes):
    assert main.retrieve_context("question", session_id=None) == []
    fakes.supabase.rpc.assert_not_called()


def test_generate_answer_streams_tokens(fakes):
    fakes.groq.chat.completions.create.return_value = make_stream("The ", None, "answer")

    answer = "".join(main.generate_answer("question", [{"content": "context"}]))

    assert answer == "The answer"
    messages = fakes.groq.chat.completions.create.call_args.kwargs["messages"]
    assert "context" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "question"}


def test_generate_answer_includes_recent_history(fakes):
    history = [
        {"role": "user", "content": "Who is Luke?"},
        {"role": "assistant", "content": "A Jedi."},
    ]

    "".join(main.generate_answer("And his father?", [{"content": "c"}], history=history))

    messages = fakes.groq.chat.completions.create.call_args.kwargs["messages"]
    assert [m["content"] for m in messages[1:-1]] == ["Who is Luke?", "A Jedi."]


def test_condense_question_without_history_skips_the_llm(fakes):
    assert main.condense_question("What is ATP?", []) == "What is ATP?"
    fakes.groq.chat.completions.create.assert_not_called()


def test_generate_summary_without_files_asks_for_document(fakes):
    summary = "".join(main.generate_summary("session-1", []))

    assert "Upload one first" in summary
    fakes.groq.chat.completions.create.assert_not_called()