import main
from conftest import make_stream


def test_is_summary_request_detects_broad_questions():
    assert main.is_summary_request("Can you summarize this?")
    assert main.is_summary_request("What is this document about")
    assert main.is_summary_request("TL;DR please")


def test_is_summary_request_ignores_specific_questions():
    assert not main.is_summary_request("Who is Luke Skywalker's father?")


def test_build_context_str_includes_source_and_page():
    context = main._build_context_str([
        {"source_file": "notes.pdf", "page_number": 3, "content": "Chunk text"},
    ])
    assert "[Source: notes.pdf, p.3]" in context
    assert "Chunk text" in context


def test_build_context_str_handles_missing_fields():
    context = main._build_context_str([{}])
    assert "[Source: Unknown, p.N/A]" in context


def test_retrieve_context_filters_by_session(fakes):
    fakes.supabase.rpc.return_value.execute.return_value.data = [{"content": "hit"}]

    result = main.retrieve_context("question", session_id="abc")

    assert result == [{"content": "hit"}]
    name, params = fakes.supabase.rpc.call_args.args
    assert name == "match_documents"
    assert params["filter_session_id"] == "abc"
    assert params["query_embedding"] == [0.1, 0.2, 0.3]


def test_generate_answer_streams_tokens(fakes):
    fakes.groq.chat.completions.create.return_value = make_stream("The ", None, "answer")

    answer = "".join(main.generate_answer("question", [{"content": "context"}]))

    assert answer == "The answer"
    messages = fakes.groq.chat.completions.create.call_args.kwargs["messages"]
    assert "context" in messages[0]["content"]


def test_generate_summary_without_upload_asks_for_document(fakes):
    fakes.supabase.table.return_value.select.return_value.eq.return_value \
        .order.return_value.limit.return_value.execute.return_value.data = []

    summary = "".join(main.generate_summary("session-1"))

    assert "upload one first" in summary
    fakes.groq.chat.completions.create.assert_not_called()
