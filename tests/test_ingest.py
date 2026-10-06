import ingest


def test_chunk_text_overlaps_chunks():
    words = " ".join(str(i) for i in range(10))

    chunks = ingest.chunk_text(words, chunk_size=4, overlap=2)

    assert chunks[0] == "0 1 2 3"
    assert chunks[1] == "2 3 4 5"
    assert chunks[-1].endswith("9")


def test_chunk_text_empty_input():
    assert ingest.chunk_text("") == []


def test_extract_and_chunk_tags_source_file(fakes):
    fakes.converter.convert.return_value.document.export_to_markdown.return_value = (
        "# Title\n\nSome document text."
    )

    chunks = ingest.extract_and_chunk("/tmp/upload.pdf", "notes.pdf")

    assert chunks
    assert all(c["source_file"] == "notes.pdf" for c in chunks)
    assert "Some document text." in chunks[0]["text"]


def test_extract_and_chunk_empty_document(fakes):
    fakes.converter.convert.return_value.document.export_to_markdown.return_value = "   "

    assert ingest.extract_and_chunk("/tmp/empty.pdf", "empty.pdf") == []


def test_embed_and_store_inserts_with_session(fakes):
    fakes.supabase.table.return_value.insert.return_value.execute.return_value.data = [{"id": 7}]
    chunks = [{"text": "hello", "source_file": "a.txt", "page_number": 1}]

    ids = ingest.embed_and_store(chunks, session_id="abc")

    assert ids == [7]
    row = fakes.supabase.table.return_value.insert.call_args.args[0]
    assert row["session_id"] == "abc"
    assert row["embedding"] == [0.1, 0.2, 0.3]


def test_delete_session_documents_without_session_does_nothing(fakes):
    assert ingest.delete_session_documents("") == 0
    fakes.supabase.table.assert_not_called()
