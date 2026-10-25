import ingest


def test_extract_and_chunk_tags_source_file(fakes):
    fakes.converter.convert.return_value.text_content = "# Title\n\nSome document text."

    chunks = ingest.extract_and_chunk("/tmp/upload.pdf", "notes.pdf")

    assert chunks
    assert all(c["source_file"] == "notes.pdf" for c in chunks)
    assert "Some document text." in chunks[0]["text"]
    assert chunks[0]["text"].startswith("[Title]")  # header context is kept


def test_extract_and_chunk_empty_document(fakes):
    fakes.converter.convert.return_value.text_content = "   "

    assert ingest.extract_and_chunk("/tmp/empty.pdf", "empty.pdf") == []


def test_extract_and_chunk_drops_tiny_chunks(fakes):
    fakes.converter.convert.return_value.text_content = "too short"

    assert ingest.extract_and_chunk("/tmp/a.txt", "a.txt") == []


def test_csv_chunks_repeat_the_header_row(fakes):
    rows = "\n".join(f"| name{i} | {i} |" for i in range(200))
    fakes.converter.convert.return_value.text_content = (
        f"| name | value |\n| --- | --- |\n{rows}"
    )

    chunks = ingest.extract_and_chunk("/tmp/data.csv", "data.csv")

    assert len(chunks) > 1
    assert all(c["text"].startswith("| name | value |\n| --- | --- |") for c in chunks)
    assert all(len(c["text"]) <= ingest.CHUNK_SIZE + 50 for c in chunks)


def test_embed_and_store_inserts_with_session(fakes):
    fakes.supabase.table.return_value.insert.return_value.execute.return_value.data = [{"id": 7}]
    chunks = [{"text": "hello", "source_file": "a.txt"}]

    ids = ingest.embed_and_store(chunks, session_id="abc")

    assert ids == [7]
    row = fakes.supabase.table.return_value.insert.call_args.args[0][0]
    assert row["session_id"] == "abc"
    assert row["source_file"] == "a.txt"
    assert row["embedding"] == [0.1, 0.2, 0.3]


def test_embed_and_store_replaces_previous_upload_of_same_file(fakes):
    fakes.supabase.table.return_value.insert.return_value.execute.return_value.data = [{"id": 1}]

    ingest.embed_and_store([{"text": "hello", "source_file": "a.txt"}], session_id="abc")

    delete_eq = fakes.supabase.table.return_value.delete.return_value.eq
    delete_eq.assert_called_with("session_id", "abc")
    delete_eq.return_value.eq.assert_called_with("source_file", "a.txt")


def test_delete_session_documents_without_session_does_nothing(fakes):
    assert ingest.delete_session_documents("") == 0
    fakes.supabase.table.assert_not_called()


def test_delete_session_documents_returns_deleted_count(fakes):
    fakes.supabase.table.return_value.delete.return_value.eq.return_value.execute.return_value.data = [
        {"id": 1},
        {"id": 2},
    ]

    assert ingest.delete_session_documents("abc") == 2