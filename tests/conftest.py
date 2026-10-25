"""
Shared test setup.

shared.py connects to Supabase, Groq, and loads the embedding model as soon as it
is imported. These patches start BEFORE the src modules are imported so the tests
run offline, with no API keys, and without downloading any models.

The src files use flat imports (`from shared import ...`) because Streamlit puts
src/ on sys.path. The tests do the same, so the tests and app.py share ONE copy of
each module and the fakes below actually reach the code app.py runs.
"""
import os
import pkgutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
APP_PATH = str(SRC / "app.py")

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")

# The real classes are kept here for the opt-in integration tests
REAL = {}
for target in (
    "supabase.create_client",
    "groq.Groq",
    "sentence_transformers.SentenceTransformer",
    "markitdown.MarkItDown",
):
    module, attr = target.rsplit(".", 1)
    REAL[target] = pkgutil.resolve_name(f"{module}:{attr}")
    patch(target, MagicMock()).start()

import ingest  # noqa: E402
import main  # noqa: E402
import shared  # noqa: E402


class FakeEmbedder:
    """Stands in for SentenceTransformer: one 3-d vector per input text."""

    def encode(self, texts, **kwargs):
        return np.tile([0.1, 0.2, 0.3], (len(texts), 1))


class FakeUpload:
    """Minimal stand-in for Streamlit's UploadedFile."""

    def __init__(self, name, data=b"file contents"):
        self.name = name
        self._data = data

    def getbuffer(self):
        return memoryview(self._data)


def make_stream(*tokens):
    """Build a fake Groq streaming response that yields the given tokens."""
    return [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))])
        for t in tokens
    ]


def _rpc_router(results):
    """supabase.rpc(name, params) -> an object whose .execute().data is results[name]."""

    def _rpc(name, params=None):
        call = MagicMock()
        call.execute.return_value.data = results.get(name, [])
        return call

    return _rpc


@pytest.fixture
def fake_supabase(monkeypatch):
    db = MagicMock()
    db.rpc.side_effect = _rpc_router({})
    # main, ingest and shared each hold their own reference to the client
    for module in (main, ingest, shared):
        monkeypatch.setattr(module, "supabase", db)
    return db


@pytest.fixture
def fake_groq(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.return_value = make_stream("Hello", " world")
    monkeypatch.setattr(main, "groq_client", client)
    return client


@pytest.fixture
def fake_embedder(monkeypatch):
    # embed_texts() lives in shared.py and reads shared.embed_model
    monkeypatch.setattr(shared, "embed_model", FakeEmbedder())


@pytest.fixture
def fake_converter(monkeypatch):
    converter = MagicMock()
    monkeypatch.setattr(ingest, "md", converter)
    return converter


@pytest.fixture
def fakes(fake_supabase, fake_groq, fake_embedder, fake_converter, monkeypatch):
    monkeypatch.setattr(main, "USE_RERANKER", False)

    def set_rpc(**results):
        """Set what each Supabase RPC returns, e.g. set_rpc(hybrid_search=[...])."""
        fake_supabase.rpc.side_effect = _rpc_router(results)

    return SimpleNamespace(
        supabase=fake_supabase,
        groq=fake_groq,
        converter=fake_converter,
        set_rpc=set_rpc,
    )


@pytest.fixture
def run_app():
    """Run the real app with no user input."""

    def _run():
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        return at

    return _run


@pytest.fixture
def submit(monkeypatch):
    """
    Run the app as if the user submitted text and/or files in the chat box.
    AppTest can't attach files to st.chat_input, so it is swapped for a fake.
    The value is returned once (like a real submit), so st.rerun() after an
    upload doesn't loop forever.
    """

    def _submit(text=None, files=None):
        pending = [SimpleNamespace(text=text, files=files or [])]
        monkeypatch.setattr(
            streamlit, "chat_input", lambda *a, **k: pending.pop() if pending else None
        )
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        return at

    return _submit