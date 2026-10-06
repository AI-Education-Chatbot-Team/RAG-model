"""
Shared test setup.

main.py and ingest.py connect to Supabase, Groq, and download ML models as soon
as they are imported. These patches start BEFORE those modules are imported so
the tests run offline, with no API keys, and without downloading any models.
"""
import os
import pkgutil
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")

# The real classes are kept here for the opt-in integration tests
REAL = {}
for target in (
    "supabase.create_client",
    "groq.Groq",
    "sentence_transformers.SentenceTransformer",
    "docling.document_converter.DocumentConverter",
):
    module, attr = target.rsplit(".", 1)
    REAL[target] = pkgutil.resolve_name(f"{module}:{attr}")
    patch(target, MagicMock()).start()

import ingest  # noqa: E402
import main  # noqa: E402


class FakeEmbedder:
    def encode(self, text):
        return np.array([0.1, 0.2, 0.3])


def make_stream(*tokens):
    """Build a fake Groq streaming response that yields the given tokens."""
    return [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))])
        for t in tokens
    ]


@pytest.fixture
def fake_supabase(monkeypatch):
    db = MagicMock()
    db.rpc.return_value.execute.return_value.data = []
    monkeypatch.setattr(main, "supabase", db)
    monkeypatch.setattr(ingest, "supabase", db)
    return db


@pytest.fixture
def fake_groq(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.return_value = make_stream("Hello", " world")
    monkeypatch.setattr(main, "client", client)
    return client


@pytest.fixture
def fake_embedder(monkeypatch):
    monkeypatch.setattr(main, "embed_model", FakeEmbedder())
    monkeypatch.setattr(ingest, "model", FakeEmbedder())


@pytest.fixture
def fake_converter(monkeypatch):
    converter = MagicMock()
    monkeypatch.setattr(ingest, "doc_converter", converter)
    return converter


@pytest.fixture
def fakes(fake_supabase, fake_groq, fake_embedder, fake_converter):
    return SimpleNamespace(
        supabase=fake_supabase, groq=fake_groq, converter=fake_converter
    )
