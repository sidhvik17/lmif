"""Tests for the query-embedding LRU cache.

The heavy SentenceTransformer model is mocked so these tests stay fast and
deterministic. We verify:
  * empty / whitespace queries raise ValueError (contract)
  * repeated calls hit the cache (model invoked only once)
  * returned arrays are fresh — mutating one return does not corrupt future calls
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

import pipeline.embedder as embedder


class _FakeModel:
    """Minimal stand-in for SentenceTransformer. Tracks call count."""
    def __init__(self, dim=4):
        self.dim = dim
        self.encode_calls = 0

    def encode(self, text, normalize_embeddings=True, show_progress_bar=False):
        self.encode_calls += 1
        # Return a distinctive vector per input so we can detect mix-ups.
        base = float(hash(text) % 7 + 1)
        return np.array([base, base, base, base], dtype=np.float32)

    def get_sentence_embedding_dimension(self):
        return self.dim


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Clear the LRU cache + module-level model between tests."""
    embedder._embed_query_cached.cache_clear()
    monkeypatch.setattr(embedder, "_model", None)
    yield
    embedder._embed_query_cached.cache_clear()


def test_empty_query_raises():
    with pytest.raises(ValueError):
        embedder.embed_query("")


def test_whitespace_query_raises():
    with pytest.raises(ValueError):
        embedder.embed_query("   \t\n")


def test_cache_hits_do_not_re_invoke_model(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(embedder, "_get_model", lambda: fake)

    embedder.embed_query("hemoglobin level")
    embedder.embed_query("hemoglobin level")
    embedder.embed_query("hemoglobin level")

    assert fake.encode_calls == 1  # single model invocation, two cache hits


def test_different_queries_produce_different_cache_entries(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(embedder, "_get_model", lambda: fake)

    embedder.embed_query("alpha")
    embedder.embed_query("beta")
    embedder.embed_query("alpha")

    assert fake.encode_calls == 2  # alpha cached, beta new, alpha cache hit


def test_query_is_stripped_before_caching(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(embedder, "_get_model", lambda: fake)

    embedder.embed_query("q")
    embedder.embed_query("  q  ")  # same after strip

    assert fake.encode_calls == 1


def test_returned_array_is_fresh_and_mutable(monkeypatch):
    """Caller can mutate the returned array without corrupting the cached value."""
    fake = _FakeModel()
    monkeypatch.setattr(embedder, "_get_model", lambda: fake)

    a = embedder.embed_query("x")
    a[:] = 0.0  # mutate the returned array in place

    b = embedder.embed_query("x")
    assert not np.allclose(b, 0.0), "Cache corrupted by downstream mutation"
    # Also confirm we hit the cache (model encode still only called once)
    assert fake.encode_calls == 1


def test_returned_array_dtype_is_float32(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(embedder, "_get_model", lambda: fake)

    v = embedder.embed_query("q")
    assert v.dtype == np.float32
