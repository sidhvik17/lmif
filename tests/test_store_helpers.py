import numpy as np
from vectorstore.store import (
    _deterministic_id, _sanitize_metadata, _normalize_embeddings, _build_where,
)


def test_deterministic_id_stable():
    a = _deterministic_id("hello world", "/x.pdf", "1")
    b = _deterministic_id("hello world", "/x.pdf", "1")
    assert a == b
    assert len(a) == 16


def test_deterministic_id_differs_on_content():
    a = _deterministic_id("foo", "/x.pdf", "1")
    b = _deterministic_id("bar", "/x.pdf", "1")
    assert a != b


def test_deterministic_id_differs_on_source():
    a = _deterministic_id("same", "/x.pdf", "1")
    b = _deterministic_id("same", "/y.pdf", "1")
    assert a != b


def test_sanitize_metadata_drops_none():
    out = _sanitize_metadata({"a": 1, "b": None, "c": "x"})
    assert out == {"a": 1, "c": "x"}


def test_sanitize_metadata_stringifies_unsupported():
    out = _sanitize_metadata({"list": [1, 2], "dict": {"k": 1}})
    assert isinstance(out["list"], str)
    assert isinstance(out["dict"], str)


def test_normalize_embeddings_unit_norm():
    v = [[3.0, 4.0, 0.0]]
    out = _normalize_embeddings(v)
    arr = np.asarray(out[0])
    assert abs(np.linalg.norm(arr) - 1.0) < 1e-5


def test_normalize_embeddings_handles_zero_vector():
    v = [[0.0, 0.0, 0.0]]
    out = _normalize_embeddings(v)
    assert out[0] == [0.0, 0.0, 0.0]


def test_normalize_embeddings_accepts_1d():
    out = _normalize_embeddings([1.0, 0.0, 0.0])
    assert out == [[1.0, 0.0, 0.0]]


def test_build_where_none_for_empty():
    assert _build_where(None) is None
    assert _build_where({}) is None


def test_build_where_single_passes_through():
    assert _build_where({"modality": "text"}) == {"modality": "text"}


def test_build_where_multi_wraps_with_and():
    w = _build_where({"modality": "text", "author": "bob"})
    assert "$and" in w
    clauses = w["$and"]
    assert {"modality": "text"} in clauses
    assert {"author": "bob"} in clauses


def test_build_where_drops_none_values():
    w = _build_where({"modality": "text", "author": None})
    assert w == {"modality": "text"}
