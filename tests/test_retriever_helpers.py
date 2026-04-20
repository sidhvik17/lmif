from retrieval.retriever import (
    _meta_matches, _apply_filter, _split_filters, _is_broad_query, _validate_query,
    _PRE_FILTER_FIELDS,
)
import pytest


META = {"modality": "text", "word_count": 100, "author": "bob", "file_ext": ".pdf"}


def test_meta_scalar_equality():
    assert _meta_matches(META, {"modality": "text"}) is True
    assert _meta_matches(META, {"modality": "audio"}) is False


def test_meta_gte():
    assert _meta_matches(META, {"word_count": {"$gte": 50}}) is True
    assert _meta_matches(META, {"word_count": {"$gte": 500}}) is False


def test_meta_in_nin():
    assert _meta_matches(META, {"author": {"$in": ["alice", "bob"]}}) is True
    assert _meta_matches(META, {"author": {"$nin": ["alice", "bob"]}}) is False


def test_meta_callable():
    assert _meta_matches(META, {"word_count": lambda v: v and v > 10}) is True
    assert _meta_matches(META, {"word_count": lambda v: v and v > 500}) is False


def test_meta_unknown_operator_fails_closed():
    assert _meta_matches(META, {"word_count": {"$weird": 50}}) is False


def test_meta_missing_field_with_scalar():
    assert _meta_matches(META, {"missing": "x"}) is False


def test_split_filters_segregates_by_field():
    pre, post = _split_filters({"modality": "text", "author": "bob", "word_count": {"$gte": 10}})
    assert pre == {"modality": "text", "word_count": {"$gte": 10}}
    assert post == {"author": "bob"}


def test_split_filters_empty():
    assert _split_filters(None) == (None, {})
    assert _split_filters({}) == (None, {})


def test_pre_filter_fields_include_indexed_keys():
    assert "modality" in _PRE_FILTER_FIELDS
    assert "file_ext" in _PRE_FILTER_FIELDS


def test_apply_filter_returns_only_matching():
    results = [("a", {"modality": "text"}), ("b", {"modality": "audio"})]
    out = _apply_filter(results, {"modality": "text"})
    assert [t for t, _ in out] == ["a"]


def test_broad_query_detection():
    assert _is_broad_query("summarize everything") is True
    assert _is_broad_query("what do you have") is True
    assert _is_broad_query("what is the capital of France") is False


def test_validate_query_strips_and_rejects_empty():
    assert _validate_query("  hello  ") == "hello"
    with pytest.raises(ValueError):
        _validate_query("   ")


def test_validate_query_truncates_long():
    long = "x" * 5000
    out = _validate_query(long)
    assert len(out) == 2000
