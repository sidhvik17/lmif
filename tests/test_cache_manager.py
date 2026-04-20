"""Tests for ingestion.cache_manager.

Uses monkeypatching to point CACHE_DIR at a per-test tmp_path — keeps tests
isolated from the real on-disk cache under the project root.
"""
from __future__ import annotations

import importlib
import os
import pickle

import pytest


@pytest.fixture
def cache_mod(tmp_path, monkeypatch):
    """Fresh cache_manager pointed at tmp_path, reloaded to pick up patched CACHE_DIR."""
    import config
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CACHE_SCHEMA_VERSION", 1)

    import ingestion.cache_manager as cm
    importlib.reload(cm)
    return cm


def _write_sample(tmp_path, name="sample.txt", content=b"hello LMIF"):
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


def test_hash_stable_across_calls(cache_mod, tmp_path):
    path = _write_sample(tmp_path)
    h1 = cache_mod.compute_file_hash(path)
    h2 = cache_mod.compute_file_hash(path)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex length


def test_hash_changes_when_content_changes(cache_mod, tmp_path):
    path = _write_sample(tmp_path, content=b"first")
    h1 = cache_mod.compute_file_hash(path)
    # Mutate a single byte
    with open(path, "wb") as f:
        f.write(b"second")
    h2 = cache_mod.compute_file_hash(path)
    assert h1 != h2


def test_load_miss_returns_none(cache_mod):
    assert cache_mod.load_cached("0" * 64) is None


def test_save_then_load_roundtrip(cache_mod, tmp_path):
    path = _write_sample(tmp_path)
    h = cache_mod.compute_file_hash(path)
    chunks = [
        {"text": "alpha", "metadata": {"source": path, "page": 1, "modality": "text"}},
        {"text": "beta", "metadata": {"source": path, "page": 2, "modality": "text"}},
    ]
    cache_mod.save_cached(h, chunks, path)

    loaded = cache_mod.load_cached(h)
    assert loaded == chunks


def test_corrupt_pickle_returns_none_and_removes_file(cache_mod, tmp_path):
    path = _write_sample(tmp_path)
    h = cache_mod.compute_file_hash(path)
    cache_mod.save_cached(h, [{"text": "ok", "metadata": {}}], path)

    pkl_path = cache_mod._pickle_path(h)
    # Overwrite with garbage
    with open(pkl_path, "wb") as f:
        f.write(b"not a valid pickle stream")

    assert cache_mod.load_cached(h) is None
    assert not os.path.isfile(pkl_path)  # corrupt entry is removed


def test_schema_version_mismatch_invalidates(cache_mod, tmp_path, monkeypatch):
    path = _write_sample(tmp_path)
    h = cache_mod.compute_file_hash(path)
    cache_mod.save_cached(h, [{"text": "ok", "metadata": {}}], path)

    # Bump schema version to simulate a breaking shape change.
    import config
    monkeypatch.setattr(config, "CACHE_SCHEMA_VERSION", 999)
    # Reload cache_mod so it sees the new version.
    importlib.reload(cache_mod)

    assert cache_mod.load_cached(h) is None


def test_manifest_tracks_source_path(cache_mod, tmp_path):
    path = _write_sample(tmp_path)
    h = cache_mod.compute_file_hash(path)
    cache_mod.save_cached(h, [{"text": "x", "metadata": {}}], path)

    manifest = cache_mod._load_manifest()
    entry = manifest.get(os.path.abspath(path))
    assert entry is not None
    assert entry["hash"] == h
    assert entry["chunk_count"] == 1


def test_invalidate_removes_entry_and_pickle(cache_mod, tmp_path):
    path = _write_sample(tmp_path)
    h = cache_mod.compute_file_hash(path)
    cache_mod.save_cached(h, [{"text": "x", "metadata": {}}], path)
    pkl_path = cache_mod._pickle_path(h)
    assert os.path.isfile(pkl_path)

    cache_mod.invalidate(path)
    assert not os.path.isfile(pkl_path)
    manifest = cache_mod._load_manifest()
    assert os.path.abspath(path) not in manifest


def test_payload_not_a_dict_is_rejected(cache_mod, tmp_path):
    """If someone wrote a raw list instead of the {schema_version, chunks} dict, reject gracefully."""
    path = _write_sample(tmp_path)
    h = cache_mod.compute_file_hash(path)
    cache_mod._ensure_dirs()
    shard = os.path.join(cache_mod._INGEST_DIR, h[:2])
    os.makedirs(shard, exist_ok=True)
    pkl_path = cache_mod._pickle_path(h)
    with open(pkl_path, "wb") as f:
        pickle.dump(["just", "a", "list"], f)

    assert cache_mod.load_cached(h) is None
    assert not os.path.isfile(pkl_path)  # dropped
