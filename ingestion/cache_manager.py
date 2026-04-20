"""Content-hash cache for parsed ingest output.

What we cache: the raw parser output (pre-chunking) for every file we ingest.
The expensive work — OCR, Whisper ASR, PDF layout — is captured here so repeat
ingestion of unchanged files skips straight to chunking/embedding (which are cheap).

Cache key is the SHA-256 of file content only. Chunking parameters (CHUNK_SIZE,
CHUNK_OVERLAP) are NOT part of the key because chunking runs fresh against cached
parser output on every ingest — changing chunk size does not require invalidating
the cache. Only bump CACHE_SCHEMA_VERSION when the cached chunk dict shape itself
changes (e.g. adding a required metadata field).

Storage layout:
    CACHE_DIR/ingest/<hash[:2]>/<hash>.pkl    sharded by first 2 hex chars to cap dir size
    CACHE_DIR/manifest.json                   {source_path: {hash, ingested_at, chunk_count, schema_version}}
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import CACHE_DIR, CACHE_SCHEMA_VERSION
from logging_config import get_logger

log = get_logger(__name__)

_INGEST_DIR = os.path.join(CACHE_DIR, "ingest")
_MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")
_HASH_CHUNK_SIZE = 1 << 20  # 1 MiB streaming buffer


def _ensure_dirs() -> None:
    os.makedirs(_INGEST_DIR, exist_ok=True)


def compute_file_hash(filepath: str) -> str:
    """Stream-hash a file's byte content with SHA-256."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            buf = f.read(_HASH_CHUNK_SIZE)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _pickle_path(file_hash: str) -> str:
    return os.path.join(_INGEST_DIR, file_hash[:2], f"{file_hash}.pkl")


def _load_manifest() -> Dict[str, Dict[str, Any]]:
    if not os.path.isfile(_MANIFEST_PATH):
        return {}
    try:
        with open(_MANIFEST_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Cache manifest unreadable (%s); treating as empty.", e)
        return {}


def _save_manifest(manifest: Dict[str, Dict[str, Any]]) -> None:
    # NOTE: not thread-safe. Full JSON rewrite races under parallel ingest.
    # Current UI and CLI ingest files serially, so safe. Revisit if the ingest
    # loop is ever parallelized.
    _ensure_dirs()
    tmp = _MANIFEST_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, _MANIFEST_PATH)
    except OSError as e:
        log.warning("Failed to persist cache manifest: %s", e)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def load_cached(file_hash: str) -> Optional[List[Dict[str, Any]]]:
    """Return cached parsed chunks for the given content hash, or None on miss/corruption."""
    path = _pickle_path(file_hash)
    if not os.path.isfile(path):
        return None

    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ValueError) as e:
        log.warning("Cache entry %s corrupt (%s); dropping.", file_hash[:12], e)
        try:
            os.unlink(path)
        except OSError:
            pass
        return None

    if not isinstance(payload, dict):
        log.warning("Cache entry %s has unexpected format; dropping.", file_hash[:12])
        try:
            os.unlink(path)
        except OSError:
            pass
        return None

    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        log.info(
            "Cache entry %s schema %s != current %s; invalidating.",
            file_hash[:12], payload.get("schema_version"), CACHE_SCHEMA_VERSION,
        )
        try:
            os.unlink(path)
        except OSError:
            pass
        return None

    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        return None
    return chunks


def save_cached(file_hash: str, chunks: List[Dict[str, Any]], source_path: str) -> None:
    """Persist parsed chunks under file_hash and update the manifest entry for source_path."""
    _ensure_dirs()
    shard = os.path.join(_INGEST_DIR, file_hash[:2])
    os.makedirs(shard, exist_ok=True)
    path = _pickle_path(file_hash)

    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "chunks": chunks,
    }
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except OSError as e:
        log.warning("Failed to write cache entry %s: %s", file_hash[:12], e)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return

    manifest = _load_manifest()
    manifest[os.path.abspath(source_path)] = {
        "hash": file_hash,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunk_count": len(chunks),
        "schema_version": CACHE_SCHEMA_VERSION,
    }
    _save_manifest(manifest)


def invalidate(source_path: str) -> None:
    """Drop manifest entry + pickle for a given source path, if present."""
    abs_src = os.path.abspath(source_path)
    manifest = _load_manifest()
    entry = manifest.pop(abs_src, None)
    _save_manifest(manifest)
    if not entry:
        return
    h = entry.get("hash")
    if h:
        path = _pickle_path(h)
        try:
            os.unlink(path)
        except OSError:
            pass
