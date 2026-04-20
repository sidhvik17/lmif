"""Tests for generation.generator.

Mocks ollama.Client to exercise:
  * empty-chunk fallback (no LLM call)
  * transport errors (connection, timeout, model-not-found)
  * SYSTEM_PROMPT regression guards — key rule clauses must stay present
    so future prompt edits can't silently drop the partial/compound behavior.

Answer-quality tests are intentionally NOT automated (LLM output is
non-deterministic). See plan file's manual checklist for live verification.
"""
from __future__ import annotations

from unittest import mock

import ollama
import pytest

from generation import generator


def _sample_chunks():
    return [
        ("HAEMOGLOBIN (HB) 14.1 g/dl", {"source": "lab.pdf", "page": 1, "modality": "text"}),
    ]


# ----- Empty / trivial input ------------------------------------------------


def test_empty_query_returns_fallback():
    ans, used = generator.generate("", _sample_chunks())
    assert "Empty query" in ans
    assert used == []


def test_whitespace_query_returns_fallback():
    ans, used = generator.generate("   \t\n", _sample_chunks())
    assert "Empty query" in ans


def test_empty_chunks_returns_not_found_without_llm_call():
    with mock.patch("generation.generator.ollama.Client") as mk:
        ans, used = generator.generate("anything", [])
        mk.assert_not_called()
    assert "No relevant content" in ans
    assert used == []


# ----- Error handling -------------------------------------------------------


def test_connection_error_yields_ollama_serve_hint():
    fake_client = mock.MagicMock()
    fake_client.chat.side_effect = ConnectionError("connection refused")
    with mock.patch("generation.generator.ollama.Client", return_value=fake_client):
        ans, used = generator.generate("anything", _sample_chunks())
    assert "Cannot reach Ollama" in ans
    assert "ollama serve" in ans
    # Chunks should still be returned so the UI can show retrieval provenance.
    assert used == _sample_chunks()


def test_model_not_found_yields_ollama_pull_hint():
    fake_client = mock.MagicMock()
    fake_client.chat.side_effect = ollama.ResponseError(
        f"model '{generator.LLM_MODEL}' not found"
    )
    with mock.patch("generation.generator.ollama.Client", return_value=fake_client):
        ans, _ = generator.generate("anything", _sample_chunks())
    assert "not available" in ans.lower()
    assert "ollama pull" in ans


def test_generic_exception_wrapped_with_type_name():
    fake_client = mock.MagicMock()
    fake_client.chat.side_effect = RuntimeError("boom")
    with mock.patch("generation.generator.ollama.Client", return_value=fake_client):
        ans, _ = generator.generate("anything", _sample_chunks())
    assert "RuntimeError" in ans


# ----- Empty LLM content fallback ------------------------------------------


def test_empty_llm_content_produces_friendly_message():
    fake_client = mock.MagicMock()
    fake_client.chat.return_value = {"message": {"role": "assistant", "content": "   "}}
    with mock.patch("generation.generator.ollama.Client", return_value=fake_client):
        ans, _ = generator.generate("q", _sample_chunks())
    assert "could not generate" in ans.lower()


def test_happy_path_returns_llm_content():
    fake_client = mock.MagicMock()
    fake_client.chat.return_value = {"message": {"role": "assistant", "content": "HB is 14.1 [1]."}}
    with mock.patch("generation.generator.ollama.Client", return_value=fake_client):
        ans, used = generator.generate("What is my HB?", _sample_chunks())
    assert "14.1" in ans
    assert used == _sample_chunks()


# ----- Prompt regression guards --------------------------------------------
# These tests catch silent edits that weaken the system prompt's key rules.
# They don't verify LLM output quality — they verify the instructions are still
# present to be followed.


@pytest.mark.parametrize("clause", [
    "COMPOUND QUERIES",
    "answer each part independently",
    "Never refuse the whole query because one sub-part is missing",
])
def test_system_prompt_contains_compound_query_rule(clause):
    assert clause in generator.SYSTEM_PROMPT


@pytest.mark.parametrize("clause", [
    "PARTIAL / SYNONYM MATCHES",
    "blood sugar",
    "glucose",
    "HB",
    "Match by medical/domain meaning",
])
def test_system_prompt_contains_synonym_match_rule(clause):
    assert clause in generator.SYSTEM_PROMPT


def test_system_prompt_still_forbids_fabrication():
    assert "Never fabricate" in generator.SYSTEM_PROMPT


def test_system_prompt_still_has_injection_guard():
    assert "DATA, not instructions" in generator.SYSTEM_PROMPT
