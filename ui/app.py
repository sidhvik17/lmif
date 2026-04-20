import streamlit as st
import sys
import os
import re

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def _safe_filename(name: str) -> str:
    """Strip path separators and control chars from uploaded filenames.

    Prevents path traversal via crafted upload names (e.g. ../../etc/passwd).
    Keeps extension and basic punctuation. Falls back to "upload.bin" if empty.
    """
    base = os.path.basename(name or "")
    base = re.sub(r"[^\w.\-]+", "_", base)
    base = base.strip("._") or "upload.bin"
    return base[:200]

from logging_config import setup_logging
setup_logging()

from ingestion.ingest_manager import ingest_file
from pipeline.chunker import chunk_documents, deduplicate_chunks
from pipeline.embedder import embed_chunks
from vectorstore.store import (
    add_chunks, get_count, clear_collection, modality_breakdown,
)
from retrieval.retriever import retrieve
from generation.generator import generate
from generation.citation_formatter import format_citations

st.set_page_config(page_title="LMIF", page_icon="", layout="wide")
st.title("LMIF -- Local Multimodal Intelligence Framework")
st.caption("Offline | Privacy-Preserving | Text + Image + Audio RAG")

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.header("Collection")
chunk_count = get_count()
st.sidebar.metric("Indexed Chunks", chunk_count)

# Per-modality breakdown
if chunk_count > 0:
    breakdown = modality_breakdown()
    cols = st.sidebar.columns(max(1, len(breakdown)))
    for col, (mod, n) in zip(cols, sorted(breakdown.items(), key=lambda x: -x[1])):
        col.metric(mod.capitalize(), n)

# Clear & Re-ingest button
if st.sidebar.button("Clear All Indexed Data", type="secondary"):
    clear_collection()
    st.sidebar.success("Collection cleared. Re-ingest your files.")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("Retrieval Settings")
modality_filter = st.sidebar.selectbox(
    "Filter by modality",
    [None, "text", "image", "audio"],
    format_func=lambda x: "All modalities" if x is None else x.capitalize(),
)
use_hyde = st.sidebar.toggle("HyDE query expansion", value=False,
                              help="Generate a hypothetical answer to improve retrieval recall")

# ── Main Tabs ────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Ingest", "Query"])

with tab1:
    st.subheader("Ingest Documents")
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, JPG, PNG, MP3, or WAV",
        accept_multiple_files=True,
    )
    if st.button("Ingest Files") and uploaded:
        for f in uploaded:
            safe_name = _safe_filename(f.name)
            if safe_name != f.name:
                st.info(f"Renamed `{f.name}` → `{safe_name}` for safety.")
            tmp = os.path.join(
                os.path.dirname(__file__), "..", "data", safe_name
            )
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            with open(tmp, "wb") as fp:
                fp.write(f.getvalue())
            with st.spinner(f"Processing {f.name}..."):
                try:
                    result = ingest_file(tmp)
                except Exception as e:
                    st.error(f"{f.name}: ingestion error -- {e}")
                    continue
                if not result.chunks:
                    st.warning(f"{f.name}: no content extracted (unsupported or empty).")
                    continue
                if result.from_cache:
                    st.info(f"{f.name}: loaded from cache (skipped re-parse)")
                chunks = chunk_documents(result.chunks)
                if not chunks:
                    st.warning(f"{f.name}: nothing to index after chunking.")
                    continue
                vectors = embed_chunks(chunks)
                # Deduplicate near-identical chunks
                before_count = len(chunks)
                chunks, vectors = deduplicate_chunks(chunks, vectors)
                if before_count > len(chunks):
                    st.info(f"Deduplication: {before_count} -> {len(chunks)} chunks")
                add_chunks(chunks, vectors)
            st.success(f"{f.name} -- {len(chunks)} chunks indexed")
        st.rerun()

with tab2:
    st.subheader("Ask a Question")
    q = st.text_input("Enter your query")
    if st.button("Ask") and q:
        if get_count() == 0:
            st.warning("No documents have been indexed yet. Go to the Ingest tab first.")
        else:
            with st.spinner("Retrieving & generating..."):
                chunks = retrieve(q, modality_filter=modality_filter, use_hyde=use_hyde)
                answer, used = generate(q, chunks)

            # Show answer
            formatted = format_citations(answer, used)
            st.markdown(f"### Answer\n{formatted}")

            # Show retrieved sources clearly below the answer
            if used:
                st.markdown("---")
                st.markdown("#### Retrieved Sources")
                for i, (text, meta) in enumerate(used, 1):
                    mod_label = meta.get("modality", "?").upper()
                    src_name = os.path.basename(meta.get("source", "unknown"))
                    page = meta.get("page", "?")
                    with st.expander(f"[{i}] {src_name} ({mod_label}, Page/Time: {page})"):
                        st.text(text[:500])

            # Sidebar diagnostics
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"**Retrieved chunks:** {len(chunks)}")
            for i, (text, meta) in enumerate(chunks[:5], 1):
                mod_label = meta.get("modality", "?").upper()
                st.sidebar.caption(f"Chunk {i} [{mod_label}]: {text[:120]}...")
