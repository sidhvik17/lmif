import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ingestion.ingest_manager import ingest_file
from pipeline.chunker import chunk_documents, deduplicate_chunks
from pipeline.embedder import embed_chunks
from vectorstore.store import add_chunks
from retrieval.retriever import retrieve
from generation.generator import generate
from generation.citation_formatter import format_citations

st.set_page_config(page_title="LMIF", page_icon="🧠", layout="wide")
st.title("🧠 LMIF — Local Multimodal Intelligence Framework")
st.caption("Offline · Privacy-Preserving · Text + Image + Audio RAG")

tab1, tab2 = st.tabs(["📥 Ingest", "🔍 Query"])

with tab1:
    st.subheader("Ingest Documents")
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, JPG, PNG, MP3, or WAV",
        accept_multiple_files=True,
    )
    if st.button("Ingest Files") and uploaded:
        for f in uploaded:
            tmp = os.path.join(
                os.path.dirname(__file__), "..", "data", f.name
            )
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            with open(tmp, "wb") as fp:
                fp.write(f.getvalue())
            raw = ingest_file(tmp)
            if not raw:
                st.warning(f"{f.name}: no content extracted (unsupported or empty).")
                continue
            chunks = chunk_documents(raw)
            if not chunks:
                st.warning(f"{f.name}: nothing to index after chunking.")
                continue
            vectors = embed_chunks(chunks)
            # Deduplicate near-identical chunks
            before_count = len(chunks)
            chunks, vectors = deduplicate_chunks(chunks, vectors)
            if before_count > len(chunks):
                st.info(f"Deduplication: {before_count} → {len(chunks)} chunks")
            add_chunks(chunks, vectors)
            st.success(f"✓ {f.name} — {len(chunks)} chunks indexed")

with tab2:
    st.subheader("Ask a Question")
    q = st.text_input("Enter your query")
    if st.button("Ask") and q:
        with st.spinner("Retrieving & generating..."):
            chunks = retrieve(q)
            answer, used = generate(q, chunks)
        # Format answer with inline citations + footer
        formatted = format_citations(answer, used)
        st.markdown(f"### Answer\n{formatted}")
