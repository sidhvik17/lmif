"""Benchmark LMIF: ingestion timings, query latency, and embedding PCA.

Outputs PNG plots + JSON metrics into ``lmif final documentation``.
Run from repo root:  python run_benchmark.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO.parent / "lmif final documentation"
DATA_DIR = REPO / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_LATENCY = OUT_DIR / "plot_latency_vs_size.png"
PLOT_INGEST  = OUT_DIR / "plot_ingest_vs_length.png"
PLOT_PCA     = OUT_DIR / "plot_pca_embeddings.png"
METRICS_JSON = OUT_DIR / "benchmark_metrics.json"

from ingestion.ingest_manager import _parse_file, detect_type  # noqa: E402
from pipeline.chunker import chunk_documents                   # noqa: E402
from pipeline.embedder import embed_chunks                     # noqa: E402
from retrieval.retriever import retrieve                       # noqa: E402
from vectorstore.store import _get_collection                  # noqa: E402


def bench_ingestion():
    """Per-file parse + chunk + embed timings."""
    rows = []
    for fpath in sorted(DATA_DIR.iterdir()):
        if not fpath.is_file():
            continue
        ftype = detect_type(str(fpath))
        if ftype is None:
            continue
        size_kb = fpath.stat().st_size / 1024.0
        t0 = time.perf_counter()
        try:
            raw = _parse_file(str(fpath), ftype)
        except Exception as e:
            print(f"[SKIP] parse failed {fpath.name}: {e}")
            continue
        t_parse = time.perf_counter() - t0
        total_chars = sum(len((r.get("text") or "")) for r in raw)
        if not raw:
            continue
        # enrich metadata so chunker has modality
        for r in raw:
            r.setdefault("metadata", {}).setdefault(
                "modality",
                {"text": "text", "image": "image", "audio": "audio"}[ftype],
            )
        t0 = time.perf_counter()
        chunks = chunk_documents(raw)
        t_chunk = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = embed_chunks(chunks) if chunks else []
        t_embed = time.perf_counter() - t0
        total_time = t_parse + t_chunk + t_embed
        rows.append({
            "file": fpath.name,
            "modality": ftype,
            "size_kb": round(size_kb, 1),
            "text_chars": total_chars,
            "chunks": len(chunks),
            "parse_s": round(t_parse, 3),
            "chunk_s": round(t_chunk, 3),
            "embed_s": round(t_embed, 3),
            "total_s": round(total_time, 3),
        })
        print(f"[INGEST] {fpath.name} size={size_kb:.1f}KB chars={total_chars} "
              f"parse={t_parse:.2f}s chunk={t_chunk:.2f}s embed={t_embed:.2f}s")
    return rows


def bench_query_latency():
    """Query latency vs current corpus size (file count)."""
    queries = [
        "What does the lab report say?",
        "Describe the image content",
        "Summarize the audio recording",
        "What tests are listed in the report?",
        "What is shown in the photograph?",
    ]
    lat = []
    for q in queries:
        t0 = time.perf_counter()
        try:
            res = retrieve(q)
        except Exception as e:
            print(f"[QUERY] {q!r} failed: {e}")
            continue
        dt = time.perf_counter() - t0
        lat.append({"query": q, "latency_s": round(dt, 3), "hits": len(res)})
        print(f"[QUERY] {q!r} -> {len(res)} hits in {dt:.2f}s")
    return lat


def plot_latency_vs_size(ingest_rows):
    """Query-stage latency proxy: per-file end-to-end ingestion time vs size.
    Real retrieval latency is small; the expensive path in RAG is ingestion,
    so file-size-to-time is the informative trace for operators."""
    sizes = [r["size_kb"] for r in ingest_rows]
    times = [r["total_s"] for r in ingest_rows]
    colors = {"text": "#1f77b4", "image": "#ff7f0e", "audio": "#2ca02c"}
    mods = [r["modality"] for r in ingest_rows]
    plt.figure(figsize=(7, 4.5))
    for m in sorted(set(mods)):
        xs = [s for s, mm in zip(sizes, mods) if mm == m]
        ys = [t for t, mm in zip(times, mods) if mm == m]
        plt.scatter(xs, ys, c=colors.get(m, "gray"), label=m, s=80,
                    edgecolors="black", linewidth=0.5)
    if len(sizes) >= 2:
        z = np.polyfit(sizes, times, 1)
        xx = np.linspace(min(sizes), max(sizes), 50)
        plt.plot(xx, np.polyval(z, xx), "--", color="gray", alpha=0.7,
                 label=f"linear fit (slope={z[0]:.4f} s/KB)")
    plt.xlabel("File size (KB)")
    plt.ylabel("End-to-end processing latency (s)")
    plt.title("Processing Latency vs File Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_LATENCY, dpi=150)
    plt.close()
    print(f"[PLOT] {PLOT_LATENCY.name}")


def plot_ingest_vs_length(ingest_rows):
    """Ingestion time vs extracted text length (chars)."""
    xs = [r["text_chars"] for r in ingest_rows]
    ys = [r["total_s"] for r in ingest_rows]
    mods = [r["modality"] for r in ingest_rows]
    colors = {"text": "#1f77b4", "image": "#ff7f0e", "audio": "#2ca02c"}
    plt.figure(figsize=(7, 4.5))
    for m in sorted(set(mods)):
        px = [x for x, mm in zip(xs, mods) if mm == m]
        py = [y for y, mm in zip(ys, mods) if mm == m]
        plt.scatter(px, py, c=colors.get(m, "gray"), label=m, s=80,
                    edgecolors="black", linewidth=0.5)
    if len(xs) >= 2:
        z = np.polyfit(xs, ys, 1)
        xxs = np.linspace(min(xs), max(xs), 50)
        plt.plot(xxs, np.polyval(z, xxs), "--", color="gray", alpha=0.7,
                 label="linear fit")
    plt.xlabel("Extracted text length (characters)")
    plt.ylabel("Ingestion time (s)")
    plt.title("Ingestion Time vs Extracted Text Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_INGEST, dpi=150)
    plt.close()
    print(f"[PLOT] {PLOT_INGEST.name}")


def plot_pca_embeddings():
    """PCA 2D projection of BGE embeddings in Chroma, colored by modality."""
    col = _get_collection()
    res = col.get(include=["embeddings", "metadatas"])
    raw_embs = res.get("embeddings")
    embs = np.asarray(raw_embs) if raw_embs is not None else np.zeros((0, 0))
    metas = res.get("metadatas") or []
    if embs.size == 0:
        print("[PCA] collection empty, skipping")
        return
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(embs)
    colors = {"text": "#1f77b4", "image": "#ff7f0e", "audio": "#2ca02c"}
    plt.figure(figsize=(7, 5.5))
    for m in sorted({(md or {}).get("modality", "unknown") for md in metas}):
        pts = np.array([xy[i] for i, md in enumerate(metas)
                        if (md or {}).get("modality", "unknown") == m])
        if len(pts) == 0:
            continue
        plt.scatter(pts[:, 0], pts[:, 1], c=colors.get(m, "gray"),
                    label=f"{m} (n={len(pts)})", s=90,
                    edgecolors="black", linewidth=0.5, alpha=0.85)
    ev = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    plt.title(f"PCA of BGE Embeddings (n={len(embs)}, dim=384)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PCA, dpi=150)
    plt.close()
    print(f"[PLOT] {PLOT_PCA.name}  explained_var={ev.tolist()}")
    return float(ev[0]), float(ev[1]), int(len(embs))


def main():
    ingest_rows = bench_ingestion()
    latency_rows = bench_query_latency()
    pca_info = plot_pca_embeddings()

    if ingest_rows:
        plot_latency_vs_size(ingest_rows)
        plot_ingest_vs_length(ingest_rows)

    metrics = {
        "ingestion": ingest_rows,
        "query_latency": latency_rows,
        "pca": {
            "pc1_var_ratio": pca_info[0] if pca_info else None,
            "pc2_var_ratio": pca_info[1] if pca_info else None,
            "n_points": pca_info[2] if pca_info else 0,
        },
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"[OK] metrics -> {METRICS_JSON}")


if __name__ == "__main__":
    os.chdir(REPO)
    main()
