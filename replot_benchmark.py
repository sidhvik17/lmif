"""Regenerate benchmark plots from existing metrics JSON + Chroma embeddings.
Thicker lines / markers for paper figures. No ingestion rerun.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO.parent / "lmif final documentation"
METRICS_JSON = OUT_DIR / "benchmark_metrics.json"
PLOT_LATENCY = OUT_DIR / "plot_latency_vs_size.png"
PLOT_INGEST  = OUT_DIR / "plot_ingest_vs_length.png"
PLOT_PCA     = OUT_DIR / "plot_pca_embeddings.png"

os.chdir(REPO)

metrics = json.loads(METRICS_JSON.read_text())
ingest_rows = metrics["ingestion"]

COLORS = {"text": "#1f77b4", "image": "#ff7f0e", "audio": "#2ca02c"}


def plot_latency_vs_size(rows):
    sizes = [r["size_kb"] for r in rows]
    times = [r["total_s"] for r in rows]
    mods  = [r["modality"] for r in rows]
    plt.figure(figsize=(7, 4.5))
    for m in sorted(set(mods)):
        xs = [s for s, mm in zip(sizes, mods) if mm == m]
        ys = [t for t, mm in zip(times, mods) if mm == m]
        plt.scatter(xs, ys, c=COLORS.get(m, "gray"), label=m, s=160,
                    edgecolors="black", linewidth=1.8, zorder=3)
    if len(sizes) >= 2:
        z = np.polyfit(sizes, times, 1)
        xx = np.linspace(min(sizes), max(sizes), 50)
        plt.plot(xx, np.polyval(z, xx), "--", color="gray", alpha=0.85,
                 linewidth=3.0, zorder=2,
                 label=f"linear fit (slope={z[0]:.4f} s/KB)")
    plt.xlabel("File size (KB)", fontsize=11)
    plt.ylabel("End-to-end processing latency (s)", fontsize=11)
    plt.title("Processing Latency vs File Size", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_LATENCY, dpi=150)
    plt.close()
    print(f"[PLOT] {PLOT_LATENCY}")


def plot_ingest_vs_length(rows):
    xs = [r["text_chars"] for r in rows]
    ys = [r["total_s"] for r in rows]
    mods = [r["modality"] for r in rows]
    plt.figure(figsize=(7, 4.5))
    for m in sorted(set(mods)):
        px = [x for x, mm in zip(xs, mods) if mm == m]
        py = [y for y, mm in zip(ys, mods) if mm == m]
        plt.scatter(px, py, c=COLORS.get(m, "gray"), label=m, s=160,
                    edgecolors="black", linewidth=1.8, zorder=3)
    if len(xs) >= 2:
        z = np.polyfit(xs, ys, 1)
        xxs = np.linspace(min(xs), max(xs), 50)
        plt.plot(xxs, np.polyval(z, xxs), "--", color="gray", alpha=0.85,
                 linewidth=3.0, zorder=2, label="linear fit")
    plt.xlabel("Extracted text length (characters)", fontsize=11)
    plt.ylabel("Ingestion time (s)", fontsize=11)
    plt.title("Ingestion Time vs Extracted Text Length", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_INGEST, dpi=150)
    plt.close()
    print(f"[PLOT] {PLOT_INGEST}")


def plot_pca():
    from vectorstore.store import _get_collection
    col = _get_collection()
    res = col.get(include=["embeddings", "metadatas"])
    raw_embs = res.get("embeddings")
    embs = np.asarray(raw_embs) if raw_embs is not None else np.zeros((0, 0))
    metas = res.get("metadatas") or []
    if embs.size == 0:
        print("[PCA] empty, skip")
        return None
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(embs)
    plt.figure(figsize=(7, 5.5))
    for m in sorted({(md or {}).get("modality", "unknown") for md in metas}):
        pts = np.array([xy[i] for i, md in enumerate(metas)
                        if (md or {}).get("modality", "unknown") == m])
        if len(pts) == 0:
            continue
        plt.scatter(pts[:, 0], pts[:, 1], c=COLORS.get(m, "gray"),
                    label=f"{m} (n={len(pts)})", s=180,
                    edgecolors="black", linewidth=1.8, alpha=0.9, zorder=3)
    ev = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({ev[0]*100:.1f}% var)", fontsize=11)
    plt.ylabel(f"PC2 ({ev[1]*100:.1f}% var)", fontsize=11)
    plt.title(f"PCA of BGE Embeddings (n={len(embs)}, dim=384)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PCA, dpi=150)
    plt.close()
    print(f"[PLOT] {PLOT_PCA}")
    return float(ev[0]), float(ev[1]), int(len(embs))


if __name__ == "__main__":
    plot_latency_vs_size(ingest_rows)
    plot_ingest_vs_length(ingest_rows)
    plot_pca()
    print("[OK]")
