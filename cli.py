import os
from typing import Optional

import typer
from rich.console import Console

from logging_config import setup_logging
setup_logging()

from ingestion.ingest_manager import ingest_file, ingest_directory
from pipeline.chunker import chunk_documents, deduplicate_chunks
from pipeline.embedder import embed_chunks
from vectorstore.store import (
    add_chunks, get_count, clear_collection,
    delete_by_source, modality_breakdown,
)
from retrieval.retriever import retrieve
from generation.generator import generate
from generation.citation_formatter import format_citations

app = typer.Typer()
console = Console()


@app.command()
def ingest(path: str):
    """Ingest a file or directory into LMIF."""
    if not os.path.exists(path):
        console.print(f"[red]Path does not exist: {path}[/red]")
        return
    try:
        if os.path.isdir(path):
            results = ingest_directory(path)
        else:
            results = [ingest_file(path)]
    except Exception as e:
        console.print(f"[red]Ingestion failed: {e}[/red]")
        return

    raw = [c for r in results for c in r.chunks]
    cached_count = sum(1 for r in results if r.from_cache and r.chunks)
    if cached_count:
        console.print(f"[dim]Cache hits: {cached_count}/{len(results)} file(s)[/dim]")

    if not raw:
        console.print("[yellow]No ingestable content found (empty folder or unsupported files).[/yellow]")
        return
    chunks = chunk_documents(raw)
    if not chunks:
        console.print("[yellow]No text extracted or all segments were empty after chunking.[/yellow]")
        return
    vectors = embed_chunks(chunks)
    # Deduplicate near-identical chunks
    before_count = len(chunks)
    chunks, vectors = deduplicate_chunks(chunks, vectors)
    if before_count > len(chunks):
        console.print(f"[dim]Deduplication: {before_count} → {len(chunks)} chunks[/dim]")
    add_chunks(chunks, vectors)
    console.print(f"[green]Done. Ingested {len(chunks)} chunks.[/green]")


@app.command()
def query(
    q: str,
    modality: Optional[str] = typer.Option(None, help="Filter by modality: text, image, audio"),
    hyde: bool = typer.Option(False, help="Use HyDE query expansion for better recall"),
):
    """Query the LMIF system with hybrid dense+sparse retrieval."""
    console.print(f"\n[cyan]Query:[/cyan] {q}")
    if modality:
        console.print(f"[dim]Modality filter: {modality}[/dim]")
    if hyde:
        console.print("[dim]HyDE query expansion: enabled[/dim]")
    console.print()

    if get_count() == 0:
        console.print("[yellow]No documents indexed yet. Run 'ingest' first.[/yellow]")
        return

    chunks = retrieve(q, modality_filter=modality, use_hyde=hyde)
    console.print(f"[dim]Retrieved {len(chunks)} chunks[/dim]")
    for i, (text, meta) in enumerate(chunks[:3], 1):
        mod = meta.get("modality", "?").upper()
        console.print(f"[dim]  Chunk {i} [{mod}]: {text[:100]}...[/dim]")
    console.print()

    answer, used = generate(q, chunks)
    # Format answer with citation footer
    formatted = format_citations(answer, used)
    console.print(f"[bold green]Answer:[/bold green]\n{formatted}\n")


@app.command()
def stats():
    """Show collection stats: total chunks and modality breakdown."""
    total = get_count()
    console.print(f"[cyan]Total chunks:[/cyan] {total}")
    if total == 0:
        return
    breakdown = modality_breakdown()
    for mod, n in sorted(breakdown.items(), key=lambda x: -x[1]):
        console.print(f"  [dim]{mod:<10}[/dim] {n}")


@app.command()
def clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
):
    """Clear all indexed data from the collection."""
    total = get_count()
    if total == 0:
        console.print("[yellow]Collection already empty.[/yellow]")
        return
    if not yes:
        confirm = typer.confirm(f"Delete all {total} chunks?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return
    clear_collection()
    console.print("[green]Collection cleared.[/green]")


@app.command()
def forget(source: str):
    """Remove all chunks originating from SOURCE (file path)."""
    n = delete_by_source(source)
    if n == 0:
        console.print(f"[yellow]No chunks found for: {source}[/yellow]")
    else:
        console.print(f"[green]Deleted {n} chunks from: {source}[/green]")


if __name__ == "__main__":
    app()
