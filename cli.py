import typer
from typing import Optional
from rich.console import Console
from ingestion.ingest_manager import ingest_file, ingest_directory
from pipeline.chunker import chunk_documents, deduplicate_chunks
from pipeline.embedder import embed_chunks
from vectorstore.store import add_chunks
from retrieval.retriever import retrieve
from generation.generator import generate
from generation.citation_formatter import format_citations
import os

app = typer.Typer()
console = Console()


@app.command()
def ingest(path: str):
    """Ingest a file or directory into LMIF."""
    raw = ingest_directory(path) if os.path.isdir(path) else ingest_file(path)
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
        console.print(f"[dim]HyDE query expansion: enabled[/dim]")
    console.print()

    chunks = retrieve(q, modality_filter=modality, use_hyde=hyde)
    answer, used = generate(q, chunks)
    # Format answer with citation footer
    formatted = format_citations(answer, used)
    console.print(f"[bold green]Answer:[/bold green]\n{formatted}\n")


if __name__ == "__main__":
    app()
