import typer
from rich.console import Console
from ingestion.ingest_manager import ingest_file, ingest_directory
from pipeline.chunker import chunk_documents
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
    add_chunks(chunks, vectors)
    console.print(f"[green]Done. Ingested {len(chunks)} chunks.[/green]")


@app.command()
def query(q: str):
    """Query the LMIF system."""
    console.print(f"\n[cyan]Query:[/cyan] {q}\n")
    chunks = retrieve(q)
    answer, used = generate(q, chunks)
    console.print(f"[bold green]Answer:[/bold green]\n{answer}\n")
    citations = format_citations(used)
    if citations:
        console.print("[bold yellow]Citations:[/bold yellow]")
        for c in citations:
            console.print(f"  - {c}")


if __name__ == "__main__":
    app()
