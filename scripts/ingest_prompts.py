"""
scripts/ingest_prompts.py
─────────────────────────
CLI tool to load trading prompts from JSON into the knowledge base.

Usage
─────
  # Load seed prompts
  python scripts/ingest_prompts.py

  # Load a custom JSON file
  python scripts/ingest_prompts.py --file path/to/my_prompts.json

  # Dry run (no writes)
  python scripts/ingest_prompts.py --dry-run

  # Show current stats
  python scripts/ingest_prompts.py --stats
"""
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

app   = typer.Typer(help="Trading RAG — prompt ingestion tool")
console = Console()

DEFAULT_SEED_FILE = Path(__file__).parent.parent / "data" / "seed_prompts" / "trading_prompts.json"


@app.command()
def ingest(
    file:    Path = typer.Option(DEFAULT_SEED_FILE, help="JSON file of prompts to ingest"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse but do not write"),
    stats:   bool = typer.Option(False, "--stats",   help="Print DB stats and exit"),
):
    """Ingest trading prompts from a JSON file into the knowledge base."""
    from app.db.database import init_db, SessionLocal
    from app.db.models import TradingPrompt
    from app.retrieval.vector_store import get_vector_store

    # Initialise DB tables
    init_db()
    db = SessionLocal()

    if stats:
        _print_stats(db)
        db.close()
        raise typer.Exit()

    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    with open(file) as f:
        raw = json.load(f)

    console.print(f"\n[cyan]Loaded {len(raw)} prompts from {file.name}[/cyan]")

    if dry_run:
        console.print("[yellow]Dry run — no writes performed.[/yellow]")
        _show_preview(raw)
        db.close()
        return

    vs = get_vector_store()
    ingested = 0
    skipped  = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting prompts…", total=len(raw))

        for item in raw:
            existing = (
                db.query(TradingPrompt)
                  .filter(TradingPrompt.title == item["title"])
                  .first()
            )
            if existing:
                progress.advance(task)
                skipped += 1
                continue

            prompt = TradingPrompt(
                title=item["title"],
                content=item["content"],
                category=item["category"],
                tags=json.dumps(item.get("tags", [])),
                source=item.get("source", "seed"),
            )
            db.add(prompt)
            db.flush()

            vs.upsert(
                prompt_id=prompt.id,
                text=f"{item['title']}\n{item['content']}",
                metadata={
                    "title":     item["title"],
                    "category":  item["category"],
                    "tags":      " ".join(item.get("tags", [])),
                    "is_active": True,
                },
            )

            db.commit()
            ingested += 1
            progress.advance(task)

    console.print(f"\n[green]✓ Ingested {ingested} prompts[/green]")
    if skipped:
        console.print(f"[yellow]↷ Skipped {skipped} duplicates[/yellow]")

    _print_stats(db)
    db.close()


def _print_stats(db):
    from app.db.models import TradingPrompt
    from app.retrieval.vector_store import get_vector_store

    rows = (
        db.query(TradingPrompt.category, TradingPrompt.category)
          .filter(TradingPrompt.is_active == True)  # noqa
          .all()
    )
    counts: dict[str, int] = {}
    for (cat, _) in rows:
        counts[cat] = counts.get(cat, 0) + 1

    table = Table(title="Prompt Knowledge Base", show_header=True)
    table.add_column("Category",     style="cyan")
    table.add_column("Count",        justify="right")

    for cat, count in sorted(counts.items()):
        table.add_row(cat, str(count))

    table.add_row("[bold]TOTAL[/bold]", f"[bold]{sum(counts.values())}[/bold]")

    vs = get_vector_store()
    console.print(table)
    console.print(f"Vector store vectors: [cyan]{vs.count()}[/cyan]\n")


def _show_preview(raw):
    table = Table(title="Prompts preview")
    table.add_column("Title")
    table.add_column("Category")
    table.add_column("Tags")
    for item in raw:
        table.add_row(
            item["title"][:60],
            item["category"],
            ", ".join(item.get("tags", [])[:3]),
        )
    console.print(table)


if __name__ == "__main__":
    app()
