"""
CLI entry point for agentforensics.

Commands

  analyze <session_id>  - run full forensic analysis, write report
  list                  - list all stored sessions
  verify  <session_id>  - check hash-chain integrity only
  export  <session_id> <path> - export raw trace as JSON
  clear   <session_id>  - delete a session DB
"""
import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text

console = Console()

# Helpers

_VERDICT_STYLE = {
    "clean":       ("green",  "[OK] CLEAN"),
    "suspicious":  ("yellow", "[!!] SUSPICIOUS"),
    "compromised": ("red",    "[XX] COMPROMISED"),
}


def _verdict_text(verdict: str) -> Text:
    style, label = _VERDICT_STYLE.get(verdict, ("white", verdict.upper()))
    return Text(label, style=f"bold {style}")


def _sanitize_path(raw: str) -> Path:
    """Resolve and return a Path; abort on traversal attempts."""
    p = Path(raw).resolve()
    return p


# CLI group
# 

@click.group()
@click.version_option(package_name="agentforensics")
def cli() -> None:
    """AgentForensics - tamper-evident LLM session forensics toolkit."""


# analyze

@cli.command()
@click.argument("session_id")
@click.option(
    "--output", "-o",
    type=click.Choice(["html", "json"], case_sensitive=False),
    default="html",
    show_default=True,
    help="Report format.",
)
@click.option(
    "--threshold", "-t",
    type=float,
    default=None,
    help="Injection detection threshold (0.0–1.0). Overrides AF_INJECT_THRESHOLD.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Output file path. Defaults to ./af_report_<session_id>.<ext>.",
)
def analyze(session_id: str, output: str, threshold: float | None, out: str | None) -> None:
    """Run full forensic analysis on SESSION_ID and write a report."""
    from .reporter import generate
    from .store import get_events

    # Validate session exists
    events = get_events(session_id)
    if not events:
        console.print(f"[red]No events found for session '{session_id}'.[/red]")
        sys.exit(1)

    # Override threshold env var if flag provided
    if threshold is not None:
        if not 0.0 <= threshold <= 1.0:
            console.print("[red]--threshold must be between 0.0 and 1.0.[/red]")
            sys.exit(1)
        os.environ["AF_INJECT_THRESHOLD"] = str(threshold)

    ext = output.lower()
    if out:
        out_path = _sanitize_path(out)
    else:
        safe_id = session_id.replace("/", "_").replace("\\", "_")[:64]
        out_path = Path(f"af_report_{safe_id}.{ext}").resolve()

    console.print(f"[bold]Analysing session[/bold] [cyan]{session_id}[/cyan] ...")

    with console.status("Running classifier and differ…"):
        report_str = generate(session_id, output=ext)

    out_path.write_text(report_str, encoding="utf-8")

    # Parse the report for display (always valid JSON internally)
    import agentforensics.reporter as _r
    from .models import ForensicReport
    report_obj = ForensicReport.model_validate_json(
        generate(session_id, output="json")
    )

    verdict_txt = _verdict_text(report_obj.verdict)
    console.print(Panel(
        f"Verdict:    {verdict_txt}\n"
        f"Confidence: [bold]{report_obj.confidence:.0%}[/bold]\n"
        f"Turns:      {report_obj.total_turns}\n"
        f"Signals:    {len(report_obj.injection_signals)}\n"
        f"Chain:      {'[green]valid[/green]' if report_obj.chain_valid else '[red]INVALID[/red]'}\n\n"
        f"[dim]{report_obj.summary}[/dim]",
        title="Forensic Report",
        border_style="cyan",
    ))
    console.print(f"[green]Report written to:[/green] [bold]{out_path}[/bold]")


# list

@cli.command(name="list")
def list_sessions() -> None:
    """List all stored sessions."""
    from .store import list_sessions as _list

    sessions = _list()
    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    table = Table(
        title="Stored Sessions",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("Session ID", style="cyan", no_wrap=True)
    table.add_column("Started At", style="dim")
    table.add_column("Model")
    table.add_column("Agent", style="dim")
    table.add_column("Turns", justify="right")

    for s in sessions:
        table.add_row(
            s.get("session_id", ""),
            s.get("started_at", "")[:19].replace("T", " "),
            s.get("model") or "-",
            s.get("agent_name") or "-",
            str(s.get("turn_count", 0)),
        )

    console.print(table)


# verify

@cli.command()
@click.argument("session_id")
def verify(session_id: str) -> None:
    """Check hash-chain integrity for SESSION_ID."""
    from .store import verify_chain, get_events

    events = get_events(session_id)
    if not events:
        console.print(f"[red]No events found for session '{session_id}'.[/red]")
        sys.exit(1)

    valid = verify_chain(session_id)
    if valid:
        console.print(
            f"[green bold][OK] Chain valid[/green bold]"
            f"{len(events)} event(s) verified for session [cyan]{session_id}[/cyan]."
        )
    else:
        console.print(
            f"[red bold][XX] Chain INVALID[/red bold]"
            f"session [cyan]{session_id}[/cyan] may have been tampered with."
        )
        sys.exit(2)



# export

@cli.command()
@click.argument("session_id")
@click.argument("path", type=click.Path(dir_okay=False, writable=True))
def export(session_id: str, path: str) -> None:
    """Export raw trace for SESSION_ID as JSON to PATH."""
    from .store import get_events

    events = get_events(session_id)
    if not events:
        console.print(f"[red]No events found for session '{session_id}'.[/red]")
        sys.exit(1)

    out_path = _sanitize_path(path)
    payload = json.dumps(
        [json.loads(e.model_dump_json()) for e in events],
        indent=2,
        ensure_ascii=False,
    )
    out_path.write_text(payload, encoding="utf-8")
    console.print(
        f"[green]Exported {len(events)} event(s) to[/green] [bold]{out_path}[/bold]"
    )


# clear

@cli.command()
@click.argument("session_id")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation.")
def clear(session_id: str, yes: bool) -> None:
    """Delete the DB for SESSION_ID (irreversible)."""
    from .store import delete_session, get_events, _db_path

    events = get_events(session_id)
    if not events:
        console.print(f"[yellow]No events found for session '{session_id}'. Nothing to clear.[/yellow]")
        return

    if not yes:
        click.confirm(
            f"Delete {len(events)} event(s) for session '{session_id}'? This cannot be undone.",
            abort=True,
        )

    delete_session(session_id)
    console.print(f"[green]Session [cyan]{session_id}[/cyan] deleted.[/green]")


# fingerprints (group)

@cli.group(name="fingerprints")
def fingerprints_group() -> None:
    """Manage attack fingerprints and campaign groupings."""


@fingerprints_group.command(name="list")
def fp_list() -> None:
    """List all stored fingerprints sorted by hit count."""
    from .fingerprinter import FingerprintStore
    from rich.table import Table
    from rich import box as _box

    fps = FingerprintStore().list_fingerprints()
    if not fps:
        console.print("[yellow]No fingerprints found.[/yellow]")
        return

    table = Table(
        title="Attack Fingerprints",
        box=_box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("ID",       style="cyan",  no_wrap=True, width=12)
    table.add_column("Hits",     justify="right")
    table.add_column("Sessions", justify="right")
    table.add_column("Campaign", style="bold")
    table.add_column("First Seen", style="dim")
    table.add_column("Last Seen",  style="dim")
    table.add_column("Snippet",    style="dim", max_width=60)

    for fp in fps:
        table.add_row(
            fp.fingerprint_id[:10] + "…",
            str(fp.hit_count),
            str(len(fp.session_ids)),
            fp.campaign_name or "[dim]-[/dim]",
            fp.first_seen.strftime("%Y-%m-%d %H:%M")[:16],
            fp.last_seen.strftime("%Y-%m-%d %H:%M")[:16],
            fp.representative_snippet[:60].replace("\n", " "),
        )

    console.print(table)


@fingerprints_group.command(name="campaigns")
def fp_campaigns() -> None:
    """Show fingerprints grouped by campaign."""
    from .fingerprinter import FingerprintStore
    from rich.table import Table
    from rich import box as _box

    campaigns = FingerprintStore().get_campaigns()
    if not campaigns:
        console.print("[yellow]No campaigns found.[/yellow]")
        return

    table = Table(
        title="Attack Campaigns",
        box=_box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("Campaign",    style="bold",  min_width=20)
    table.add_column("Hits",        justify="right")
    table.add_column("Sessions",    justify="right")
    table.add_column("First Seen",  style="dim")
    table.add_column("Last Seen",   style="dim")
    table.add_column("Top Snippet", style="dim", max_width=55)

    for c in campaigns:
        name = c["campaign_name"] if c["named"] else f"[dim]{c['campaign_name'][:16]}…[/dim]"
        table.add_row(
            name,
            str(c["total_hits"]),
            str(c["session_count"]),
            c["first_seen"][:16].replace("T", " "),
            c["last_seen"][:16].replace("T", " "),
            c["top_snippet"][:55].replace("\n", " "),
        )

    console.print(table)


@fingerprints_group.command(name="label")
@click.argument("fingerprint_id")
@click.argument("name")
def fp_label(fingerprint_id: str, name: str) -> None:
    """Attach a campaign NAME to FINGERPRINT_ID."""
    from .fingerprinter import FingerprintStore

    store = FingerprintStore()
    fps = store.list_fingerprints()
    match = next((f for f in fps if f.fingerprint_id.startswith(fingerprint_id)), None)
    if match is None:
        console.print(f"[red]No fingerprint found with ID prefix '{fingerprint_id}'.[/red]")
        sys.exit(1)

    store.label_campaign(match.fingerprint_id, name)
    console.print(
        f"[green]Fingerprint [cyan]{match.fingerprint_id[:12]}…[/cyan] "
        f"labelled as campaign '[bold]{name}[/bold]'.[/green]"
    )


# dashboard

@cli.command()
@click.option("--port", default=7890, show_default=True, help="Port to listen on.")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind to.")
def dashboard(port: int, host: str) -> None:
    """Start the web dashboard and open it in the browser."""
    try:
        from .dashboard.server import create_app
        import uvicorn
    except ImportError:
        console.print(
            "[red]Dashboard dependencies not installed.[/red]\n"
            "Run: [bold]pip install 'agentforensics[dashboard]'[/bold]"
        )
        sys.exit(1)

    import threading
    import webbrowser

    url = f"http://{host}:{port}"
    console.print(
        f"[green]Starting AgentForensics Dashboard at[/green] "
        f"[bold cyan]{url}[/bold cyan]"
    )
    console.print("[dim]Press Ctrl+C to stop.[/dim]")

    def _open_browser() -> None:
        import time
        time.sleep(1.2)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()

    app = create_app()
    uvicorn.run(app, host=host, port=port)


# Allow `python -m agentforensics.cli` invocation

if __name__ == "__main__":
    cli()
