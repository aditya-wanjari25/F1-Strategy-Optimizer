"""
F1 Race Strategy Optimizer — CLI entrypoint.

Usage:
    uv run python main.py --year 2023 --gp Bahrain --driver VER
    uv run python main.py --year 2022 --gp Monaco --driver LEC
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from graph.workflow import run_graph

console = Console()


def print_strategy_report(result: dict, year: int, gp: str, driver: str):
    """Renders the final strategy recommendation as a rich terminal report."""

    console.print()
    console.print(Panel(
        f"[bold white]F1 Race Strategy Optimizer[/bold white]\n"
        f"[dim]{gp} Grand Prix {year} — Driver: {driver}[/dim]",
        style="bold cyan",
        box=box.DOUBLE,
    ))

    # ── Errors ────────────────────────────────────────────────────────────────
    errors = result.get("errors", [])
    if errors:
        console.print("\n[bold red]⚠ Errors during analysis:[/bold red]")
        for e in errors:
            console.print(f"  [red]• {e}[/red]")
        return

    # ── Tire Analysis ─────────────────────────────────────────────────────────
    tire = result.get("tire_analysis")
    if tire:
        console.print("\n[bold yellow]🔧 Tyre Analysis[/bold yellow]")
        console.print(f"  Compounds : [cyan]{' → '.join(tire.recommended_compounds)}[/cyan]")
        console.print(f"  Pit laps  : [cyan]{tire.optimal_pit_laps}[/cyan]")
        console.print(f"  [dim]{tire.summary}[/dim]")

    # ── Weather Analysis ──────────────────────────────────────────────────────
    weather = result.get("weather_analysis")
    if weather:
        risk_color = {"low": "green", "medium": "yellow", "high": "red"}[weather.risk_level]
        console.print("\n[bold blue]🌡  Weather Analysis[/bold blue]")
        console.print(f"  Rain      : [cyan]{weather.has_rain}[/cyan]")
        console.print(f"  Temp trend: [cyan]{weather.track_temp_trend}[/cyan]")
        console.print(f"  Risk      : [{risk_color}]{weather.risk_level.upper()}[/{risk_color}]")
        console.print(f"  [dim]{weather.summary}[/dim]")

    # ── Competitor Analysis ───────────────────────────────────────────────────
    competitor = result.get("competitor_analysis")
    if competitor:
        console.print("\n[bold magenta]🏁 Competitor Analysis[/bold magenta]")

        if competitor.undercut_opportunities:
            console.print("  Undercut opportunities:")
            for u in competitor.undercut_opportunities:
                console.print(
                    f"    [cyan]{u['driver']}[/cyan] on lap {u['lap']} "
                    f"(gap: {u['gap_sec']}s) — {u['reasoning']}"
                )
        else:
            console.print("  Undercut opportunities: [dim]none[/dim]")

        if competitor.overcut_opportunities:
            console.print("  Overcut opportunities:")
            for o in competitor.overcut_opportunities:
                console.print(
                    f"    [cyan]{o['driver']}[/cyan] on lap {o['lap']} "
                    f"(gap: {o['gap_sec']}s) — {o['reasoning']}"
                )
        else:
            console.print("  Overcut opportunities: [dim]none[/dim]")

        console.print(f"  [dim]{competitor.summary}[/dim]")

    # ── Final Recommendation ──────────────────────────────────────────────────
    sr = result.get("strategy_recommendation")
    if sr:
        confidence_color = {"high": "green", "medium": "yellow", "low": "red"}[sr.confidence]

        # Build strategy table
        table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold white")
        table.add_column("Stint", style="dim", width=8)
        table.add_column("Compound", style="cyan", width=12)
        table.add_column("Pit on lap", style="cyan", width=12)

        for i, compound in enumerate(sr.compounds):
            pit_lap = str(sr.pit_laps[i]) if i < len(sr.pit_laps) else "—  (finish)"
            table.add_row(f"Stint {i+1}", compound, pit_lap)

        console.print("\n")
        console.print(Panel(
            f"[bold white]RECOMMENDED STRATEGY[/bold white]\n\n"
            f"Confidence: [{confidence_color}]{sr.confidence.upper()}[/{confidence_color}]\n\n"
            f"{sr.rationale}",
            style=confidence_color,
            box=box.HEAVY,
        ))
        console.print(table)

    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="F1 Race Strategy Optimizer — powered by FastF1 + LangGraph"
    )
    parser.add_argument("--year",   type=int, required=True, help="Race year e.g. 2023")
    parser.add_argument("--gp",     type=str, required=True, help="Grand Prix name e.g. Bahrain")
    parser.add_argument("--driver", type=str, required=True, help="Driver code e.g. VER")
    parser.add_argument(
    "--mode",
    type=str,
    default="analysis",
    choices=["analysis", "prediction"],
    help="analysis = post-race | prediction = pre-race historical"
)

    args = parser.parse_args()

    console.print(f"\n[dim]Running strategy analysis for {args.driver} — {args.gp} {args.year}...[/dim]")

    result = run_graph(args.year, args.gp, args.driver,args.mode)
    print_strategy_report(result, args.year, args.gp, args.driver)


if __name__ == "__main__":
    main()