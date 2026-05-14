"""
Eval report — renders eval results as a rich terminal report.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from evals.scorer import EvalScore

console = Console()


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "green"
    elif score >= 0.5:
        return "yellow"
    return "red"


def _flag_str(score: EvalScore) -> str:
    flags = []
    if score.dnf:        flags.append("DNF")
    if score.safety_car: flags.append("SC")
    if score.rain:       flags.append("RAIN")
    return " ".join(flags) if flags else "—"


def print_report(scores: list[EvalScore]):
    """Renders a full eval report to the terminal."""

    console.print()
    console.print(Panel(
        "[bold white]F1 Strategy Optimizer — Eval Report[/bold white]\n"
        f"[dim]{len(scores)} cases evaluated[/dim]",
        style="bold cyan",
        box=box.DOUBLE,
    ))

    # ── Per-case results table ─────────────────────────────────────────────
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold white")
    table.add_column("Race",              style="dim",  width=14)
    table.add_column("Driver",            width=8)
    table.add_column("Circuit",           width=10)
    table.add_column("Compounds",         width=10)
    table.add_column("Pit Δ",             width=8)
    table.add_column("Stints",            width=8)
    table.add_column("Overall",           width=10)
    table.add_column("Flags",             width=10)

    for s in sorted(scores, key=lambda x: x.overall_score, reverse=True):
        color = _score_color(s.overall_score)
        stint_str = "✅" if s.stint_count_match else "❌"

        table.add_row(
            f"{s.grand_prix} {s.year}",
            s.driver,
            s.circuit_type,
            f"{s.compound_accuracy:.0%}",
            f"{s.pit_lap_delta:.1f} laps",
            stint_str,
            f"[{color}]{s.overall_score:.2f}[/{color}]",
            f"[dim]{_flag_str(s)}[/dim]",
        )

    console.print(table)

    # ── Aggregate stats ────────────────────────────────────────────────────
    avg_overall   = sum(s.overall_score      for s in scores) / len(scores)
    avg_compounds = sum(s.compound_accuracy  for s in scores) / len(scores)
    avg_pit_delta = sum(s.pit_lap_delta      for s in scores) / len(scores)
    stint_matches = sum(1 for s in scores if s.stint_count_match)

    console.print(Panel(
        f"[bold white]Aggregate Results[/bold white]\n\n"
        f"Overall score    : [{_score_color(avg_overall)}]{avg_overall:.2f}[/{_score_color(avg_overall)}]\n"
        f"Compound accuracy: [{_score_color(avg_compounds)}]{avg_compounds:.0%}[/{_score_color(avg_compounds)}]\n"
        f"Avg pit lap delta: {avg_pit_delta:.1f} laps\n"
        f"Stint count match: {stint_matches}/{len(scores)}",
        style="cyan",
        box=box.HEAVY,
    ))

    # ── Breakdown by circuit type ──────────────────────────────────────────
    console.print("\n[bold white]Breakdown by Circuit Type[/bold white]")
    circuit_types = sorted(set(s.circuit_type for s in scores))

    for ct in circuit_types:
        ct_scores = [s for s in scores if s.circuit_type == ct]
        avg = sum(s.overall_score for s in ct_scores) / len(ct_scores)
        color = _score_color(avg)
        console.print(
            f"  {ct:<12} {len(ct_scores)} cases   "
            f"avg [{color}]{avg:.2f}[/{color}]"
        )

    # ── Failure analysis ───────────────────────────────────────────────────
    failures = [s for s in scores if s.overall_score < 0.5]
    if failures:
        console.print("\n[bold red]⚠ Low scoring cases (< 0.50):[/bold red]")
        for s in failures:
            console.print(
                f"  [red]{s.grand_prix} {s.year} {s.driver}[/red] — "
                f"score: {s.overall_score:.2f} | "
                f"predicted: {s.recommendation} | "
                f"actual: {s.ground_truth}"
            )
    else:
        console.print("\n[green]✅ No low scoring cases[/green]")

    console.print()