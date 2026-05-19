"""
Shared state for the F1 Strategy Optimizer graph.

For parallel agent execution, each agent output field needs a reducer
that tells LangGraph how to merge concurrent writes safely.
"""

from typing import Annotated, Any
from dataclasses import dataclass
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# ── Reducer for parallel agent outputs ───────────────────────────────────────
def keep_latest(current: Any, update: Any) -> Any:
    """
    Reducer that keeps the latest non-None write.
    Used for fields that only one agent writes to —
    prevents a parallel agent returning None from wiping
    a completed agent's output.
    """
    if update is None:
        return current
    return update


# ── Agent output dataclasses ──────────────────────────────────────────────────
@dataclass
class TireAnalysis:
    driver: str
    stints: list[dict]
    recommended_compounds: list[str]
    optimal_pit_laps: list[int]
    summary: str


@dataclass
class WeatherAnalysis:
    has_rain: bool
    track_temp_trend: str       # "rising", "falling", "stable"
    risk_level: str             # "low", "medium", "high"
    summary: str


@dataclass
class CompetitorAnalysis:
    undercut_opportunities: list[dict]
    overcut_opportunities: list[dict]
    summary: str


@dataclass
class StrategyRecommendation:
    pit_laps: list[int]
    compounds: list[str]
    rationale: str
    confidence: str             # "high", "medium", "low"


# ── Errors reducer ────────────────────────────────────────────────────────────
def merge_errors(current: list, update: list) -> list:
    """Merges error lists from parallel agents without losing any."""
    if not update:
        return current
    return current + update


# ── Shared state ──────────────────────────────────────────────────────────────
class AgentState(dict):
    # ── Inputs ───────────────────────────────────────────────────────────────
    year: int
    grand_prix: str
    driver: str

    # ── Agent outputs — each uses keep_latest reducer ─────────────────────────
    tire_analysis:      Annotated[TireAnalysis | None,      keep_latest]
    weather_analysis:   Annotated[WeatherAnalysis | None,   keep_latest]
    competitor_analysis: Annotated[CompetitorAnalysis | None, keep_latest]
    strategy_recommendation: Annotated[StrategyRecommendation | None, keep_latest]

    # ── Message history ───────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Control flow ──────────────────────────────────────────────────────────
    next_agent: str
    errors: Annotated[list[str], merge_errors]

    mode: str # "analysis" = post-race | "prediction" = pre-race historical