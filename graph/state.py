"""
Shared state for the F1 Strategy Optimizer graph.

This is the single source of truth that flows through every agent.
Each agent reads what it needs and writes back its findings.
LangGraph merges writes automatically using the `Annotated` reducers.
"""

from typing import Annotated
from dataclasses import dataclass, field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


@dataclass
class TireAnalysis:
    driver: str
    stints: list[dict]          # raw stint data from FastF1
    recommended_compounds: list[str]
    optimal_pit_laps: list[int]
    summary: str                # human-readable finding from the Tire Agent


@dataclass
class WeatherAnalysis:
    has_rain: bool
    track_temp_trend: str       # "rising", "falling", "stable"
    risk_level: str             # "low", "medium", "high"
    summary: str


@dataclass
class CompetitorAnalysis:
    undercut_opportunities: list[dict]   # {driver, lap, gap_sec}
    overcut_opportunities: list[dict]
    summary: str


@dataclass 
class StrategyRecommendation:
    pit_laps: list[int]
    compounds: list[str]
    rationale: str
    confidence: str             # "high", "medium", "low"


class AgentState(dict):
    """
    The shared state passed between all nodes in the LangGraph graph.
    
    We subclass dict so LangGraph can merge partial updates from each agent
    without overwriting the whole state.
    """

    # ── Inputs (set once at the start) ──────────────────────────────────────
    year: int
    grand_prix: str
    driver: str

    # ── Agent findings (each agent populates its own section) ────────────────
    tire_analysis: TireAnalysis | None
    weather_analysis: WeatherAnalysis | None
    competitor_analysis: CompetitorAnalysis | None

    # ── Final output ─────────────────────────────────────────────────────────
    strategy_recommendation: StrategyRecommendation | None

    # ── Message history (LangGraph built-in — tracks agent conversation) ─────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Control flow ─────────────────────────────────────────────────────────
    next_agent: str             # supervisor uses this to route to next agent
    errors: list[str]           # any agent can append errors here
    trace: object | None        # LangFuse trace object for observability