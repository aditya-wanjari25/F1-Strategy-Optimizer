"""
LangGraph workflow for the F1 Race Strategy Optimizer.

Phase 4: Parallel agent execution via fan-out.

Flow:
  supervisor → [tire_agent, weather_agent, competitor_agent] → synthesizer → END
                     (all three run in parallel)
"""

from dotenv import load_dotenv
load_dotenv()

import structlog
from langgraph.graph import StateGraph, END
from graph.state import AgentState
from agents.tire_agent import run_tire_agent
from agents.weather_agent import run_weather_agent
from agents.competitor_agent import run_competitor_agent
from agents.synthesizer import run_synthesizer
from observability.tracing import get_langfuse, flush

log = structlog.get_logger()


# ── Supervisor ────────────────────────────────────────────────────────────────
def supervisor_node(state: AgentState) -> dict:
    """
    In the parallel architecture the supervisor's job is simpler —
    it no longer routes one agent at a time.
    It just initializes the run and hands off to the fan-out.
    """
    log.info("supervisor_start", driver=state["driver"], grand_prix=state["grand_prix"])
    return {}


# ── Fan-out router ────────────────────────────────────────────────────────────
def route_to_specialists(_: AgentState) -> list[str]:
    """
    Returns all three specialist agent names simultaneously.
    LangGraph sees a list and fans out — runs all three in parallel.
    """
    return ["tire_agent", "weather_agent", "competitor_agent"]


# ── Graph Assembly ────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """
    Parallel fan-out graph:

      supervisor
          │
          ├──────────────────────────┐
          │                          │
      tire_agent   weather_agent   competitor_agent
          │                          │
          └──────────────────────────┘
                        │
                   synthesizer
                        │
                       END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor",       supervisor_node)
    graph.add_node("tire_agent",       run_tire_agent)
    graph.add_node("weather_agent",    run_weather_agent)
    graph.add_node("competitor_agent", run_competitor_agent)
    graph.add_node("synthesizer",      run_synthesizer)

    # Entry point
    graph.set_entry_point("supervisor")

    # Fan-out: supervisor → all three specialists in parallel
    graph.add_conditional_edges(
        "supervisor",
        route_to_specialists,
        ["tire_agent", "weather_agent", "competitor_agent"],
    )

    # Fan-in: all three specialists → synthesizer
    # LangGraph waits for ALL parallel nodes to complete before continuing
    graph.add_edge("tire_agent",       "synthesizer")
    graph.add_edge("weather_agent",    "synthesizer")
    graph.add_edge("competitor_agent", "synthesizer")

    # Synthesizer → END
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ── Convenience runner ────────────────────────────────────────────────────────
def run_graph(year: int, grand_prix: str, driver: str) -> AgentState:
    lf = get_langfuse()

    with lf.start_as_current_observation(
        name="f1_strategy_optimizer",
        as_type="agent",
        input={"year": year, "grand_prix": grand_prix, "driver": driver},
    ) as trace:

        graph = build_graph()

        initial_state: AgentState = {
            "year": year,
            "grand_prix": grand_prix,
            "driver": driver,
            "tire_analysis": None,
            "weather_analysis": None,
            "competitor_analysis": None,
            "strategy_recommendation": None,
            "messages": [],
            "next_agent": "",
            "errors": [],
        }

        log.info("graph_start", year=year, grand_prix=grand_prix, driver=driver)
        result = graph.invoke(initial_state)

        sr = result.get("strategy_recommendation")
        if sr:
            trace.update(output={
                "compounds":  sr.compounds,
                "pit_laps":   sr.pit_laps,
                "confidence": sr.confidence,
            })

        log.info("graph_complete")

    flush()
    return result