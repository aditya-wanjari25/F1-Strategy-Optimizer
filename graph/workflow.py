"""
LangGraph workflow for the F1 Race Strategy Optimizer.

This file defines:
  - The graph structure (nodes + edges)
  - How the Supervisor routes between agents
  - The entry and exit points of the system

Think of this as the circuit board — agents are the chips,
this file is the wiring between them.
"""

import structlog
from langgraph.graph import StateGraph, END
from graph.state import AgentState
from agents.tire_agent import run_tire_agent

log = structlog.get_logger()


# ── Routing Logic ────────────────────────────────────────────────────────────
def route_next(state: AgentState) -> str:
    """
    Called after the Supervisor runs.
    Reads `next_agent` from state and returns the name of the next node.

    This is LangGraph's conditional edge — instead of a fixed path,
    the graph dynamically decides where to go based on state.
    """
    next_agent = state.get("next_agent", END)
    log.info("routing", next_agent=next_agent)
    return next_agent


# ── Placeholder Nodes ────────────────────────────────────────────────────────
# These will be replaced with real agent calls in upcoming phases.
# For now they just log and pass state through unchanged.
def tire_agent_node(state: AgentState) -> dict:
    log.info("node_called", node="tire_agent")
    return {"tire_analysis": {"stub": True}}


def weather_agent_node(state: AgentState) -> dict:
    log.info("node_called", node="weather_agent")
    return {"weather_analysis": {"stub": True}}


def competitor_agent_node(state: AgentState) -> dict:
    log.info("node_called", node="competitor_agent")
    return {"competitor_analysis": {"stub": True}}


def synthesizer_node(state: AgentState) -> dict:
    log.info("node_called", node="synthesizer")
    return {"strategy_recommendation": {"stub": True}}

def supervisor_node(state: AgentState) -> dict:
    """
    The supervisor decides what has already run and what to call next.
    
    Routing order: tire → weather → competitor → synthesizer → END
    We'll replace this with an LLM-backed supervisor in Phase 3.
    For now it follows a fixed sequence so we can test the wiring.
    """
    tire_done     = state.get("tire_analysis") is not None
    weather_done  = state.get("weather_analysis") is not None
    competitor_done = state.get("competitor_analysis") is not None
    strategy_done = state.get("strategy_recommendation") is not None

    if not tire_done:
        next_agent = "tire_agent"
    elif not weather_done:
        next_agent = "weather_agent"
    elif not competitor_done:
        next_agent = "competitor_agent"
    elif not strategy_done:
        next_agent = "synthesizer"
    else:
        next_agent = END

    log.info("supervisor_routing", next_agent=next_agent)
    return {"next_agent": next_agent}


# ── Graph Assembly ────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """
    Assembles and compiles the full LangGraph workflow.

    Flow:
      supervisor → [tire_agent | weather_agent | competitor_agent | synthesizer | END]
             ↑______________|___________________|____________________|

    Every specialist agent loops back to the supervisor after finishing,
    so the supervisor always decides what runs next.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor",    supervisor_node)
    graph.add_node("tire_agent",    run_tire_agent)
    graph.add_node("weather_agent", weather_agent_node)
    graph.add_node("competitor_agent", competitor_agent_node)
    graph.add_node("synthesizer",   synthesizer_node)

    # Entry point — always start at the supervisor
    graph.set_entry_point("supervisor")

    # Supervisor has a conditional edge — it can go to any agent or END
    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "tire_agent":       "tire_agent",
            "weather_agent":    "weather_agent",
            "competitor_agent": "competitor_agent",
            "synthesizer":      "synthesizer",
            END:                END,
        }
    )

    # Every specialist loops back to supervisor when done
    for node in ["tire_agent", "weather_agent", "competitor_agent", "synthesizer"]:
        graph.add_edge(node, "supervisor")

    return graph.compile()


# ── Convenience runner ────────────────────────────────────────────────────────
def run_graph(year: int, grand_prix: str, driver: str) -> AgentState:
    """Entry point to run the full strategy optimizer."""
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
    log.info("graph_complete")
    return result