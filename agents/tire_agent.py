"""
Tire Agent — Analyzes tyre strategy for a given driver.

Responsibilities:
  - Fetch stint data via FastF1 tools
  - Reason about compound choices and degradation using an LLM
  - Write a structured TireAnalysis back to AgentState

This agent does ONE thing and does it well.
"""

import json
import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState, TireAnalysis
from tools.fastf1_tools import get_driver_stints, get_race_context
from dotenv import load_dotenv

load_dotenv()

log = structlog.get_logger()

# ── LLM setup ────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── System Prompt ─────────────────────────────────────────────────────────────
TIRE_SYSTEM_PROMPT = """
You are an expert F1 tyre strategist working for a top constructor.

You will be given stint-by-stint tyre data for a driver in a race, including:
- Which compound was used (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
- How many laps each stint lasted
- Average lap time per stint in seconds
- Degradation rate (seconds lost per lap as the tyre wears)

Your job is to analyze this data and return a JSON object with exactly this shape:
{
  "recommended_compounds": ["SOFT", "MEDIUM"],
  "optimal_pit_laps": [18, 38],
  "summary": "A clear 2-3 sentence explanation of your recommendation"
}

STRICT RULES:
- Base ALL recommendations strictly on the stint data provided. Do not use general F1 knowledge.
- recommended_compounds lists the compound for each stint in order
- optimal_pit_laps lists the lap number WHEN YOU PIT — always exactly len(recommended_compounds) - 1 values
- Example: 2 compounds = 1 pit stop = 1 pit lap. 3 compounds = 2 pit stops = 2 pit laps.
- If the data shows N stints, recommend exactly N compounds and N-1 pit laps
- Return ONLY the JSON object, no markdown, no explanation outside it
"""


# ── Data Formatter ────────────────────────────────────────────────────────────
def format_stint_data(year: int, grand_prix: str, driver: str) -> str:
    """
    Fetches stint data and formats it into a clean string for the LLM.
    The LLM doesn't need raw dataframes — it needs structured, readable context.
    """
    context = get_race_context(year, grand_prix)
    stints = get_driver_stints(year, grand_prix, driver)

    lines = [
        f"Race: {grand_prix} {year}",
        f"Driver: {driver}",
        f"Total race laps: {context.total_laps}",
        f"Number of stints: {len(stints)}",
        "",
        "Stint breakdown:",
    ]

    for s in stints:
        lines.append(
            f"  Stint {s.stint_number}: {s.compound} | "
            f"Laps {s.start_lap}-{s.end_lap} ({s.lap_count} laps) | "
            f"Avg lap: {s.avg_lap_time_sec}s | "
            f"Degradation: {s.degradation_sec_per_lap}s/lap"
        )

    return "\n".join(lines)


# ── Agent Runner ──────────────────────────────────────────────────────────────
def run_tire_agent(state: AgentState) -> dict:
    """
    LangGraph node function for the Tire Agent.
    Reads from state, calls LLM, writes TireAnalysis back to state.
    """
    year        = state["year"]
    grand_prix  = state["grand_prix"]
    driver      = state["driver"]

    log.info("tire_agent_start", driver=driver, grand_prix=grand_prix, year=year)

    try:
        # 1. Fetch and format data from FastF1
        stint_text = format_stint_data(year, grand_prix, driver)
        log.info("tire_agent_data_ready", char_count=len(stint_text))

        # 2. Call the LLM
        messages = [
            SystemMessage(content=TIRE_SYSTEM_PROMPT),
            HumanMessage(content=stint_text),
        ]
        response = llm.invoke(messages)
        raw = response.content.strip()
        log.info("tire_agent_llm_response", raw=raw)

        # 3. Parse the JSON response
        parsed = json.loads(raw)

        # 4. Build typed TireAnalysis
        stints = get_driver_stints(year, grand_prix, driver)
        tire_analysis = TireAnalysis(
            driver=driver,
            stints=[s.__dict__ for s in stints],
            recommended_compounds=parsed["recommended_compounds"],
            optimal_pit_laps=parsed["optimal_pit_laps"],
            summary=parsed["summary"],
        )

        log.info(
            "tire_agent_complete",
            compounds=tire_analysis.recommended_compounds,
            pit_laps=tire_analysis.optimal_pit_laps,
        )

        return {"tire_analysis": tire_analysis}

    except Exception as e:
        log.error("tire_agent_error", error=str(e))
        return {"errors": state.get("errors", []) + [f"TireAgent: {str(e)}"]}