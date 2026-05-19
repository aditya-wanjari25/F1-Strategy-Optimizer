"""
Competitor Agent — Identifies undercut and overcut opportunities.

Responsibilities:
  - Fetch pit stop data for all drivers via FastF1 tools
  - Compare rival pit windows against our driver's strategy
  - Identify undercut/overcut opportunities
  - Write a structured CompetitorAnalysis back to AgentState
"""

from dotenv import load_dotenv
load_dotenv()

import json
import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState, CompetitorAnalysis
from tools.fastf1_tools import get_pit_stop_summary, get_driver_laps, get_race_context
from observability.tracing import agent_observation, generation_observation
from tools.retry import safe_llm_call
from tools.historical_tools import get_historical_pit_summary

log = structlog.get_logger()

# ── LLM setup ────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── System Prompt ─────────────────────────────────────────────────────────────
COMPETITOR_SYSTEM_PROMPT = """
You are an expert F1 race strategist specializing in competitor analysis.

You will be given:
1. Our driver's lap-by-lap data including their pit stop lap
2. A summary of when all other drivers pitted

Your job is to identify undercut and overcut opportunities and return a JSON object
with exactly this shape:
{
  "undercut_opportunities": [
    {"driver": "HAM", "lap": 28, "gap_sec": 2.1, "reasoning": "brief explanation"}
  ],
  "overcut_opportunities": [
    {"driver": "LEC", "lap": 32, "gap_sec": 1.8, "reasoning": "brief explanation"}
  ],
  "summary": "2-3 sentence overall assessment of the competitive situation"
}

STRICT RULES:
- Undercut opportunity: a rival pitted within 3 laps AFTER our driver could have pitted,
  where our driver was within 5 seconds of that rival
- Overcut opportunity: a rival pitted within 3 laps BEFORE our driver could have pitted,
  where our driver was within 5 seconds of that rival  
- Only include opportunities where the gap makes it genuinely viable
- If no opportunities exist for a category return an empty list []
- Return ONLY the JSON object, no markdown, no explanation outside it
"""


# ── Data Formatter ────────────────────────────────────────────────────────────
def format_competitor_data(year: int, grand_prix: str, driver: str, mode: str = "analysis") -> str:
    context  = get_race_context(year, grand_prix)

    if mode == "prediction":
        all_pits = get_historical_pit_summary(grand_prix, lookback_years=3, reference_year=year)
        header = f"Race: {grand_prix} {year} (PRE-RACE PREDICTION — historical pit windows)"
    else:
        our_laps = get_driver_laps(year, grand_prix, driver)
        all_pits = get_pit_stop_summary(year, grand_prix)
        header = f"Race: {grand_prix} {year}"

    rival_pits = all_pits[all_pits["Driver"] != driver][["Driver", "PitLap", "NewCompound"]]

    lines = [
        header,
        f"Our driver: {driver}",
        f"Total laps: {context.total_laps}",
        "",
        "Typical rival pit windows (historical):" if mode == "prediction" else "Rival pit stops:",
    ]

    # Show distribution of pit laps per compound in prediction mode
    if mode == "prediction":
        for compound, group in rival_pits.groupby("NewCompound"):
            avg_lap = round(group["PitLap"].mean())
            min_lap = int(group["PitLap"].min())
            max_lap = int(group["PitLap"].max())
            lines.append(
                f"  → {compound}: typically lap {avg_lap} (range {min_lap}-{max_lap})"
            )
    else:
        our_laps = get_driver_laps(year, grand_prix, driver)
        our_pit_laps = our_laps[our_laps["PitOutTime"].notna()]["LapNumber"].tolist()
        lines.append(f"Our pit lap(s): {our_pit_laps if our_pit_laps else 'No pit stop detected'}")
        lines.append("")
        for _, row in rival_pits.iterrows():
            lines.append(f"  {row['Driver']} pitted on lap {int(row['PitLap'])} → {row['NewCompound']}")

    return "\n".join(lines)

# ── Agent Runner ──────────────────────────────────────────────────────────────
def run_competitor_agent(state: AgentState) -> dict:
    """
    LangGraph node function for the Competitor Agent.
    Reads from state, calls LLM, writes CompetitorAnalysis back to state.
    """
    year       = state["year"]
    grand_prix = state["grand_prix"]
    driver     = state["driver"]
    mode       = state.get("mode", "analysis")

    log.info("competitor_agent_start", driver=driver, grand_prix=grand_prix, year=year)

    try:
        with agent_observation("competitor_agent", {"driver": driver, "grand_prix": grand_prix, "year": year}) as obs:

            competitor_text = format_competitor_data(year, grand_prix, driver, mode)

            messages = [
                SystemMessage(content=COMPETITOR_SYSTEM_PROMPT),
                HumanMessage(content=competitor_text),
            ]

            with generation_observation("competitor_agent", "gpt-4o-mini", competitor_text) as gen:
                parsed, raw = safe_llm_call(llm, messages, "competitor_agent")
                gen.update(output=raw)

            # parsed = json.loads(raw)
            competitor_analysis = CompetitorAnalysis(
                undercut_opportunities=parsed["undercut_opportunities"],
                overcut_opportunities=parsed["overcut_opportunities"],
                summary=parsed["summary"],
            )

            obs.update(output={
                "undercuts": len(competitor_analysis.undercut_opportunities),
                "overcuts":  len(competitor_analysis.overcut_opportunities),
            })

            log.info(
                "competitor_agent_complete",
                undercuts=len(competitor_analysis.undercut_opportunities),
                overcuts=len(competitor_analysis.overcut_opportunities),
            )

            return {"competitor_analysis": competitor_analysis}

    except Exception as e:
        log.error("competitor_agent_error", error=str(e))
        return {"errors": state.get("errors", []) + [f"CompetitorAgent: {str(e)}"]}