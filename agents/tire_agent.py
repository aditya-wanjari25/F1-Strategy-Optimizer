from dotenv import load_dotenv
load_dotenv()

import json
import time
import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState, TireAnalysis
from tools.fastf1_tools import get_driver_stints, get_race_context
from observability.tracing import agent_observation, generation_observation
from tools.retry import safe_llm_call

log = structlog.get_logger()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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


def format_stint_data(year: int, grand_prix: str, driver: str) -> str:
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


def run_tire_agent(state: AgentState) -> dict:
    year       = state["year"]
    grand_prix = state["grand_prix"]
    driver     = state["driver"]

    log.info("tire_agent_start", driver=driver, grand_prix=grand_prix, year=year)

    try:
        with agent_observation("tire_agent", {"driver": driver, "grand_prix": grand_prix, "year": year}) as obs:

            # 1. Fetch and format data
            stint_text = format_stint_data(year, grand_prix, driver)

            # 2. LLM call — wrapped in a generation observation
            messages = [
                SystemMessage(content=TIRE_SYSTEM_PROMPT),
                HumanMessage(content=stint_text),
            ]

            with generation_observation("tire_agent", "gpt-4o-mini", stint_text) as gen:
                parsed, raw = safe_llm_call(llm, messages, "tire_agent")
                gen.update(output=raw)

            # 3. Parse response
            # parsed = json.loads(raw)
            stints = get_driver_stints(year, grand_prix, driver)
            tire_analysis = TireAnalysis(
                driver=driver,
                stints=[s.__dict__ for s in stints],
                recommended_compounds=parsed["recommended_compounds"],
                optimal_pit_laps=parsed["optimal_pit_laps"],
                summary=parsed["summary"],
            )

            # 4. Update agent observation with output
            obs.update(output={
                "compounds": tire_analysis.recommended_compounds,
                "pit_laps":  tire_analysis.optimal_pit_laps,
                "summary":   tire_analysis.summary,
            })

            log.info(
                "tire_agent_complete",
                compounds=tire_analysis.recommended_compounds,
                pit_laps=tire_analysis.optimal_pit_laps,
            )

            return {"tire_analysis": tire_analysis}

    except Exception as e:
        log.error("tire_agent_error", error=str(e))
        return {"errors": state.get("errors", []) + [f"TireAgent: {str(e)}"]}