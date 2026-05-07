"""
Weather Agent — Analyzes race weather conditions and assesses strategic risk.

Responsibilities:
  - Fetch weather data via FastF1 tools
  - Reason about temperature trends and rainfall risk using an LLM
  - Write a structured WeatherAnalysis back to AgentState
"""

from dotenv import load_dotenv
load_dotenv()

import json
import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState, WeatherAnalysis
from tools.fastf1_tools import get_race_weather
from observability.tracing import agent_observation, generation_observation
from tools.retry import safe_llm_call


log = structlog.get_logger()

# ── LLM setup ────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── System Prompt ─────────────────────────────────────────────────────────────
WEATHER_SYSTEM_PROMPT = """
You are an expert F1 weather strategist.

You will be given weather snapshots taken throughout a race, including:
- Time into the race (minutes)
- Air temperature (°C)
- Track temperature (°C)
- Rainfall (True/False)
- Wind speed (m/s)

Analyze this data and return a JSON object with exactly this shape:
{
  "has_rain": false,
  "track_temp_trend": "stable",
  "risk_level": "low",
  "summary": "2-3 sentence summary of weather conditions and strategic implications"
}

STRICT RULES:
- has_rain must be true only if Rainfall is True in ANY snapshot
- track_temp_trend must be exactly one of: "rising", "falling", "stable"
  - rising: track temp increases by more than 3°C across the race
  - falling: track temp decreases by more than 3°C across the race
  - stable: anything in between
- risk_level must be exactly one of: "low", "medium", "high"
  - low: no rain, stable temps
  - medium: no rain but significant temp swings, or light rain early that stops
  - high: rainfall during the race
- Base ALL assessments strictly on the data provided
- Return ONLY the JSON object, no markdown, no explanation outside it
"""


# ── Data Formatter ────────────────────────────────────────────────────────────
def format_weather_data(year: int, grand_prix: str) -> str:
    """
    Fetches weather snapshots and formats them for the LLM.
    We sample every 10 snapshots to keep the context concise —
    the LLM doesn't need every reading, just the trend.
    """
    weather = get_race_weather(year, grand_prix)

    # Sample to keep prompt concise — every 10th snapshot
    sampled = weather.iloc[::10].reset_index(drop=True)

    lines = [
        f"Race: {grand_prix} {year}",
        f"Total weather snapshots: {len(weather)} (showing every 10th)",
        "",
        "Weather readings:",
    ]

    for _, row in sampled.iterrows():
        lines.append(
            f"  T+{row['TimeMin']:.1f}min | "
            f"Air: {row['AirTemp']:.1f}°C | "
            f"Track: {row['TrackTemp']:.1f}°C | "
            f"Rain: {row['Rainfall']} | "
            f"Wind: {row['WindSpeed']:.1f}m/s"
        )

    return "\n".join(lines)


# ── Agent Runner ──────────────────────────────────────────────────────────────
def run_weather_agent(state: AgentState) -> dict:
    """
    LangGraph node function for the Weather Agent.
    Reads from state, calls LLM, writes WeatherAnalysis back to state.
    """
    year       = state["year"]
    grand_prix = state["grand_prix"]

    log.info("weather_agent_start", grand_prix=grand_prix, year=year)

    try:
        with agent_observation("weather_agent", {"grand_prix": grand_prix, "year": year}) as obs:

            weather_text = format_weather_data(year, grand_prix)

            messages = [
                SystemMessage(content=WEATHER_SYSTEM_PROMPT),
                HumanMessage(content=weather_text),
            ]

            with generation_observation("weather_agent", "gpt-4o-mini", weather_text) as gen:
                parsed, raw = safe_llm_call(llm, messages, "weather_agent")
                gen.update(output=raw)

            # parsed = json.loads(raw)
            weather_analysis = WeatherAnalysis(
                has_rain=parsed["has_rain"],
                track_temp_trend=parsed["track_temp_trend"],
                risk_level=parsed["risk_level"],
                summary=parsed["summary"],
            )

            obs.update(output={
                "has_rain":         weather_analysis.has_rain,
                "track_temp_trend": weather_analysis.track_temp_trend,
                "risk_level":       weather_analysis.risk_level,
            })

            log.info(
                "weather_agent_complete",
                has_rain=weather_analysis.has_rain,
                risk_level=weather_analysis.risk_level,
            )

            return {"weather_analysis": weather_analysis}

    except Exception as e:
        log.error("weather_agent_error", error=str(e))
        return {"errors": state.get("errors", []) + [f"WeatherAgent: {str(e)}"]}