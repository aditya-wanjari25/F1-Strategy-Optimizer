"""
Synthesizer Agent — Produces the final race strategy recommendation.

Responsibilities:
  - Read findings from Tire, Weather, and Competitor agents via state
  - Reconcile potentially conflicting signals
  - Produce a single, justified StrategyRecommendation
  
This agent never calls FastF1 directly — it only reasons over
the structured outputs of the three specialist agents.
This is what makes it a true synthesizer, not another data agent.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState, StrategyRecommendation
from observability.tracing import agent_observation, generation_observation
from tools.retry import safe_llm_call

log = structlog.get_logger()

# ── LLM setup ────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0)  # upgrade to gpt-4o — final reasoning deserves it

# ── System Prompt ─────────────────────────────────────────────────────────────
SYNTHESIZER_SYSTEM_PROMPT = """
You are the chief race strategist for a top F1 constructor.

You will receive structured findings from three specialist analysts:
1. Tyre Analyst — optimal compounds and pit lap based on degradation data
2. Weather Analyst — track conditions and strategic risk level
3. Competitor Analyst — undercut and overcut opportunities vs rivals

Your job is to synthesize all three inputs into a single optimal race strategy.
Return a JSON object with exactly this shape:
{
  "pit_laps": [35],
  "compounds": ["SOFT", "HARD"],
  "rationale": "A clear 3-4 sentence explanation that references all three analysts",
  "confidence": "high"
}

STRICT RULES:
- pit_laps and compounds must always be consistent: len(compounds) = len(pit_laps) + 1
- confidence must be exactly one of: "high", "medium", "low"
  - high: analysts agree, clear optimal strategy
  - medium: minor conflicts between analysts, reasonable trade-offs made
  - low: significant conflicts or uncertainty in the data
- If competitor analysis shows an undercut opportunity earlier than the tyre analyst's
  pit lap, evaluate whether to adopt it based on the gap and risk level
- Weather risk must influence confidence: high weather risk = maximum medium confidence
- rationale must explicitly reference findings from all three analysts
- Return ONLY the JSON object, no markdown, no explanation outside it
"""


# ── State Formatter ───────────────────────────────────────────────────────────
def format_agent_findings(state: AgentState) -> str:
    """
    Translates the structured AgentState findings into
    a readable brief for the Synthesizer LLM.
    """
    tire = state["tire_analysis"]
    weather = state["weather_analysis"]
    competitor = state["competitor_analysis"]

    lines = [
        f"Race: {state['grand_prix']} {state['year']}",
        f"Driver: {state['driver']}",
        "",
        "── TYRE ANALYST FINDINGS ──",
        f"Recommended compounds : {tire.recommended_compounds}",
        f"Optimal pit laps      : {tire.optimal_pit_laps}",
        f"Analysis              : {tire.summary}",
        "",
        "── WEATHER ANALYST FINDINGS ──",
        f"Rainfall detected     : {weather.has_rain}",
        f"Track temp trend      : {weather.track_temp_trend}",
        f"Risk level            : {weather.risk_level}",
        f"Analysis              : {weather.summary}",
        "",
        "── COMPETITOR ANALYST FINDINGS ──",
        f"Undercut opportunities: {json.dumps(weather.__dict__) if False else json.dumps(competitor.undercut_opportunities)}",
        f"Overcut opportunities : {json.dumps(competitor.overcut_opportunities)}",
        f"Analysis              : {competitor.summary}",
    ]

    return "\n".join(lines)


# ── Agent Runner ──────────────────────────────────────────────────────────────
def run_synthesizer(state: AgentState) -> dict:
    """
    LangGraph node function for the Synthesizer Agent.
    Reads all specialist findings from state, calls LLM,
    writes final StrategyRecommendation back to state.
    """
    year       = state["year"]
    grand_prix = state["grand_prix"]
    driver     = state["driver"]

    log.info("synthesizer_start", driver=driver, grand_prix=grand_prix, year=year)

    try:
        with agent_observation("synthesizer", {"driver": driver, "grand_prix": grand_prix, "year": year}) as obs:

            findings_text = format_agent_findings(state)

            messages = [
                SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
                HumanMessage(content=findings_text),
            ]

            with generation_observation("synthesizer", "gpt-4o", findings_text) as gen:
                parsed, raw = safe_llm_call(llm, messages, "synthesizer")
                gen.update(output=raw)

            # parsed = json.loads(raw)
            recommendation = StrategyRecommendation(
                pit_laps=parsed["pit_laps"],
                compounds=parsed["compounds"],
                rationale=parsed["rationale"],
                confidence=parsed["confidence"],
            )

            obs.update(output={
                "pit_laps":   recommendation.pit_laps,
                "compounds":  recommendation.compounds,
                "confidence": recommendation.confidence,
            })

            log.info(
                "synthesizer_complete",
                pit_laps=recommendation.pit_laps,
                compounds=recommendation.compounds,
                confidence=recommendation.confidence,
            )

            return {"strategy_recommendation": recommendation}

    except Exception as e:
        log.error("synthesizer_error", error=str(e))
        return {"errors": state.get("errors", []) + [f"Synthesizer: {str(e)}"]}