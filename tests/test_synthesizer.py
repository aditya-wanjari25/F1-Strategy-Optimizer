"""Test the Synthesizer Agent with pre-populated state."""

from agents.synthesizer import run_synthesizer
from graph.state import TireAnalysis, WeatherAnalysis, CompetitorAnalysis

YEAR = 2023
GP = "Bahrain"
DRIVER = "VER"


def test_run_synthesizer():
    # Pre-populate state as if all three specialist agents have already run
    state = {
        "year": YEAR,
        "grand_prix": GP,
        "driver": DRIVER,
        "tire_analysis": TireAnalysis(
            driver=DRIVER,
            stints=[],
            recommended_compounds=["SOFT", "HARD"],
            optimal_pit_laps=[35],
            summary=(
                "The SOFT compound performed well for the first stint with minimal "
                "degradation. Switching to HARD at lap 35 provided strong pace to the finish."
            ),
        ),
        "weather_analysis": WeatherAnalysis(
            has_rain=False,
            track_temp_trend="falling",
            risk_level="low",
            summary=(
                "No rainfall detected. Track temperature fell gradually across the race, "
                "improving grip as the race progressed. Low strategic risk."
            ),
        ),
        "competitor_analysis": CompetitorAnalysis(
            undercut_opportunities=[
                {
                    "driver": "GAS",
                    "lap": 10,
                    "gap_sec": 4.0,
                    "reasoning": "GAS pitted on lap 10 while VER was within 5 seconds.",
                }
            ],
            overcut_opportunities=[],
            summary=(
                "VER had undercut opportunities vs GAS and NOR who pitted early. "
                "No overcut opportunities were present."
            ),
        ),
        "strategy_recommendation": None,
        "messages": [],
        "next_agent": "",
        "errors": [],
    }

    result = run_synthesizer(state)

    assert "strategy_recommendation" in result
    sr = result["strategy_recommendation"]
    assert len(sr.compounds) == len(sr.pit_laps) + 1
    assert sr.confidence in ["high", "medium", "low"]

    print(f"\n✅ Final Strategy for {DRIVER} at {GP} {YEAR}:")
    print(f"   Compounds  : {sr.compounds}")
    print(f"   Pit laps   : {sr.pit_laps}")
    print(f"   Confidence : {sr.confidence}")
    print(f"   Rationale  : {sr.rationale}")