"""Test the Tire Agent against real FastF1 data."""

from agents.tire_agent import format_stint_data, run_tire_agent

YEAR = 2023
GP = "Bahrain"
DRIVER = "VER"


def test_format_stint_data():
    text = format_stint_data(YEAR, GP, DRIVER)
    assert "Stint" in text
    assert DRIVER in text
    print(f"\n✅ Formatted stint data:\n{text}")


def test_run_tire_agent():
    state = {
        "year": YEAR,
        "grand_prix": GP,
        "driver": DRIVER,
        "tire_analysis": None,
        "weather_analysis": None,
        "competitor_analysis": None,
        "strategy_recommendation": None,
        "messages": [],
        "next_agent": "",
        "errors": [],
    }

    result = run_tire_agent(state)

    assert "tire_analysis" in result
    ta = result["tire_analysis"]
    assert ta.driver == DRIVER
    assert len(ta.recommended_compounds) > 0
    assert len(ta.optimal_pit_laps) > 0
    print(f"\n✅ Tire Analysis for {DRIVER}:")
    print(f"   Compounds : {ta.recommended_compounds}")
    print(f"   Pit laps  : {ta.optimal_pit_laps}")
    print(f"   Summary   : {ta.summary}")