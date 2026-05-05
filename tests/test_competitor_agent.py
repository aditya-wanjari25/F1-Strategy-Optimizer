"""Test the Competitor Agent against real FastF1 data."""

from agents.competitor_agent import format_competitor_data, run_competitor_agent

YEAR = 2023
GP = "Bahrain"
DRIVER = "VER"


def test_format_competitor_data():
    text = format_competitor_data(YEAR, GP, DRIVER)
    assert "Rival pit stops" in text
    assert DRIVER in text
    print(f"\n✅ Competitor data:\n{text}")


def test_run_competitor_agent():
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

    result = run_competitor_agent(state)

    assert "competitor_analysis" in result
    ca = result["competitor_analysis"]
    assert isinstance(ca.undercut_opportunities, list)
    assert isinstance(ca.overcut_opportunities, list)
    print(f"\n✅ Competitor Analysis for {DRIVER} at {GP} {YEAR}:")
    print(f"   Undercuts : {ca.undercut_opportunities}")
    print(f"   Overcuts  : {ca.overcut_opportunities}")
    print(f"   Summary   : {ca.summary}")