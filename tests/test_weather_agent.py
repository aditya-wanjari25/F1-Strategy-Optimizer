"""Test the Weather Agent against real FastF1 data."""

from agents.weather_agent import format_weather_data, run_weather_agent

YEAR = 2023
GP = "Bahrain"


def test_format_weather_data():
    text = format_weather_data(YEAR, GP)
    assert "Air" in text
    assert "Track" in text
    print(f"\n✅ Formatted weather data:\n{text}")


def test_run_weather_agent():
    state = {
        "year": YEAR,
        "grand_prix": GP,
        "driver": "VER",
        "tire_analysis": None,
        "weather_analysis": None,
        "competitor_analysis": None,
        "strategy_recommendation": None,
        "messages": [],
        "next_agent": "",
        "errors": [],
    }

    result = run_weather_agent(state)

    assert "weather_analysis" in result
    wa = result["weather_analysis"]
    assert wa.risk_level in ["low", "medium", "high"]
    assert wa.track_temp_trend in ["rising", "falling", "stable"]
    print(f"\n✅ Weather Analysis for {GP} {YEAR}:")
    print(f"   Rain        : {wa.has_rain}")
    print(f"   Temp trend  : {wa.track_temp_trend}")
    print(f"   Risk level  : {wa.risk_level}")
    print(f"   Summary     : {wa.summary}")