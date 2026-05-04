"""Smoke tests for the FastF1 data layer."""

from tools.fastf1_tools import (
    get_race_context,
    get_driver_laps,
    get_driver_stints,
    get_race_weather,
    get_pit_stop_summary,
)

# We'll use the 2023 Bahrain GP — reliable, well-known race
YEAR = 2023
GP = "Bahrain"
DRIVER = "VER"


def test_race_context():
    ctx = get_race_context(YEAR, GP)
    assert ctx.total_laps > 0
    assert DRIVER in ctx.drivers
    print(f"\n✅ Race: {ctx.grand_prix} {ctx.year} — {ctx.total_laps} laps, {len(ctx.drivers)} drivers")


def test_driver_laps():
    laps = get_driver_laps(YEAR, GP, DRIVER)
    assert not laps.empty
    assert "LapTimeSec" in laps.columns
    print(f"\n✅ {DRIVER} laps: {len(laps)} accurate laps fetched")
    print(laps.head(3).to_string())


def test_driver_stints():
    stints = get_driver_stints(YEAR, GP, DRIVER)
    assert len(stints) > 0
    for s in stints:
        print(f"\n  Stint {s.stint_number}: {s.compound}, laps {s.start_lap}-{s.end_lap}, "
              f"avg {s.avg_lap_time_sec}s, deg {s.degradation_sec_per_lap}s/lap")
    print(f"\n✅ {DRIVER} stints: {len(stints)} stints found")


def test_weather():
    weather = get_race_weather(YEAR, GP)
    assert not weather.empty
    print(f"\n✅ Weather: {len(weather)} snapshots")
    print(weather.head(3).to_string())


def test_pit_stops():
    pits = get_pit_stop_summary(YEAR, GP)
    assert not pits.empty
    print(f"\n✅ Pit stops: {len(pits)} total stops across all drivers")
    print(pits.head(5).to_string())