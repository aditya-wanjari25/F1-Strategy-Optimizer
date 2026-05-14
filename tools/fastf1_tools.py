"""
FastF1 Tools — Data layer for the F1 Strategy Optimizer.
These are pure data-fetching functions that will be wrapped
as LangChain tools for agent use in Phase 2.
"""

import warnings
import structlog
import fastf1
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────────────────────
log = structlog.get_logger()

# ── Cache setup ─────────────────────────────────────────────────────────────
CACHE_DIR = Path("fastf1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


# ── Data Models ─────────────────────────────────────────────────────────────
@dataclass
class StintInfo:
    stint_number: int
    compound: str
    start_lap: int
    end_lap: int
    lap_count: int
    avg_lap_time_sec: float
    degradation_sec_per_lap: float  # how much slower per lap on this stint


@dataclass
class RaceContext:
    year: int
    grand_prix: str
    session_type: str  # 'R' = Race, 'Q' = Qualifying, 'FP1/FP2/FP3'
    total_laps: int
    drivers: list[str]


# ── Session Loader ───────────────────────────────────────────────────────────
def load_session(year: int, grand_prix: str, session_type: str = "R") -> fastf1.core.Session:
    """Load and cache a FastF1 session."""
    log.info("loading_session", year=year, grand_prix=grand_prix, session_type=session_type)
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load(telemetry=False, weather=True, messages=True)
    log.info("session_loaded", drivers=len(session.drivers))
    return session


# ── Tool 1: Race Context ─────────────────────────────────────────────────────
def get_race_context(year: int, grand_prix: str) -> RaceContext:
    """
    Returns high-level race metadata.
    Agents use this to understand the race they're analyzing.
    """
    session = load_session(year, grand_prix, "R")
    drivers = [session.get_driver(d)["Abbreviation"] for d in session.drivers]

    return RaceContext(
        year=year,
        grand_prix=grand_prix,
        session_type="R",
        total_laps=session.total_laps,
        drivers=drivers,
    )


# ── Tool 2: Lap Data ─────────────────────────────────────────────────────────
def get_driver_laps(year: int, grand_prix: str, driver: str) -> pd.DataFrame:
    """
    Returns cleaned lap-by-lap data for a driver.
    Includes: LapNumber, LapTime (sec), Compound, TyreLife, PitInTime, PitOutTime, IsAccurate
    """
    session = load_session(year, grand_prix, "R")
    laps = session.laps.pick_drivers(driver).pick_accurate()

    df = laps[["LapNumber", "LapTime", "Compound", "TyreLife", "PitInTime", "PitOutTime"]].copy()
    df["LapTimeSec"] = df["LapTime"].dt.total_seconds()
    df = df.drop(columns=["LapTime"])
    df = df.reset_index(drop=True)

    log.info("laps_fetched", driver=driver, lap_count=len(df))
    return df


# ── Tool 3: Stint Analysis ───────────────────────────────────────────────────
def get_driver_stints(year: int, grand_prix: str, driver: str) -> list[StintInfo]:
    laps = get_driver_laps(year, grand_prix, driver)

    if laps.empty:
        log.warning("no_laps_found", driver=driver)
        return []

    # Detect new stint when:
    # 1. Compound changes (different tyre)
    # 2. TyreLife drops (same compound, pitted again)
    laps["NewStint"] = (
        (laps["Compound"] != laps["Compound"].shift(1)) |
        (laps["TyreLife"] < laps["TyreLife"].shift(1))
    ).fillna(True)

    laps["StintNumber"] = laps["NewStint"].cumsum()

    stints = []
    for stint_num, group in laps.groupby("StintNumber"):
        group = group.dropna(subset=["LapTimeSec"])
        if len(group) < 2:
            continue

        compound = group["Compound"].mode()[0]
        lap_times = group["LapTimeSec"].values
        deg = float(pd.Series(lap_times).diff().mean())

        stints.append(
            StintInfo(
                stint_number=int(stint_num),
                compound=compound,
                start_lap=int(group["LapNumber"].iloc[0]),
                end_lap=int(group["LapNumber"].iloc[-1]),
                lap_count=len(group),
                avg_lap_time_sec=round(float(lap_times.mean()), 3),
                degradation_sec_per_lap=round(deg, 4),
            )
        )

    log.info("stints_computed", driver=driver, stint_count=len(stints))
    return stints

# ── Tool 4: Weather Data ─────────────────────────────────────────────────────
def get_race_weather(year: int, grand_prix: str) -> pd.DataFrame:
    """
    Returns weather snapshots across the race.
    Includes: Time, AirTemp, TrackTemp, Rainfall, WindSpeed
    Used by the Weather Agent.
    """
    session = load_session(year, grand_prix, "R")
    weather = session.weather_data[
        ["Time", "AirTemp", "TrackTemp", "Rainfall", "WindSpeed"]
    ].copy()
    weather["TimeMin"] = weather["Time"].dt.total_seconds() / 60
    log.info("weather_fetched", snapshots=len(weather))
    return weather


# ── Tool 5: Competitor Pit Stop Summary ──────────────────────────────────────
def get_pit_stop_summary(year: int, grand_prix: str) -> pd.DataFrame:
    """
    Returns a summary of all pit stops in the race — who pitted, when, and what compound they switched to.
    Used by the Competitor Agent.
    """
    session = load_session(year, grand_prix, "R")
    all_laps = session.laps

    pit_laps = all_laps[all_laps["PitOutTime"].notna()][
        ["Driver", "LapNumber", "Compound", "TyreLife"]
    ].copy()

    pit_laps = pit_laps.rename(columns={"LapNumber": "PitLap", "Compound": "NewCompound"})
    pit_laps = pit_laps.sort_values("PitLap").reset_index(drop=True)

    log.info("pit_stops_fetched", total_stops=len(pit_laps))
    return pit_laps