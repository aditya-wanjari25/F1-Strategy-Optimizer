"""
Historical data layer for pre-race strategy prediction.

Instead of loading a specific race, these tools aggregate
telemetry across multiple seasons at the same circuit to
derive expected degradation curves and typical strategies.

This is what enables Mode 2 — pre-race prediction.
The agents receive the same StintInfo structure they already
understand, just derived from historical averages instead
of a specific race.
"""

from dotenv import load_dotenv
load_dotenv()

import warnings
import structlog
import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
from tools.fastf1_tools import StintInfo, CACHE_DIR

warnings.filterwarnings("ignore")
log = structlog.get_logger()

fastf1.Cache.enable_cache(str(CACHE_DIR))


# ── Historical stint loader ───────────────────────────────────────────────────
def get_historical_stints(
    grand_prix: str,
    lookback_years: int = 3,
    reference_year: int = 2023,
) -> list[StintInfo]:
    """
    Aggregates stint data across multiple seasons at the same circuit.
    Returns averaged StintInfo objects the Tire Agent already understands.

    Args:
        grand_prix: circuit name e.g. "Bahrain"
        lookback_years: how many previous seasons to aggregate
        reference_year: the year we're predicting for (excluded from history)

    Returns:
        list[StintInfo] with averaged degradation per compound
    """
    years = list(range(reference_year - lookback_years, reference_year))
    log.info("historical_stints_loading", grand_prix=grand_prix, years=years)

    all_stints: list[dict] = []

    for year in years:
        try:
            session = fastf1.get_session(year, grand_prix, "R")
            session.load(telemetry=False, weather=False, messages=False)

            laps = session.laps.pick_accurate()
            if laps.empty:
                log.warning("no_accurate_laps", year=year, grand_prix=grand_prix)
                continue

            # Get lap times in seconds
            laps = laps.copy()
            laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

            # Detect stint boundaries — same logic as get_driver_stints
            laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
            laps["NewStint"] = (
                (laps["Compound"] != laps.groupby("Driver")["Compound"].shift(1)) |
                (laps["TyreLife"] < laps.groupby("Driver")["TyreLife"].shift(1))
            ).fillna(True)
            laps["StintNumber"] = laps.groupby("Driver")["NewStint"].cumsum()

            # Aggregate per compound per driver per stint
            for (driver, stint_num), group in laps.groupby(["Driver", "StintNumber"]):
                group = group.dropna(subset=["LapTimeSec"])
                if len(group) < 3:
                    continue

                compound = group["Compound"].mode()[0]
                if compound not in ["SOFT", "MEDIUM", "HARD"]:
                    continue  # skip WET/INTER for dry race prediction

                lap_times = group["LapTimeSec"].values
                deg = float(pd.Series(lap_times).diff().mean())

                all_stints.append({
                    "year":        year,
                    "driver":      driver,
                    "compound":    compound,
                    "lap_count":   len(group),
                    "avg_lap_time": round(float(lap_times.mean()), 3),
                    "degradation": round(deg, 4),
                })

            log.info("year_loaded", year=year, stints=len(all_stints))

        except Exception as e:
            log.warning("year_load_failed", year=year, grand_prix=grand_prix, error=str(e))
            continue

    if not all_stints:
        log.error("no_historical_data", grand_prix=grand_prix)
        return []

    return _aggregate_by_compound(all_stints)


def _aggregate_by_compound(stints: list[dict]) -> list[StintInfo]:
    """
    Averages stint metrics per compound across all drivers and years.
    Returns one StintInfo per compound, ordered by typical stint sequence.
    """
    df = pd.DataFrame(stints)

    # Typical stint order at most circuits
    compound_order = ["SOFT", "MEDIUM", "HARD"]
    available = [c for c in compound_order if c in df["compound"].values]

    result = []
    for i, compound in enumerate(available):
        subset = df[df["compound"] == compound]

        avg_lap_time = round(float(subset["avg_lap_time"].mean()), 3)
        avg_deg      = round(float(subset["degradation"].mean()), 4)
        avg_laps     = round(float(subset["lap_count"].mean()))
        total_laps   = sum(avg_laps for _ in available)  # rough race length estimate

        # Estimate pit lap — evenly distributed across stints
        start_lap = sum(
            round(float(df[df["compound"] == c]["lap_count"].mean()))
            for c in available[:i]
        ) + 1

        result.append(StintInfo(
            stint_number=i + 1,
            compound=compound,
            start_lap=start_lap,
            end_lap=start_lap + avg_laps - 1,
            lap_count=avg_laps,
            avg_lap_time_sec=avg_lap_time,
            degradation_sec_per_lap=avg_deg,
        ))

    log.info("historical_aggregation_complete", compounds=available, stint_count=len(result))
    return result


# ── Historical weather loader ─────────────────────────────────────────────────
def get_historical_weather(
    grand_prix: str,
    lookback_years: int = 3,
    reference_year: int = 2023,
) -> pd.DataFrame:
    """
    Aggregates weather data across multiple seasons at the same circuit.
    Returns averaged weather snapshots for pre-race risk assessment.
    """
    years = list(range(reference_year - lookback_years, reference_year))
    log.info("historical_weather_loading", grand_prix=grand_prix, years=years)

    all_weather = []

    for year in years:
        try:
            session = fastf1.get_session(year, grand_prix, "R")
            session.load(telemetry=False, weather=True, messages=False)

            weather = session.weather_data[
                ["Time", "AirTemp", "TrackTemp", "Rainfall", "WindSpeed"]
            ].copy()
            weather["TimeMin"] = weather["Time"].dt.total_seconds() / 60
            weather["Year"]    = year
            all_weather.append(weather)

        except Exception as e:
            log.warning("weather_load_failed", year=year, error=str(e))
            continue

    if not all_weather:
        return pd.DataFrame()

    combined = pd.concat(all_weather, ignore_index=True)

    # Bin by time (every 5 minutes) and average across years
    combined["TimeBin"] = (combined["TimeMin"] // 5) * 5
    averaged = combined.groupby("TimeBin").agg(
        AirTemp=("AirTemp",   "mean"),
        TrackTemp=("TrackTemp", "mean"),
        Rainfall=("Rainfall",  "any"),   # True if rain in ANY year at this time
        WindSpeed=("WindSpeed", "mean"),
        TimeMin=("TimeMin",   "mean"),
    ).reset_index(drop=True)

    log.info("historical_weather_complete", snapshots=len(averaged))
    return averaged


# ── Historical pit stop summary ───────────────────────────────────────────────
def get_historical_pit_summary(
    grand_prix: str,
    lookback_years: int = 3,
    reference_year: int = 2023,
) -> pd.DataFrame:
    """
    Aggregates typical pit stop windows across seasons.
    Returns a summary of when drivers typically pit at this circuit.
    """
    years = list(range(reference_year - lookback_years, reference_year))
    log.info("historical_pits_loading", grand_prix=grand_prix, years=years)

    all_pits = []

    for year in years:
        try:
            session = fastf1.get_session(year, grand_prix, "R")
            session.load(telemetry=False, weather=False, messages=False)

            laps = session.laps.copy()
            laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

            pit_laps = laps[laps["PitOutTime"].notna()][
                ["Driver", "LapNumber", "Compound", "TyreLife"]
            ].copy()
            pit_laps["Year"] = year
            all_pits.append(pit_laps)

        except Exception as e:
            log.warning("pits_load_failed", year=year, error=str(e))
            continue

    if not all_pits:
        return pd.DataFrame()

    combined = pd.concat(all_pits, ignore_index=True)
    combined = combined.rename(columns={
        "LapNumber": "PitLap",
        "Compound":  "NewCompound"
    })

    log.info("historical_pits_complete", total_stops=len(combined))
    return combined