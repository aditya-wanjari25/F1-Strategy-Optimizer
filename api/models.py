"""
Pydantic models for the F1 Strategy Optimizer API.

These define the exact shape of every request and response.
FastAPI uses these for:
  - Automatic request validation
  - Auto-generated OpenAPI docs
  - Type-safe response serialization
"""

from enum import Enum
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────
class JobStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    COMPLETE = "complete"
    FAILED   = "failed"


class Confidence(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


# ── Request ───────────────────────────────────────────────────────────────────
class StrategyRequest(BaseModel):
    year:        int = Field(..., ge=2018, le=2025, description="Race year")
    grand_prix:  str = Field(..., min_length=3,     description="Grand Prix name e.g. Bahrain")
    driver:      str = Field(..., min_length=2, max_length=3, description="Driver code e.g. VER")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"year": 2023, "grand_prix": "Bahrain", "driver": "VER"}
            ]
        }
    }


# ── Inner response models ─────────────────────────────────────────────────────
class TireAnalysisResponse(BaseModel):
    recommended_compounds: list[str]
    optimal_pit_laps:      list[int]
    summary:               str


class WeatherAnalysisResponse(BaseModel):
    has_rain:         bool
    track_temp_trend: str
    risk_level:       str
    summary:          str


class CompetitorAnalysisResponse(BaseModel):
    undercut_opportunities: list[dict]
    overcut_opportunities:  list[dict]
    summary:                str


class StrategyResultResponse(BaseModel):
    compounds:  list[str]
    pit_laps:   list[int]
    confidence: Confidence
    rationale:  str


# ── Job response ──────────────────────────────────────────────────────────────
class JobResponse(BaseModel):
    job_id:     str
    status:     JobStatus
    request:    StrategyRequest

    # Populated when status == COMPLETE
    tire_analysis:           TireAnalysisResponse | None = None
    weather_analysis:        WeatherAnalysisResponse | None = None
    competitor_analysis:     CompetitorAnalysisResponse | None = None
    strategy_recommendation: StrategyResultResponse | None = None

    # Populated when status == FAILED
    errors: list[str] = []


# ── Health check ──────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status:  str
    version: str