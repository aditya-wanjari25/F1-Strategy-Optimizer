"""
FastAPI router for the F1 Strategy Optimizer.

Endpoints:
  GET  /health                — liveness check
  POST /strategy              — submit a strategy analysis job
  GET  /strategy/{job_id}     — poll for job result
"""

from dotenv import load_dotenv
load_dotenv()

import structlog
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.models import (
    JobResponse,
    JobStatus,
    StrategyRequest,
    HealthResponse,
    TireAnalysisResponse,
    WeatherAnalysisResponse,
    CompetitorAnalysisResponse,
    StrategyResultResponse,
)
from api.job_store import job_store
from graph.workflow import run_graph

log = structlog.get_logger()

router = APIRouter()

# One thread pool shared across all requests
# max_workers=4 means up to 4 strategy jobs can run concurrently
executor = ThreadPoolExecutor(max_workers=4)


# ── Helper — run graph and update job store ───────────────────────────────────
def _run_strategy_job(job_id: str, request: StrategyRequest):
    """
    Runs the full agent graph and writes results back to the job store.
    This runs in a background thread — never in the request handler itself.
    """
    log.info("job_running", job_id=job_id)
    job_store.update(job_id, status=JobStatus.RUNNING)

    try:
        result = run_graph(request.year, request.grand_prix, request.driver)
        errors = result.get("errors", [])

        if errors:
            job_store.update(job_id, status=JobStatus.FAILED, errors=errors)
            log.error("job_failed", job_id=job_id, errors=errors)
            return

        # Map internal dataclasses → API response models
        tire = result["tire_analysis"]
        weather = result["weather_analysis"]
        competitor = result["competitor_analysis"]
        sr = result["strategy_recommendation"]

        job_store.update(
            job_id,
            status=JobStatus.COMPLETE,
            tire_analysis=TireAnalysisResponse(
                recommended_compounds=tire.recommended_compounds,
                optimal_pit_laps=tire.optimal_pit_laps,
                summary=tire.summary,
            ),
            weather_analysis=WeatherAnalysisResponse(
                has_rain=weather.has_rain,
                track_temp_trend=weather.track_temp_trend,
                risk_level=weather.risk_level,
                summary=weather.summary,
            ),
            competitor_analysis=CompetitorAnalysisResponse(
                undercut_opportunities=competitor.undercut_opportunities,
                overcut_opportunities=competitor.overcut_opportunities,
                summary=competitor.summary,
            ),
            strategy_recommendation=StrategyResultResponse(
                compounds=sr.compounds,
                pit_laps=sr.pit_laps,
                confidence=sr.confidence,
                rationale=sr.rationale,
            ),
        )
        log.info("job_complete", job_id=job_id)

    except Exception as e:
        log.error("job_exception", job_id=job_id, error=str(e))
        job_store.update(job_id, status=JobStatus.FAILED, errors=[str(e)])


# ── Routes ────────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
def health():
    """Liveness check — returns 200 if the server is up."""
    return HealthResponse(status="ok", version="0.1.0")


@router.post("/strategy", response_model=JobResponse, status_code=202)
def submit_strategy(request: StrategyRequest, background_tasks: BackgroundTasks):
    """
    Submit a strategy analysis job.
    Returns immediately with a job_id — use GET /strategy/{job_id} to poll.

    202 Accepted means: request received, processing started, check back later.
    """
    job = job_store.create(request)
    background_tasks.add_task(_run_strategy_job, job.job_id, request)
    log.info("job_submitted", job_id=job.job_id)
    return job


@router.get("/strategy/{job_id}", response_model=JobResponse)
def get_strategy(job_id: str):
    """
    Poll for a strategy job result.

    Status lifecycle:
      pending  → job created, not started yet
      running  → agents are executing
      complete → results ready
      failed   → something went wrong, check errors field
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job