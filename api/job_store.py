"""
In-memory job store for the F1 Strategy Optimizer API.

Tracks the lifecycle of every strategy request:
  pending → running → complete | failed

In production you'd replace this with Redis or a database.
For now, a thread-safe in-memory dict is perfect for learning
the async job pattern without infrastructure overhead.
"""

import uuid
import structlog
from datetime import datetime, timezone
from threading import Lock
from api.models import JobResponse, JobStatus, StrategyRequest

log = structlog.get_logger()

# ── Job Store ─────────────────────────────────────────────────────────────────
class JobStore:
    """
    Thread-safe in-memory store for strategy jobs.

    Why thread-safe? FastAPI runs request handlers concurrently.
    Without a Lock, two requests writing to _jobs simultaneously
    could corrupt the dict. The Lock ensures only one thread
    writes at a time.
    """

    def __init__(self):
        self._jobs: dict[str, JobResponse] = {}
        self._lock = Lock()

    def create(self, request: StrategyRequest) -> JobResponse:
        """Creates a new job in PENDING state and returns it."""
        job_id = str(uuid.uuid4())
        job = JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            request=request,
        )
        with self._lock:
            self._jobs[job_id] = job
        log.info("job_created", job_id=job_id, driver=request.driver, grand_prix=request.grand_prix)
        return job

    def get(self, job_id: str) -> JobResponse | None:
        """Returns a job by ID, or None if not found."""
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs) -> JobResponse | None:
        """
        Updates specific fields on a job.
        Used to transition status and attach results.

        Example:
            store.update(job_id, status=JobStatus.RUNNING)
            store.update(job_id, status=JobStatus.COMPLETE, strategy_recommendation=sr)
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                log.warning("job_not_found", job_id=job_id)
                return None

            # Pydantic models are immutable — rebuild with updated fields
            updated = job.model_copy(update=kwargs)
            self._jobs[job_id] = updated
            log.info("job_updated", job_id=job_id, updates=list(kwargs.keys()))
            return updated

    def list_all(self) -> list[JobResponse]:
        """Returns all jobs — useful for debugging."""
        with self._lock:
            return list(self._jobs.values())


# ── Singleton instance ────────────────────────────────────────────────────────
# One store shared across the entire application lifetime
job_store = JobStore()