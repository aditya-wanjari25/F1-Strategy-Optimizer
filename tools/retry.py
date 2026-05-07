"""
Retry and resilience utilities for the F1 Strategy Optimizer.

Covers:
  - Exponential backoff for transient failures (rate limits, timeouts)
  - JSON repair for malformed LLM responses
  - Safe LLM call wrapper combining both
"""

import re
import json
import time
import structlog
from functools import wraps
from typing import Callable, TypeVar, Any

log = structlog.get_logger()

T = TypeVar("T")


# ── Exponential Backoff Retry ─────────────────────────────────────────────────
def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
):
    """
    Decorator that retries a function with exponential backoff.

    Attempt 1: immediate
    Attempt 2: wait 1s
    Attempt 3: wait 2s
    Attempt 4: wait 4s  (if max_attempts allows)

    Usage:
        @with_retry(max_attempts=3, retryable_exceptions=(RateLimitError,))
        def call_llm(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        log.info("retry_succeeded", func=func.__name__, attempt=attempt)
                    return result

                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        log.error(
                            "retry_exhausted",
                            func=func.__name__,
                            attempts=max_attempts,
                            error=str(e),
                        )
                        raise

                    log.warning(
                        "retry_attempt",
                        func=func.__name__,
                        attempt=attempt,
                        next_delay_sec=delay,
                        error=str(e),
                    )
                    time.sleep(delay)
                    delay *= backoff_factor

            raise last_exception

        return wrapper
    return decorator


# ── JSON Repair ───────────────────────────────────────────────────────────────
def repair_json(raw: str) -> str:
    """
    Attempts to extract valid JSON from a messy LLM response.

    Handles these common failure modes:
      1. Markdown fences:  ```json { ... } ```
      2. Preamble text:    "Sure! Here's the result: { ... }"
      3. Trailing text:    "{ ... } Let me know if you need more."
      4. Single quotes:    {'key': 'value'} instead of {"key": "value"}
    """
    # 1. Strip markdown fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        log.warning("json_repair_stripped_fence")
        return fenced.group(1).strip()

    # 2. Extract first {...} block (handles preamble and trailing text)
    braces = re.search(r"\{.*\}", raw, re.DOTALL)
    if braces:
        candidate = braces.group(0).strip()
        # 3. Replace single quotes with double quotes
        candidate = re.sub(r"(?<![\\])'", '"', candidate)
        log.warning("json_repair_extracted_block")
        return candidate

    # 4. Give up — return as-is and let json.loads raise
    log.error("json_repair_failed", raw=raw[:200])
    return raw


def parse_llm_json(raw: str) -> dict:
    """
    Parses JSON from an LLM response, attempting repair if needed.
    Raises json.JSONDecodeError if repair also fails.
    """
    # First attempt — clean response
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("json_parse_failed_attempting_repair")

    # Second attempt — after repair
    repaired = repair_json(raw)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        log.error("json_repair_failed_giving_up", error=str(e), raw=raw[:200])
        raise


# ── Safe LLM Call ─────────────────────────────────────────────────────────────
def safe_llm_call(
    llm,
    messages: list,
    agent_name: str,
    max_attempts: int = 3,
) -> dict:
    """
    Wraps an LLM call with:
      1. Retry with exponential backoff on transient errors
      2. JSON repair on malformed responses
      3. Structured logging at every step

    Returns parsed dict on success.
    Raises on exhausted retries or unrecoverable JSON failure.
    """
    delay = 1.0

    for attempt in range(1, max_attempts + 1):
        try:
            log.info("llm_call_start", agent=agent_name, attempt=attempt)
            start = time.time()
            response = llm.invoke(messages)
            duration = round(time.time() - start, 3)
            raw = response.content.strip()

            log.info(
                "llm_call_complete",
                agent=agent_name,
                attempt=attempt,
                duration_sec=duration,
                response_chars=len(raw),
            )

            # Parse with repair fallback
            parsed = parse_llm_json(raw)
            return parsed, raw

        except json.JSONDecodeError as e:
            # JSON is unrecoverable — no point retrying
            log.error("llm_json_unrecoverable", agent=agent_name, error=str(e))
            raise

        except Exception as e:
            if attempt == max_attempts:
                log.error(
                    "llm_call_exhausted",
                    agent=agent_name,
                    attempts=max_attempts,
                    error=str(e),
                )
                raise

            log.warning(
                "llm_call_retry",
                agent=agent_name,
                attempt=attempt,
                next_delay_sec=delay,
                error=str(e),
            )
            time.sleep(delay)
            delay *= 2.0

    raise RuntimeError(f"LLM call failed after {max_attempts} attempts")