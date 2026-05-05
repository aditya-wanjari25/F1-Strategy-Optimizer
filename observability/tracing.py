"""
Observability layer for the F1 Strategy Optimizer.
Uses Langfuse v4 start_as_current_observation API.

Hierarchy:
  Trace  (one full graph run)
    └── Agent observation  (one agent)
          └── Generation   (one LLM call)
"""

from dotenv import load_dotenv
load_dotenv()

import structlog
from langfuse import Langfuse

log = structlog.get_logger()

# ── Singleton client ──────────────────────────────────────────────────────────
_langfuse: Langfuse | None = None

def get_langfuse() -> Langfuse:
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse()
        log.info("langfuse_initialized")
    return _langfuse


def flush():
    """Flush all pending traces — call at end of graph run."""
    get_langfuse().flush()


def agent_observation(agent_name: str, input_data: dict):
    """
    Context manager that wraps an agent call as a Langfuse observation.
    
    Usage:
        with agent_observation("tire_agent", {"driver": "VER"}) as obs:
            result = run_tire_agent(state)
            obs.update(output=result)
    """
    lf = get_langfuse()
    return lf.start_as_current_observation(
        name=agent_name,
        as_type="agent",
        input=input_data,
    )


def generation_observation(agent_name: str, model: str, prompt: str):
    """
    Context manager that wraps an LLM call as a Langfuse generation.
    Nested inside an agent observation.
    
    Usage:
        with generation_observation("tire_agent", "gpt-4o-mini", prompt) as gen:
            response = llm.invoke(...)
            gen.update(output=response.content)
    """
    lf = get_langfuse()
    return lf.start_as_current_observation(
        name=f"{agent_name}_llm",
        as_type="generation",
        model=model,
        input=prompt,
    )