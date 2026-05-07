# F1 Race Strategy Optimizer 🏎️

A production-grade multi-agent system that analyzes Formula 1 race strategy using real telemetry data. Built with LangGraph, FastF1, and FastAPI.

---

## What It Does

Given a race, year, and driver, the system spins up four specialized AI agents that run in parallel to produce an optimal race strategy recommendation:

- **Tire Agent** — analyzes stint-by-stint degradation data to recommend compounds and pit windows
- **Weather Agent** — assesses track temperature trends and rainfall risk
- **Competitor Agent** — identifies undercut and overcut opportunities vs rivals
- **Synthesizer Agent** — reconciles all three findings into a single justified strategy

```
supervisor
    ├── tire_agent ──────┐
    ├── weather_agent ───┼──→ synthesizer → recommendation
    └── competitor_agent ┘
```

---

## Architecture

```
f1-strategy-optimizer/
├── agents/
│   ├── tire_agent.py         # Tyre degradation and compound analysis
│   ├── weather_agent.py      # Weather risk assessment
│   ├── competitor_agent.py   # Undercut/overcut opportunity detection
│   └── synthesizer.py        # Final strategy synthesis
├── api/
│   ├── app.py                # FastAPI application factory
│   ├── job_store.py          # Thread-safe in-memory job store
│   ├── models.py             # Pydantic request/response models
│   └── router.py             # REST endpoints
├── graph/
│   ├── state.py              # Shared AgentState with typed reducers
│   └── workflow.py           # LangGraph graph definition
├── observability/
│   └── tracing.py            # Langfuse tracing helpers
├── tools/
│   ├── fastf1_tools.py       # FastF1 data layer
│   └── retry.py              # Exponential backoff and JSON repair
├── tests/                    # Unit tests per agent and tool
├── main.py                   # CLI entrypoint
└── server.py                 # Uvicorn server entrypoint
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph 1.1+ |
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| F1 data | FastF1 3.4+ |
| REST API | FastAPI + Uvicorn |
| Observability | Langfuse v4 |
| Resilience | Exponential backoff + JSON repair |
| Logging | structlog |
| Package manager | uv |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/f1-strategy-optimizer.git
cd f1-strategy-optimizer
uv sync
uv sync --extra dev
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Run via CLI

```bash
uv run python main.py --year 2023 --gp Bahrain --driver VER
```

Example output:

```
╔══════════════════════════════════════╗
║   F1 Race Strategy Optimizer         ║
║   Bahrain Grand Prix 2023 — VER      ║
╚══════════════════════════════════════╝

🔧 Tyre Analysis
  Compounds : SOFT → HARD
  Pit laps  : [35]

🌡  Weather Analysis
  Rain      : False
  Temp trend: falling
  Risk      : LOW

🏁 Competitor Analysis
  Undercuts : GAS on lap 10 (gap: 4.0s)

┌─────────────────────────────────────────────────────────────────┐
│ RECOMMENDED STRATEGY                    Confidence: HIGH        │
│                                                                  │
│ Start on SOFT, pit lap 35 for HARD. Weather poses no risk.      │
│ Undercut vs GAS not viable given 4s gap. High confidence.       │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Run via REST API

Start the server:

```bash
uv run python server.py
```

Submit a strategy job:

```bash
curl -X POST http://localhost:8000/strategy \
  -H "Content-Type: application/json" \
  -d '{"year": 2023, "grand_prix": "Bahrain", "driver": "VER"}'

# Returns immediately:
# {"job_id": "abc-123", "status": "pending", ...}
```

Poll for result:

```bash
curl http://localhost:8000/strategy/abc-123
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Key Engineering Concepts

### Multi-Agent Pattern
The system uses a **supervisor fan-out** pattern — a lightweight supervisor kicks off three specialist agents in parallel, waits for all to complete, then routes to a synthesizer. Each agent has a single responsibility and writes to its own isolated section of shared state.

### Parallel Execution
The three specialist agents run **concurrently** via LangGraph's fan-out edges. Total latency is `max(tire, weather, competitor)` rather than their sum — roughly 60% faster than sequential execution.

### Resilience
Every LLM call is wrapped with:
- **Exponential backoff retry** — handles rate limits and transient failures
- **JSON repair** — handles markdown fences, preamble text, and trailing noise from LLM responses

### Observability
Every graph run produces a **Langfuse trace** with nested spans per agent and generations per LLM call. You get token counts, cost, and latency breakdown per agent out of the box.

### Async Job Pattern
The REST API returns a `job_id` immediately (HTTP 202 Accepted) and runs the graph in a background thread. Clients poll `GET /strategy/{job_id}` for results. This prevents HTTP timeouts on long-running agent graphs.

---

## Running Tests

```bash
uv run pytest tests/ -v
```

---

## Observability

Traces are sent to [Langfuse](https://langfuse.com). Each run produces:

```
f1_strategy_optimizer
  ├── tire_agent
  │     └── tire_agent_llm
  ├── weather_agent
  │     └── weather_agent_llm
  ├── competitor_agent
  │     └── competitor_agent_llm
  └── synthesizer
        └── synthesizer_llm
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse host (default: https://cloud.langfuse.com) |

---
