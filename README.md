# F1 Race Strategy Optimizer 🏎️

A production-grade multi-agent system that analyzes Formula 1 race strategy using real telemetry data. Built with LangGraph, FastF1, and FastAPI.

---

## What It Does

Given a race, year, and driver, the system spins up four specialized AI agents that run in parallel to produce an optimal race strategy recommendation:

- **Tire Agent** — analyzes stint-by-stint degradation data to recommend compounds and pit windows
- **Weather Agent** — assesses track temperature trends and rainfall risk
- **Competitor Agent** — identifies undercut and overcut opportunities vs rivals
- **Synthesizer Agent** — reconciles all three findings into a single justified strategy


<img width="634" height="677" alt="image" src="https://github.com/user-attachments/assets/2e89f5fd-03e7-4fb6-b758-151d0328348f" />



---


## Key Engineering Concepts

### Multi-Agent Pattern
The system uses a **supervisor fan-out** pattern — a lightweight supervisor kicks off three specialist agents in parallel, waits for all to complete, then routes to a synthesizer. Each agent has a single responsibility and writes to its own isolated section of shared state.

### Parallel Execution
The three specialist agents run **concurrently** via LangGraph's fan-out edges. Total latency is `max(tire, weather, competitor)` rather than their sum — roughly 60% faster than sequential execution. Confirmed via overlapping spans in Langfuse.

### Resilience
Every LLM call is wrapped with:
- **Exponential backoff retry** — handles rate limits and transient failures (3 attempts, 1s → 2s → 4s delay)
- **JSON repair** — handles markdown fences, preamble text, trailing noise, and single quotes in LLM responses

### Observability
Every graph run produces a **Langfuse trace** with nested spans per agent and generations per LLM call. Token counts, cost, and latency are tracked per agent out of the box.


### Eval Framework
A rigorous eval system scores recommendations against manually curated ground truth:

- **Compound accuracy** — % of compounds correctly predicted per stint
- **Pit lap delta** — average absolute difference in pit lap numbers
- **Stint count match** — correct number of stops predicted
- **Weighted composite** — safety car and rain cases adjust weights automatically
- **DNF/DNS handling** — DNF cases excluded from position scoring, DNS excluded entirely

The eval framework drove a real data layer fix: detecting same-compound multi-stop strategies via TyreLife drops improved overall score from **0.63 → 0.88**.

### Async Job Pattern
The REST API returns a `job_id` immediately (HTTP 202 Accepted) and runs the graph in a background thread. Clients poll `GET /strategy/{job_id}` for results. This prevents HTTP timeouts on long-running agent graphs.

### Docker Deployment
The FastAPI server runs in a `python:3.11-slim` container. Dependency installation is layer-cached — rebuilds after code changes take seconds. FastF1 cache is persisted via a named Docker volume so race data survives container restarts.


---


## Eval Results

Tested against 10 manually curated cases across 5 circuits from the 2023 season:

| Metric | Score |
|---|---|
| Overall score | **0.88 / 1.00** |
| Compound accuracy | **90%** |
| Avg pit lap delta | **1.6 laps** |
| Stint count match | **9 / 10** |

**Breakdown by circuit type:**

| Circuit Type | Cases | Avg Score |
|---|---|---|
| High degradation (Bahrain, Abu Dhabi) | 4 | 0.97 |
| Mixed (British) | 2 | 0.88 |
| Low degradation (Monaco, Monza) | 4 | 0.79 |

Known limitation: rain-affected races with late safety car interventions score lower — the system has no real-time race state awareness and cannot predict strategy errors by teams.

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
| Evals | Custom scorer with ground truth dataset |
| Logging | structlog |
| Containerization | Docker + Docker Compose |
| Package manager | uv |

---

## Quickstart

### Option A — Docker (recommended)

```bash
git clone https://github.com/YOUR_USERNAME/f1-strategy-optimizer.git
cd f1-strategy-optimizer

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Build and run
docker compose up
```

The API will be available at `http://localhost:8000`.

FastF1 race data is cached in a Docker volume — first run downloads data, subsequent runs use the cache.

### Option B — Local

```bash
git clone https://github.com/YOUR_USERNAME/f1-strategy-optimizer.git
cd f1-strategy-optimizer

# Install dependencies
uv sync
uv sync --extra dev

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

---

## Usage

### CLI

```bash
uv run python main.py --year 2023 --gp Bahrain --driver VER
uv run python main.py --year 2023 --gp British --driver LEC
uv run python main.py --year 2023 --gp Italian --driver SAI
```

Example output:

```
╔══════════════════════════════════════╗
║   F1 Race Strategy Optimizer         ║
║   Bahrain Grand Prix 2023 — VER      ║
╚══════════════════════════════════════╝

🔧 Tyre Analysis
  Compounds : SOFT → SOFT → HARD
  Pit laps  : [14, 36]

🌡  Weather Analysis
  Rain      : False
  Temp trend: falling
  Risk      : LOW

🏁 Competitor Analysis
  Undercuts : HAM on lap 13 (gap: 3.2s)

┌─────────────────────────────────────────────────────────────────┐
│ RECOMMENDED STRATEGY                    Confidence: HIGH        │
│                                                                  │
│ Two-stop SOFT-SOFT-HARD strategy. Pit lap 14 and 36.            │
│ Weather poses no risk. Undercut vs HAM not viable. High conf.   │
└─────────────────────────────────────────────────────────────────┘
```

### REST API

Start the server:

```bash
uv run python server.py
```

**Submit a strategy job:**

```bash
curl -X POST http://localhost:8000/strategy \
  -H "Content-Type: application/json" \
  -d '{"year": 2023, "grand_prix": "Bahrain", "driver": "VER"}'

# Returns immediately with job_id:
# {"job_id": "abc-123", "status": "pending", ...}
```

**Poll for result:**

```bash
curl http://localhost:8000/strategy/abc-123
# status: pending → running → complete
```

**Health check:**

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "0.1.0"}
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Running Evals

```bash
# Run all 10 cases
uv run python -m evals.run_evals

# Run a single case
uv run python -m evals.run_evals --year 2023 --gp Italian --driver VER

# Run by circuit type
uv run python -m evals.run_evals --circuit high_deg
uv run python -m evals.run_evals --circuit low_deg
uv run python -m evals.run_evals --circuit mixed
```


### Running Tests

```bash
uv run pytest tests/ -v
```

Test coverage:
- FastF1 data layer (5 tools)
- Tire Agent (data formatting + LLM output)
- Weather Agent (data formatting + LLM output)
- Competitor Agent (data formatting + LLM output)
- Synthesizer Agent (multi-signal reconciliation)
- Retry and JSON repair utilities (6 cases)
- Eval scorer (7 cases including DNF, SC, rain handling)
- Full graph compilation and end-to-end run

