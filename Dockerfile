# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# gcc needed for some Python packages that compile C extensions
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv ────────────────────────────────────────────────────────────────
RUN curl -Lsf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── Copy dependency files first ───────────────────────────────────────────────
# We copy pyproject.toml before the rest of the code so Docker can
# cache the dependency installation layer. If only code changes,
# Docker reuses the cached deps layer — much faster rebuilds.
COPY pyproject.toml .

# ── Install dependencies ──────────────────────────────────────────────────────
RUN uv sync --no-dev

# ── Copy application code ─────────────────────────────────────────────────────
COPY agents/      ./agents/
COPY api/         ./api/
COPY graph/       ./graph/
COPY observability/ ./observability/
COPY tools/       ./tools/
COPY server.py    .

# ── FastF1 cache directory ────────────────────────────────────────────────────
# This directory will be mounted as a Docker volume at runtime
# so cache persists between container restarts
RUN mkdir -p /app/fastf1_cache

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
# Docker will call this every 30s to verify the container is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]