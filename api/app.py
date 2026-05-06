"""
FastAPI application factory.
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from api.router import router

def create_app() -> FastAPI:
    app = FastAPI(
        title="F1 Race Strategy Optimizer",
        description="Multi-agent F1 strategy analysis powered by LangGraph and FastF1",
        version="0.1.0",
    )
    app.include_router(router)
    return app

app = create_app()