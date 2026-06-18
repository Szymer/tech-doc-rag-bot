from __future__ import annotations

from fastapi import FastAPI

from app.api.routes.ask_router import ask_router
from app.api.routes.health_router import health_router
from app.core.logging import setup_logging

setup_logging("INFO")

app = FastAPI(
    title="Control Cabinet RAG API",
    version="0.1.0",
)

app.include_router(ask_router, prefix="/api")
app.include_router(health_router, prefix="/api")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "control-cabinet-rag",
        "status": "running",
    }