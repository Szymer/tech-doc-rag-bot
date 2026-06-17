from fastapi import FastAPI

from app.api.routes.ask_router import router as ask_router
from app.api.routes.health_router import router as health_router
from app.core.logging import configure_logging


configure_logging()

app = FastAPI(title="Control Cabinet RAG")
app.include_router(health_router)
app.include_router(ask_router)
