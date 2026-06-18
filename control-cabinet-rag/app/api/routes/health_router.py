from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import settings

health_router = APIRouter()


@health_router.get("/health")
def health() -> JSONResponse:
    chroma_dir = Path(settings.chroma_dir)
    chunks_path = Path(settings.chunks_path)

    checks = {
        "chroma_dir_exists": chroma_dir.exists(),
        "chunks_path_exists": chunks_path.exists(),
    }

    ok = all(checks.values())

    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "status": "ok" if ok else "degraded",
            "checks": checks,
        },
    )