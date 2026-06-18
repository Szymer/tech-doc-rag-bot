from __future__ import annotations

import concurrent.futures
import uuid
from typing import Any

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from app.models.schemas import AskRequest, AskResponse
from app.rag.rag_services import RAGService
from app.graphs.rag_graph import build_rag_graph
from app.api.deps.auth import require_api_key
from app.api.deps.rate_limit import rate_limit

ask_router = APIRouter()

service = RAGService()
graph = build_rag_graph(service)

GRAPH_TIMEOUT_SEC = 45


@ask_router.post(
    "/ask",
    response_model=AskResponse,
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
def ask(body: AskRequest, request: Request) -> AskResponse | JSONResponse:
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                graph.invoke,
                {
                    "question": body.question,
                    "request_id": request_id,
                },
            )
            result: dict[str, Any] = future.result(timeout=GRAPH_TIMEOUT_SEC)

        return AskResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            request_id=request_id,
        )

    except concurrent.futures.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "detail": "Request timed out",
                "request_id": request_id,
            },
        )

    except Exception:
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal error",
                "request_id": request_id,
            },
        )