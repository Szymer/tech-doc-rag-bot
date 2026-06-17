from fastapi import APIRouter

from app.models.schemas import AskRequest, AskResponse

router = APIRouter(prefix="/api", tags=["ask"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest) -> AskResponse:
    # Placeholder endpoint - connect to RAG graph in next step.
    return AskResponse(answer=f"Pytanie odebrane: {payload.question}")
