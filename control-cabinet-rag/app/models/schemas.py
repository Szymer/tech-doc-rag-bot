from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    source: str
    page: str | int | None = None
    chunk_id: int | str | None = None


class QAResult(BaseModel):
    can_answer: bool = Field(description="Whether the answer is supported by the context.")
    answer: str = Field(description="Grounded answer.")
    

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]



class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    request_id: str