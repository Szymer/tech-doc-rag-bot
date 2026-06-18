from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from app.core.config import settings
from app.rag.models import get_embeddings, get_llm
from app.rag.vectorstore import load_chroma
from app.rag.retrievers import (
    load_chunks_jsonl,
    build_bm25_retriever,
    build_ensemble_retriever,
)


def doc_key(doc: Document) -> tuple:
    return (
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.metadata.get("chunk_id"),
    )


class RAGService:
    def __init__(self) -> None:
        self.settings = settings

        self.llm = get_llm(temperature=0.2)
        self.llm_rewrite = get_llm(temperature=0.0)
        self.llm_rerank = get_llm(temperature=0.0)

        self.embeddings = get_embeddings()
        self.vectorstore = load_chroma(self.embeddings, settings.chroma_path)
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})

        self.chunks = load_chunks_jsonl(settings.chunks_path)
        self.bm25 = build_bm25_retriever(self.chunks, k=6)

    def build_ensemble(self):
        return build_ensemble_retriever(
            bm25=self.bm25,
            vector=self.vector_retriever,
        )