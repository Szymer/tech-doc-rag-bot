from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.vectorstores import VectorStoreRetriever



from app.rag.vectorstore import load_chroma
from app.rag.models import get_embeddings
from app.core.config import settings

embeddings = get_embeddings()
vectorstore = load_chroma(embeddings, settings.chroma_path)

vector = vectorstore.as_retriever(
    search_kwargs={"k": 6}
)


retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 24,
        "lambda_mult": 0.7,
    },
)


def load_chunks_jsonl(path: str) -> List[Document]:
    docs: List[Document] = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(
                Document(
                    page_content=obj["page_content"],
                    metadata=obj["metadata"],
                )
            )

    return docs


def build_bm25_retriever(chunks: List[Document], k: int = 6) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


def build_ensemble_retriever(
    bm25: BM25Retriever,
    vector: VectorStoreRetriever,
) -> EnsembleRetriever:
    return EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[0.45, 0.55],
    )