from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings



def build_chroma(
    chunks: List[Document],
    embeddings: OpenAIEmbeddings,
    persist_directory: str,
) -> Chroma:
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )


def load_chroma( embeddings: OpenAIEmbeddings, persist_directory: str,) -> Chroma:
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )