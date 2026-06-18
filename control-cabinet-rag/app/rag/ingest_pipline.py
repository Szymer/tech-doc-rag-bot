from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.core.config import settings
from app.rag.ingest import load_documents
from app.rag.splitters import split_documents
from app.rag.vectorstore import build_chroma
from app.rag.models import get_embeddings


def save_chunks_jsonl(chunks: List[Document], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        for doc in chunks:
            f.write(
                json.dumps(
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    docs = load_documents(settings.docs_dir)
    chunks = split_documents(docs)

    save_chunks_jsonl(chunks, settings.chunks_path)

    embeddings = get_embeddings()
    build_chroma(chunks, embeddings, settings.chroma_path)

    print(f"Indexed {len(chunks)} chunks")


if __name__ == "__main__":
    main()