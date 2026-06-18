from __future__ import annotations

import json
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def rerank_with_llm(
    question: str,
    docs: List[Document],
    llm: ChatOpenAI,
    top_n: int = 5,
) -> List[Document]:
    items = []

    for i, doc in enumerate(docs):
        items.append(
            {
                "id": i,
                "source": doc.metadata.get("filename", doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page", doc.metadata.get("page_label", "?")),
                "chunk_id": doc.metadata.get("chunk_id", "?"),
                "text": doc.page_content[:1000],
            }
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a strict reranker for technical documentation.
Select the most relevant passages for answering the original question.
Return JSON only: {{"top_ids": [0, 3, 5]}}.
""",
            ),
            (
                "human",
                """
Question:
{question}

Candidate passages:
{items}
""",
            ),
        ]
    )

    response = (prompt | llm).invoke(
        {
            "question": question,
            "items": json.dumps(items, ensure_ascii=False),
        }
    )

    try:
        data = json.loads(response.content)
        ids = data["top_ids"][:top_n]
        return [docs[i] for i in ids if isinstance(i, int) and 0 <= i < len(docs)]
    except Exception:
        return docs[:top_n]