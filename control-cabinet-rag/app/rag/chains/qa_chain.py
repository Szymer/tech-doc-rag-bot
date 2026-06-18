from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.models.schemas import QAResult


def format_docs(docs: List[Document]) -> str:
    blocks = []

    for doc in docs:
        source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
        chunk_id = doc.metadata.get("chunk_id", "?")

        blocks.append(
            f"[source={source} page={page} chunk={chunk_id}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(blocks)


def extract_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    sources = []

    for doc in docs:
        sources.append(
            {
                "source": doc.metadata.get("filename", doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page_label", doc.metadata.get("page", None)),
                "chunk_id": doc.metadata.get("chunk_id", None),
            }
        )

    return sources


def answer_question(
    question: str,
    docs: List[Document],
    llm: ChatOpenAI,
) -> Dict[str, Any]:
    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You answer questions using ONLY the provided technical documentation context.

Rules:
- If the answer is not clearly supported by the context, set can_answer=false.
- Do not guess.
- Do not use outside knowledge.
- For safety-related electrical work, mention that procedures must follow local safety rules and qualified personnel requirements if the context supports it.
- Keep the answer practical and precise.
""",
            ),
            (
                "human",
                """
Question:
{question}

Context:
{context}
""",
            ),
        ]
    )

    structured_llm = llm.with_structured_output(QAResult, method="function_calling")
    result: QAResult = (prompt | structured_llm).invoke(
        {"question": question, "context": context}
    )

    if not result.can_answer:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": [],
        }

    return {
        "answer": result.answer,
        "sources": extract_sources(docs),
    }