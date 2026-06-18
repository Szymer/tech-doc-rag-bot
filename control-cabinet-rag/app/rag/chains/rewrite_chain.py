from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def rewrite_question(
    question: str,
    llm: ChatOpenAI,
    style: str,
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You rewrite user questions for retrieval over technical documentation
for control cabinets, PLCs, drives, power supplies, safety relays and wiring.

Return only the rewritten query.
Do not answer the question.
""",
            ),
            (
                "human",
                """
Original question:
{question}

Rewrite style:
{style}
""",
            ),
        ]
    )

    chain = prompt | llm
    result = chain.invoke({"question": question, "style": style})
    return result.content.strip()