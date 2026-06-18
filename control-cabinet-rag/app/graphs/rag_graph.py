from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, NotRequired, TypedDict
import logging

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

from app.rag.rag_services import RAGService, doc_key
from app.rag.chains.rewrite_chain import rewrite_question
from app.rag.chains.rerank_chain import rerank_with_llm
from app.rag.chains.qa_chain import answer_question

logger = logging.getLogger("graph.rag")

MAX_RERANK_RETRIES = 3


class RAGState(TypedDict):
    question: str
    request_id: NotRequired[str]

    rewrites: NotRequired[Dict[str, str]]
    candidates: NotRequired[List[Document]]
    deduped: NotRequired[List[Document]]
    top_docs: NotRequired[List[Document]]

    answer: NotRequired[str]
    sources: NotRequired[List[Dict[str, Any]]]

    rerank_attempts: NotRequired[int]
    rerank_error: NotRequired[str]


def latency_ms(t0: float) -> int:
    return int((perf_counter() - t0) * 1000)


def build_rag_graph(rag_service: RAGService) -> Any:
    graph = StateGraph(RAGState)

    def rewrite_node(state: RAGState) -> Dict[str, Any]:
        t0 = perf_counter()
        q = state["question"]
        rid = state.get("request_id", "no-rid")

        rewrites = {
            "bm25": rewrite_question(q, rag_service.llm_rewrite, "BM25 keyword query"),
            "semantic": rewrite_question(q, rag_service.llm_rewrite, "semantic query"),
            "section": rewrite_question(q, rag_service.llm_rewrite, "technical manual section query"),
        }

        logger.info(
            "rewrite_done",
            extra={"request_id": rid, "latency_ms": latency_ms(t0)},
        )

        return {
            "rewrites": rewrites,
            "rerank_attempts": 0,
            "rerank_error": "",
        }

    def retrieve_node(state: RAGState) -> Dict[str, Any]:
        t0 = perf_counter()
        rid = state.get("request_id", "no-rid")

        ensemble = rag_service.build_ensemble()
        rw = state["rewrites"]

        candidates = (
            ensemble.invoke(rw["bm25"])
            + ensemble.invoke(rw["semantic"])
            + ensemble.invoke(rw["section"])
        )

        logger.info(
            "retrieve_done",
            extra={
                "request_id": rid,
                "latency_ms": latency_ms(t0),
                "candidates": len(candidates),
            },
        )

        return {"candidates": candidates}

    def dedupe_node(state: RAGState) -> Dict[str, Any]:
        t0 = perf_counter()
        rid = state.get("request_id", "no-rid")

        deduped_map = {doc_key(doc): doc for doc in state["candidates"]}
        deduped = list(deduped_map.values())[:40]

        logger.info(
            "dedupe_done",
            extra={
                "request_id": rid,
                "latency_ms": latency_ms(t0),
                "deduped": len(deduped),
            },
        )

        return {"deduped": deduped}

    def route_after_dedupe(state: RAGState) -> str:
        if not state.get("deduped"):
            return "no_answer"
        return "rerank"

    def rerank_node(state: RAGState) -> Dict[str, Any]:
        t0 = perf_counter()
        rid = state.get("request_id", "no-rid")
        attempts = state.get("rerank_attempts", 0)

        try:
            top_docs = rerank_with_llm(
                question=state["question"],
                docs=state["deduped"],
                llm=rag_service.llm_rerank,
                top_n=5,
            )

            logger.info(
                "rerank_done",
                extra={
                    "request_id": rid,
                    "latency_ms": latency_ms(t0),
                    "top_docs": len(top_docs),
                    "rerank_attempts": attempts,
                },
            )

            return {
                "top_docs": top_docs,
                "rerank_error": "",
                "rerank_attempts": attempts,
            }

        except Exception as exc:
            logger.warning(
                "rerank_failed",
                extra={
                    "request_id": rid,
                    "latency_ms": latency_ms(t0),
                    "top_docs": 0,
                    "rerank_attempts": attempts + 1,
                    "error": repr(exc),
                },
            )

            return {
                "top_docs": [],
                "rerank_error": repr(exc),
                "rerank_attempts": attempts + 1,
            }

    def route_after_rerank(state: RAGState) -> str:
        if state.get("top_docs"):
            return "answer"

        if state.get("rerank_attempts", 0) >= MAX_RERANK_RETRIES:
            return "no_answer"

        return "rerank"

    def answer_node(state: RAGState) -> Dict[str, Any]:
        t0 = perf_counter()
        rid = state.get("request_id", "no-rid")

        response = answer_question(
            question=state["question"],
            docs=state["top_docs"],
            llm=rag_service.llm,
        )

        logger.info(
            "answer_done",
            extra={"request_id": rid, "latency_ms": latency_ms(t0)},
        )

        return {
            "answer": response["answer"],
            "sources": response["sources"],
        }

    def no_answer_node(state: RAGState) -> Dict[str, Any]:
        rid = state.get("request_id", "no-rid")

        logger.info(
            "no_answer_done",
            extra={"request_id": rid},
        )

        return {
            "answer": "I don't know based on the provided documents.",
            "sources": [],
            "top_docs": [],
        }

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("dedupe", dedupe_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("answer", answer_node)
    graph.add_node("no_answer", no_answer_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "dedupe")

    graph.add_conditional_edges(
        "dedupe",
        route_after_dedupe,
        {
            "rerank": "rerank",
            "no_answer": "no_answer",
        },
    )

    graph.add_conditional_edges(
        "rerank",
        route_after_rerank,
        {
            "answer": "answer",
            "rerank": "rerank",
            "no_answer": "no_answer",
        },
    )

    graph.add_edge("answer", END)
    graph.add_edge("no_answer", END)

    return graph.compile()