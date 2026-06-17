def run_qa_chain(question: str, context: str) -> str:
    return f"Q: {question}\nA (mock): {context[:200]}"
