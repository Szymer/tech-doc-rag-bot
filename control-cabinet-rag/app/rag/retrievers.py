from langchain_core.retrievers import BaseRetriever


def build_retriever(vectorstore: BaseRetriever, k: int = 5) -> BaseRetriever:
    return vectorstore.as_retriever(search_kwargs={"k": k})
