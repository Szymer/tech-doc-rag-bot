from langchain_chroma import Chroma


def get_vectorstore(collection_name: str = "control-cabinet") -> Chroma:
    return Chroma(collection_name=collection_name)
