from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_splitter(chunk_size: int = 800, chunk_overlap: int = 120) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
