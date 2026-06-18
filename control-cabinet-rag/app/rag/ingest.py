from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document



def ingest_documents(data_dir: str = "data/raw") -> list[Path]:
    path = Path(data_dir)
    return list(path.glob("*.pdf"))



def load_documents(file_paths:str) -> list[Document]:
    loader = DirectoryLoader(file_paths,  glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    
    docs = loader.load()
    
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        doc.metadata["doc_type"] = "DTR"
        doc.metadata["filename"] = Path(source).name
    
    return docs


loaded_docs = load_documents("/workspaces/tech-doc-rag-bot/control-cabinet-rag/data/raw")