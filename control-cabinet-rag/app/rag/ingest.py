from pathlib import Path


def ingest_documents(data_dir: str = "data/raw") -> list[Path]:
    path = Path(data_dir)
    return list(path.glob("*.pdf"))
