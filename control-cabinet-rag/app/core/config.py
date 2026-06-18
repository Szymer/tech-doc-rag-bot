import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class Settings(BaseModel):
    app_name: str = Field(default="control-cabinet-rag")
    environment: str = Field(default=os.getenv("ENVIRONMENT", "dev"))
    openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    chroma_path: str = Field(default=os.getenv("CHROMA_PATH", "./chroma_db"))
    docs_dir: str = Field(default=os.getenv("DOCS_DIR", "./data/raw"))
    chunks_path: str = Field(default=os.getenv("CHUNKS_PATH", "./data/chunks.jsonl"))
    embeddings_model: str = Field(default=os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"))
    openai_model: str = Field(default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    api_key: str = Field(default=os.getenv("APP_API_KEY", ""))


settings = Settings()
