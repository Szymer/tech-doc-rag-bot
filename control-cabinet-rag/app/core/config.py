import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class Settings(BaseModel):
    app_name: str = Field(default="control-cabinet-rag")
    environment: str = Field(default=os.getenv("ENVIRONMENT", "dev"))
    openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    chroma_path: str = Field(default=os.getenv("CHROMA_PATH", "./chroma_db"))


settings = Settings()
