from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.core.config import settings


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=temperature,
        timeout=20,
        max_retries=2,
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embeddings_model)