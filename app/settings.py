from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_MODEL: str = "llama3.1"
    TEMPERATURE: float = 0.2
    RAG_ENABLED: bool = True
    VECTOR_DIR: str = "chroma_db"
    SYSTEM_PROMPT: str = (
        "You are a helpful assistant. Use context when available."
    )

settings = Settings()