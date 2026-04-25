from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "rag_docs"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
