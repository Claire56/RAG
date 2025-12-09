"""Configuration management for the RAG system."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Config(BaseSettings):
    """Application configuration."""
    
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
    # Redis
    redis_host: Optional[str] = os.getenv("REDIS_HOST")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    
    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")
    
    # Vector DB
    vector_db: str = os.getenv("VECTOR_DB", "chromadb")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    
    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "fixed_size")
    
    # RAG
    top_k_retrieve: int = int(os.getenv("TOP_K_RETRIEVE", "10"))
    top_k_rerank: int = int(os.getenv("TOP_K_RERANK", "5"))
    use_reranking: bool = os.getenv("USE_RERANKING", "true").lower() == "true"
    
    # Cache
    cache_ttl_embeddings: int = int(os.getenv("CACHE_TTL_EMBEDDINGS", "86400"))
    cache_ttl_responses: int = int(os.getenv("CACHE_TTL_RESPONSES", "3600"))
    cache_ttl_retrieval: int = int(os.getenv("CACHE_TTL_RETRIEVAL", "3600"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "./logs/rag_system.log")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


config = Config()
