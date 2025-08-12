"""
Configuration module for the Teradata SQL Agent project.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

# Obtener la ruta absoluta del directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Azure OpenAI Configuration
    azure_openai_api_key: str = Field(default="", env="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(
        default="https://bci-sub-genai-poc-comunidades-oai.openai.azure.com", 
        env="AZURE_OPENAI_ENDPOINT"
    )
    azure_openai_deployment_name: str = Field(
        default="gpt-4o", 
        env="AZURE_OPENAI_DEPLOYMENT_NAME"
    )
    azure_openai_api_version: str = Field(
        default="2025-01-01-preview", 
        env="AZURE_OPENAI_API_VERSION"
    )
    
    # OpenAI Configuration (legacy - mantenido por compatibilidad)
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    
    # Teradata Database Configuration
    teradata_host: str = Field(default="localhost", env="TERADATA_HOST")
    teradata_user: str = Field(default="dbc", env="TERADATA_USER")
    teradata_password: str = Field(default="dbc", env="TERADATA_PASSWORD")
    teradata_database: str = Field(default="DBC", env="TERADATA_DATABASE")
    
    # Teradata MCP Server Configuration (siguiendo estándares oficiales)
    database_uri: str = Field(
        default="", 
        env="DATABASE_URI"
    )
    teradata_mcp_server_url: str = Field(
        default="http://localhost:3000", 
        env="TERADATA_MCP_SERVER_URL"
    )
    mcp_server_token: Optional[str] = Field(default=None, env="MCP_SERVER_TOKEN")
    
    # Configuración MCP Transport
    mcp_transport_type: str = Field(
        default="http", 
        env="MCP_TRANSPORT_TYPE"  # "stdio", "http", o "sse"
    )
    
    # Vector Database Configuration
    chroma_persist_directory: str = Field(
        default="./data/chroma_db",
        env="CHROMA_PERSIST_DIRECTORY"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    
    # Application Configuration
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    
    # RAG Configuration
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(default=5, env="TOP_K_RETRIEVAL")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # LangGraph Configuration
    max_iterations: int = Field(default=10, env="MAX_ITERATIONS")
    timeout_seconds: int = Field(default=300, env="TIMEOUT_SECONDS")
    
    class Config:
        env_file = ENV_FILE_PATH
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
