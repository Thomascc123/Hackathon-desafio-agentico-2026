from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import YamlConfigSettingsSource


class ModelConfig(BaseSettings):
    provider: str = Field(default="ollama", description="LLM provider: ollama, openai, anthropic")
    name: str = Field(default="llama3.2:3b", description="Model name")
    base_url: str = Field(default="http://localhost:11434/v1", description="Base URL for Ollama")
    api_key: Optional[str] = Field(default=None, description="API key for cloud providers")


class EmbeddingConfig(BaseSettings):
    provider: str = Field(default="local", description="Embedding provider: local, openai")
    model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")
    device: str = Field(default="cpu", description="Device for embeddings: cpu, cuda")


class ChromaConfig(BaseSettings):
    persist_dir: str = Field(default="./storage/chromadb", description="ChromaDB persistence path")


class SecurityConfig(BaseSettings):
    rate_limit: int = Field(default=30, ge=1, description="Max queries per minute")
    max_query_length: int = Field(default=500, ge=1, le=2000, description="Max query length")
    audit_db: str = Field(default="./storage/audit.db", description="SQLite audit path")


class GraphConfig(BaseSettings):
    json_path: str = Field(default="./graphrag_output/knowledge_graph.json", description="Path to graph JSON export")


class DirectoriesConfig(BaseSettings):
    pdf_dir: str = Field(default="/tmp/normativa_pdfs", description="PDF download directory")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(yaml_file="config.yaml", env_prefix="NORMATIA_")

    model: ModelConfig = ModelConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chroma: ChromaConfig = ChromaConfig()
    security: SecurityConfig = SecurityConfig()
    graph: GraphConfig = GraphConfig()
    directories: DirectoriesConfig = DirectoriesConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        yaml_file = cls.model_config.get("yaml_file")
        if yaml_file:
            return (
                init_settings,
                env_settings,
                YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file),
                file_secret_settings,
            )
        return (init_settings, env_settings, file_secret_settings)


settings = Settings()
