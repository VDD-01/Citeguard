"""Central configuration management for CiteGuard-RAG.

Loads settings from YAML files, applies sensible defaults based on the research paper,
and supports environment variable overrides.
"""

from pathlib import Path
from typing import Optional
import os
import yaml
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configuration for dense sentence embeddings."""
    model_name: str = "BAAI/bge-base-en-v1.5"
    device: str = "cpu"
    normalize: bool = True


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    chunk_size: int = 600
    overlap: int = 120
    min_chunk_length: int = 50


class RetrievalConfig(BaseModel):
    """Configuration for hybrid retrieval and fusion."""
    semantic_weight: float = 0.65
    lexical_weight: float = 0.35
    retrieval_top_k: int = 8
    generation_chunks: int = 4


class EvidenceConfig(BaseModel):
    """Configuration for evidence sufficiency evaluation."""
    minimum_score: float = 0.15
    refusal_message: str = (
        "The document does not contain sufficient information to answer this question."
    )


class ValidationConfig(BaseModel):
    """Configuration for sentence-level grounding validation."""
    lexical_threshold: float = 0.10
    semantic_threshold: float = 0.40
    supported_sentence_ratio_threshold: float = 1.0


class GenerationConfig(BaseModel):
    """Configuration for LLM generation."""
    temperature: float = 0.0
    max_regeneration_attempts: int = 1
    provider: str = "ollama"


class LoggingConfig(BaseModel):
    """Configuration for application logging."""
    level: str = "INFO"


class AppConfig(BaseModel):
    """Root application configuration."""
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: Optional[Path | str] = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to default config/config.yaml.

    Args:
        config_path: Optional path to the configuration YAML file.

    Returns:
        AppConfig: Validated application configuration instance.
    """
    if config_path is None:
        # Default to config/config.yaml relative to workspace root
        base_dir = Path(__file__).resolve().parent.parent.parent
        default_path = base_dir / "config" / "config.yaml"
        if default_path.exists():
            config_path = default_path

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            # Apply environment variable overrides if present
            if os.getenv("EMBEDDING_DEVICE"):
                data.setdefault("embedding", {})["device"] = os.getenv("EMBEDDING_DEVICE")
            if os.getenv("LOG_LEVEL"):
                data.setdefault("logging", {})["level"] = os.getenv("LOG_LEVEL")
            return AppConfig(**data)

    return AppConfig()
