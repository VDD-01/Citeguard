"""CiteGuard retrieval, deduplication, and sufficiency checking modules."""

from src.citeguard.retrieval.embeddings import EmbeddingModel, get_embedding_model
from src.citeguard.retrieval.semantic_retriever import SemanticRetriever
from src.citeguard.retrieval.bm25_retriever import BM25Retriever
from src.citeguard.retrieval.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from src.citeguard.retrieval.deduplicator import EvidenceDeduplicator, check_evidence_sufficiency

__all__ = [
    "EmbeddingModel",
    "get_embedding_model",
    "SemanticRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "get_hybrid_retriever",
    "EvidenceDeduplicator",
    "check_evidence_sufficiency",
]
