"""Semantic retrieval module for CiteGuard-RAG.

Performs dense vector retrieval using sentence-transformers embeddings.
Ranks document chunks against a user query based purely on cosine similarity.

Matches Section 2.3 of Barua et al. (2026):
Sim_semantic(q, C_i) = (q · c_i) / (||q|| ||c_i||)
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from src.citeguard.models.chunk import DocumentChunk
from src.citeguard.retrieval.embeddings import EmbeddingModel, get_embedding_model
from src.citeguard.utils.logging import setup_logger

logger = setup_logger(__name__)


class SemanticRetriever:
    """Dense retriever that indexes DocumentChunks and performs cosine similarity search."""

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        """Initialize the semantic retriever.

        Args:
            embedding_model: EmbeddingModel instance. If None, loaded from global config.
        """
        self.embedding_model = embedding_model or get_embedding_model()
        self.chunks: List[DocumentChunk] = []
        self.chunk_embeddings: Optional[np.ndarray] = None

    def index(
        self,
        chunks: List[DocumentChunk],
        cache_name: Optional[str] = None,
    ) -> None:
        """Index a collection of document chunks by generating or loading embeddings.

        Args:
            chunks: List of DocumentChunk objects to index.
            cache_name: Optional persistent cache key name.
        """
        self.chunks = list(chunks)
        if not self.chunks:
            self.chunk_embeddings = np.empty((0, 768), dtype=np.float32)
            logger.warning("Indexed 0 chunks in SemanticRetriever.")
            return

        self.chunk_embeddings = self.embedding_model.get_or_compute_chunk_embeddings(
            self.chunks,
            cache_name=cache_name,
        )
        logger.info(
            f"SemanticRetriever indexed {len(self.chunks)} chunks "
            f"(embedding shape: {self.chunk_embeddings.shape})."
        )

    def search(
        self,
        query: str,
        top_k: int = 8,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search the indexed chunks using dense semantic similarity.

        Args:
            query: The user query string.
            top_k: Number of highest-scoring chunks to return.

        Returns:
            List[Tuple[DocumentChunk, float]]: Ranked list of (chunk, cosine_similarity_score)
                sorted in descending order of similarity.
        """
        if not query or not query.strip():
            logger.debug("Empty query passed to SemanticRetriever.")
            return []

        if not self.chunks or self.chunk_embeddings is None or len(self.chunks) == 0:
            logger.warning("Search called on an empty SemanticRetriever index.")
            return []

        # Encode query
        query_embedding = self.embedding_model.encode_query(query)

        # Compute cosine similarity against all indexed chunk embeddings
        scores = self.embedding_model.similarity(query_embedding, self.chunk_embeddings)

        # Determine top_k indices sorted descending
        k = min(top_k, len(self.chunks))
        if k <= 0:
            return []

        # argpartition or argsort for ranking
        ranked_indices = np.argsort(scores)[::-1][:k]

        results: List[Tuple[DocumentChunk, float]] = [
            (self.chunks[idx], float(scores[idx])) for idx in ranked_indices
        ]

        logger.debug(
            f"SemanticRetriever retrieved top-{len(results)} chunks for query: '{query[:40]}...'"
        )
        return results
