"""Hybrid retrieval module for CiteGuard-RAG.

Combines dense semantic retrieval and sparse lexical retrieval (BM25)
using min-max normalization and weighted score fusion.

Matches Section 2.3 of Barua et al. (2026):
Score_hybrid = α · ŝ_semantic + (1 - α) · ŝ_lexical
Default: α = 0.65, 1 - α = 0.35.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.citeguard.config import RetrievalConfig, load_config
from src.citeguard.models.chunk import DocumentChunk
from src.citeguard.models.retrieval import RetrievedChunk
from src.citeguard.retrieval.bm25_retriever import BM25Retriever
from src.citeguard.retrieval.semantic_retriever import SemanticRetriever
from src.citeguard.utils.logging import setup_logger

logger = setup_logger(__name__)


class HybridRetriever:
    """Combines dense semantic retrieval and sparse lexical BM25 retrieval."""

    def __init__(
        self,
        semantic_retriever: Optional[SemanticRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        semantic_weight: float = 0.65,
        lexical_weight: float = 0.35,
        retrieval_top_k: int = 8,
    ):
        """Initialize the hybrid retriever.

        Args:
            semantic_retriever: Dense retriever instance.
            bm25_retriever: Sparse BM25 retriever instance.
            semantic_weight: Alpha weight for semantic scores (default 0.65).
            lexical_weight: Weight for lexical scores (default 0.35).
            retrieval_top_k: Number of candidate chunks to return (default 8).
        """
        self.semantic_retriever = semantic_retriever or SemanticRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.retrieval_top_k = retrieval_top_k
        self.chunks_by_id: Dict[str, DocumentChunk] = {}

    def index(
        self,
        chunks: List[DocumentChunk],
        cache_name: Optional[str] = None,
    ) -> None:
        """Index chunks in both semantic and lexical retrievers.

        Args:
            chunks: List of DocumentChunk objects.
            cache_name: Optional cache name for precomputed embeddings.
        """
        self.chunks_by_id = {c.chunk_id: c for c in chunks}
        logger.info(f"HybridRetriever indexing {len(chunks)} chunks across dense and sparse engines...")
        self.semantic_retriever.index(chunks, cache_name=cache_name)
        self.bm25_retriever.index(chunks)
        logger.info("HybridRetriever indexing complete.")

    @staticmethod
    def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize a dictionary of chunk_id -> raw_score into [0.0, 1.0] using min-max scaling.

        Handles edge cases:
        - Empty score dictionary.
        - Identical scores (max_val == min_val): assigns 1.0 if score > 0, else 0.0.

        Args:
            scores: Dictionary mapping chunk_id to raw numeric score.

        Returns:
            Dict[str, float]: Normalized scores in [0.0, 1.0].
        """
        if not scores:
            return {}

        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)

        # Edge case: all scores are identical
        if abs(max_val - min_val) < 1e-9:
            uniform_val = 1.0 if max_val > 0 else 0.0
            return {k: uniform_val for k in scores}

        norm_scores = {}
        for k, v in scores.items():
            norm_val = (v - min_val) / (max_val - min_val)
            norm_scores[k] = float(np.clip(norm_val, 0.0, 1.0))

        return norm_scores

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Perform hybrid retrieval by querying both retrievers, normalizing scores, and fusing.

        Args:
            query: The user search query.
            top_k: Number of candidates to return. Defaults to self.retrieval_top_k.

        Returns:
            List[RetrievedChunk]: Ranked list of RetrievedChunk objects sorted descending
                by hybrid_score.
        """
        k = top_k if top_k is not None else self.retrieval_top_k
        if not query or not query.strip() or not self.chunks_by_id:
            logger.debug("Empty query or unindexed HybridRetriever.")
            return []

        # Retrieve a broader pool of candidates from each engine to ensure good candidate union
        candidate_pool_size = max(k * 2, 20)
        sem_results = self.semantic_retriever.search(query, top_k=candidate_pool_size)
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_pool_size)

        # Collect unique chunk IDs from both retrievers
        sem_scores_raw: Dict[str, float] = {chunk.chunk_id: score for chunk, score in sem_results}
        bm25_scores_raw: Dict[str, float] = {chunk.chunk_id: score for chunk, score in bm25_results}

        all_candidate_ids = set(sem_scores_raw.keys()).union(set(bm25_scores_raw.keys()))
        if not all_candidate_ids:
            return []

        # For any candidate retrieved by one engine but missing from the other:
        # - Default missing BM25 score to 0.0
        # - Default missing semantic score to minimum observed semantic score or 0.0
        min_sem = min(sem_scores_raw.values()) if sem_scores_raw else 0.0
        for cid in all_candidate_ids:
            if cid not in sem_scores_raw:
                sem_scores_raw[cid] = min(0.0, min_sem)
            if cid not in bm25_scores_raw:
                bm25_scores_raw[cid] = 0.0

        # Min-max normalize both score sets
        norm_sem = self.min_max_normalize(sem_scores_raw)
        norm_lex = self.min_max_normalize(bm25_scores_raw)

        # Fuse scores using hybrid formula: Score_hybrid = α · ŝ_semantic + (1 - α) · ŝ_lexical
        retrieved_chunks: List[RetrievedChunk] = []
        for cid in all_candidate_ids:
            chunk = self.chunks_by_id.get(cid)
            if chunk is None:
                continue

            s_sem_raw = sem_scores_raw[cid]
            s_lex_raw = bm25_scores_raw[cid]
            s_sem_norm = norm_sem[cid]
            s_lex_norm = norm_lex[cid]

            fused_score = (self.semantic_weight * s_sem_norm) + (self.lexical_weight * s_lex_norm)

            retrieved_chunk = RetrievedChunk(
                chunk=chunk,
                semantic_score=float(s_sem_raw),
                lexical_score=float(s_lex_raw),
                normalized_semantic_score=float(s_sem_norm),
                normalized_lexical_score=float(s_lex_norm),
                hybrid_score=float(fused_score),
            )
            retrieved_chunks.append(retrieved_chunk)

        # Sort descending by hybrid_score
        retrieved_chunks.sort(key=lambda x: x.hybrid_score, reverse=True)

        final_results = retrieved_chunks[:k]
        logger.info(
            f"HybridRetriever fused {len(all_candidate_ids)} candidates into top-{len(final_results)} chunks "
            f"(top score: {final_results[0].hybrid_score:.4f} if results else 0.0)."
        )
        return final_results


def get_hybrid_retriever(
    config: Optional[RetrievalConfig] = None,
    semantic_retriever: Optional[SemanticRetriever] = None,
    bm25_retriever: Optional[BM25Retriever] = None,
) -> HybridRetriever:
    """Factory helper to instantiate a HybridRetriever configured via config.yaml.

    Args:
        config: Optional RetrievalConfig. If None, loaded from global configuration.
        semantic_retriever: Optional pre-configured SemanticRetriever.
        bm25_retriever: Optional pre-configured BM25Retriever.

    Returns:
        HybridRetriever: Configured hybrid retriever.
    """
    if config is None:
        app_cfg = load_config()
        config = app_cfg.retrieval

    return HybridRetriever(
        semantic_retriever=semantic_retriever,
        bm25_retriever=bm25_retriever,
        semantic_weight=config.semantic_weight,
        lexical_weight=config.lexical_weight,
        retrieval_top_k=config.retrieval_top_k,
    )
