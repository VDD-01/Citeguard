"""BM25 lexical retrieval module for CiteGuard-RAG.

Implements Okapi BM25 sparse keyword retrieval on DocumentChunk collections,
enforcing consistent tokenization, preserving chunk metadata, and returning raw BM25 scores.

Matches Section 2.3 of Barua et al. (2026):
Score_lexical(q, C_i) = ∑_{t ∈ q} IDF(t) · [tf_{t,i}(k1 + 1)] / [tf_{t,i} + k1(1 - b + b · |C_i| / avgdl)]
"""

from typing import List, Optional, Tuple
import re
import numpy as np
from rank_bm25 import BM25Okapi

from src.citeguard.models.chunk import DocumentChunk
from src.citeguard.utils.logging import setup_logger

logger = setup_logger(__name__)


class BM25Retriever:
    """Sparse keyword retriever using Okapi BM25."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 retriever.

        Args:
            k1: BM25 term frequency saturation parameter (default 1.5).
            b: BM25 document length normalization parameter (default 0.75).
        """
        self.k1 = k1
        self.b = b
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric tokens.

        Args:
            text: Raw input string.

        Returns:
            List[str]: Cleaned word tokens.
        """
        if not text:
            return []
        return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())

    def index(self, chunks: List[DocumentChunk]) -> None:
        """Build the BM25 index from a list of DocumentChunk objects.

        Args:
            chunks: List of DocumentChunk objects to index.
        """
        self.chunks = list(chunks)
        if not self.chunks:
            self.bm25 = None
            self._tokenized_corpus = []
            logger.warning("Indexed 0 chunks in BM25Retriever.")
            return

        self._tokenized_corpus = [self.tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
        logger.info(f"BM25Retriever successfully indexed {len(self.chunks)} chunks.")

    def search(
        self,
        query: str,
        top_k: int = 8,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search the indexed chunks using BM25 keyword matching.

        Args:
            query: The user query string.
            top_k: Maximum number of results to return.

        Returns:
            List[Tuple[DocumentChunk, float]]: Ranked list of (chunk, raw_bm25_score)
                sorted in descending order.
        """
        if not query or not query.strip():
            logger.debug("Empty query passed to BM25Retriever.")
            return []

        if not self.chunks or self.bm25 is None:
            logger.warning("Search called on an empty BM25 index.")
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Get raw BM25 scores across all corpus documents
        raw_scores = np.array(self.bm25.get_scores(query_tokens), dtype=np.float32)

        k = min(top_k, len(self.chunks))
        if k <= 0:
            return []

        # Rank descending by score
        ranked_indices = np.argsort(raw_scores)[::-1][:k]

        results: List[Tuple[DocumentChunk, float]] = [
            (self.chunks[idx], float(raw_scores[idx])) for idx in ranked_indices
        ]

        logger.debug(f"BM25Retriever retrieved top-{len(results)} chunks for query: '{query[:40]}...'")
        return results
