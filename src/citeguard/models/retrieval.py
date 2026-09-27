"""Retrieval data models representing retrieved and scored evidence chunks.

Matches Section 2.3 of Barua et al. (2026):
Score_hybrid = α * Score_semantic_norm + (1 - α) * Score_lexical_norm
"""

from pydantic import BaseModel, Field
from src.citeguard.models.chunk import DocumentChunk


class RetrievedChunk(BaseModel):
    """Represents a retrieved document chunk with individual and fused retrieval scores.

    Attributes:
        chunk: The underlying DocumentChunk.
        semantic_score: Raw cosine similarity from dense embeddings [-1.0, 1.0].
        lexical_score: Raw BM25 score >= 0.
        normalized_semantic_score: Min-max normalized semantic score [0.0, 1.0].
        normalized_lexical_score: Min-max normalized BM25 score [0.0, 1.0].
        hybrid_score: Fused weighted score α*S_sem + (1-α)*S_lex in [0.0, 1.0].
    """
    chunk: DocumentChunk = Field(..., description="Underlying document chunk")
    semantic_score: float = Field(default=0.0, description="Raw cosine similarity")
    lexical_score: float = Field(default=0.0, description="Raw BM25 score")
    normalized_semantic_score: float = Field(default=0.0, description="Min-max normalized semantic score")
    normalized_lexical_score: float = Field(default=0.0, description="Min-max normalized BM25 score")
    hybrid_score: float = Field(default=0.0, description="Weighted hybrid fusion score")
