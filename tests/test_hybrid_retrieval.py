"""Unit tests for hybrid retrieval and score fusion (Phase 6).

Tests:
- Min-max score normalization and edge cases (identical scores, empty dict)
- Multi-engine score fusion (0.65 semantic + 0.35 lexical)
- Ranking behavior combining semantic paraphrase and exact keyword matching
- Configurable alpha weighting adjustments
- Top-k limits and empty query handling
"""

import pytest
from src.citeguard.models.chunk import DocumentChunk
from src.citeguard.models.retrieval import RetrievedChunk
from src.citeguard.retrieval.bm25_retriever import BM25Retriever
from src.citeguard.retrieval.embeddings import EmbeddingModel
from src.citeguard.retrieval.semantic_retriever import SemanticRetriever
from src.citeguard.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture(scope="module")
def hybrid_chunks() -> list[DocumentChunk]:
    """Provide synthetic legal chunks with distinct keyword and conceptual characteristics."""
    return [
        DocumentChunk(
            chunk_id="chunk_statute_escrow",
            document_id="housing_code.pdf",
            text="Under Section 504-A, tenants may initiate rent escrow procedures if the landlord fails to remedy code violations.",
            page_start=4,
            page_end=4,
        ),
        DocumentChunk(
            chunk_id="chunk_habitable_paraphrase",
            document_id="tenants_rights.pdf",
            text="A renter can deposit payments with the court when dwelling conditions become hazardous, unlivable, or severely defective.",
            page_start=8,
            page_end=8,
        ),
        DocumentChunk(
            chunk_id="chunk_library_unrelated",
            document_id="community_guide.pdf",
            text="The city public library is open from 9:00 AM to 5:00 PM on Saturdays for book borrowing and community workshops.",
            page_start=12,
            page_end=12,
        ),
    ]


@pytest.fixture(scope="module")
def hybrid_retriever(tmp_path_factory, hybrid_chunks) -> HybridRetriever:
    """Fixture initializing HybridRetriever with fast offline model."""
    cache_dir = tmp_path_factory.mktemp("hybrid_cache")
    emb_model = EmbeddingModel(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        normalize=True,
        cache_dir=cache_dir,
    )
    sem_retriever = SemanticRetriever(embedding_model=emb_model)
    bm25_retriever = BM25Retriever()

    hybrid = HybridRetriever(
        semantic_retriever=sem_retriever,
        bm25_retriever=bm25_retriever,
        semantic_weight=0.65,
        lexical_weight=0.35,
        retrieval_top_k=8,
    )
    hybrid.index(hybrid_chunks)
    return hybrid


def test_min_max_normalize():
    """Test min-max normalization behavior and edge cases."""
    # Typical case
    raw = {"a": 10.0, "b": 20.0, "c": 30.0}
    norm = HybridRetriever.min_max_normalize(raw)
    assert norm["a"] == 0.0
    assert norm["b"] == 0.5
    assert norm["c"] == 1.0

    # Identical positive scores
    identical = {"a": 5.0, "b": 5.0}
    norm_ident = HybridRetriever.min_max_normalize(identical)
    assert norm_ident["a"] == 1.0
    assert norm_ident["b"] == 1.0

    # Empty dictionary
    assert HybridRetriever.min_max_normalize({}) == {}


def test_hybrid_search_fusion(hybrid_retriever: HybridRetriever):
    """Test that hybrid retrieval accurately surfaces both exact keyword and semantic matches above noise."""
    # Query with both exact legal terminology ("Section 504-A") and concept ("court rent deposit")
    query = "Section 504-A rent escrow and depositing payments with the court for unlivable housing"
    results = hybrid_retriever.search(query, top_k=3)

    assert len(results) == 3
    # Top 2 must be the escrow and habitable chunks, unrelated library must be lowest
    result_ids = [r.chunk.chunk_id for r in results]
    assert "chunk_library_unrelated" == result_ids[-1]
    assert "chunk_statute_escrow" in result_ids[:2]
    assert "chunk_habitable_paraphrase" in result_ids[:2]

    # Verify score types and bounds
    for r in results:
        assert isinstance(r, RetrievedChunk)
        assert 0.0 <= r.normalized_semantic_score <= 1.0
        assert 0.0 <= r.normalized_lexical_score <= 1.0
        assert 0.0 <= r.hybrid_score <= 1.0
        # Formula check: hybrid = 0.65 * norm_sem + 0.35 * norm_lex
        expected_hybrid = 0.65 * r.normalized_semantic_score + 0.35 * r.normalized_lexical_score
        assert pytest.approx(r.hybrid_score, abs=1e-5) == expected_hybrid


def test_hybrid_custom_weights(hybrid_chunks, tmp_path_factory):
    """Test custom semantic/lexical weighting configuration."""
    cache_dir = tmp_path_factory.mktemp("weight_cache")
    emb_model = EmbeddingModel(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        cache_dir=cache_dir,
    )
    sem_retriever = SemanticRetriever(embedding_model=emb_model)
    bm25_retriever = BM25Retriever()

    # 100% BM25 weight
    bm25_dominant = HybridRetriever(
        semantic_retriever=sem_retriever,
        bm25_retriever=bm25_retriever,
        semantic_weight=0.0,
        lexical_weight=1.0,
    )
    bm25_dominant.index(hybrid_chunks)

    # Search exact keyword "Section 504-A"
    results = bm25_dominant.search("Section 504-A", top_k=1)
    assert results[0].chunk.chunk_id == "chunk_statute_escrow"
    assert results[0].hybrid_score == results[0].normalized_lexical_score


def test_hybrid_top_k_and_empty_query(hybrid_retriever: HybridRetriever):
    """Test top_k truncation and empty query handling."""
    res_1 = hybrid_retriever.search("housing code", top_k=1)
    assert len(res_1) == 1

    assert hybrid_retriever.search("", top_k=3) == []
    assert hybrid_retriever.search("   ", top_k=3) == []
