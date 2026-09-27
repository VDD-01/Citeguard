"""Unit tests for semantic retrieval (Phase 4).

Tests:
- Indexing chunks and verifying dense embeddings
- Semantic search ranking accuracy
- Top-k parameter constraints
- Empty query and empty index handling
"""

import pytest
from src.citeguard.models.chunk import DocumentChunk
from src.citeguard.retrieval.embeddings import EmbeddingModel
from src.citeguard.retrieval.semantic_retriever import SemanticRetriever


@pytest.fixture(scope="module")
def sample_chunks() -> list[DocumentChunk]:
    """Provide a diverse set of synthetic legal chunks."""
    return [
        DocumentChunk(
            chunk_id="doc_c001",
            document_id="lease_agreement.pdf",
            text="Tenants must pay the monthly rent on or before the first day of every calendar month.",
            page_start=1,
            page_end=1,
        ),
        DocumentChunk(
            chunk_id="doc_c002",
            document_id="lease_agreement.pdf",
            text="No pets exceeding twenty-five pounds are permitted on the premises without written consent.",
            page_start=2,
            page_end=2,
        ),
        DocumentChunk(
            chunk_id="doc_c003",
            document_id="lease_agreement.pdf",
            text="In the event of a plumbing emergency or gas leak, call the 24-hour maintenance hotline immediately.",
            page_start=3,
            page_end=3,
        ),
    ]


@pytest.fixture(scope="module")
def semantic_retriever(tmp_path_factory, sample_chunks) -> SemanticRetriever:
    """Initialize a SemanticRetriever with a fast test embedding model."""
    cache_dir = tmp_path_factory.mktemp("sem_cache")
    model = EmbeddingModel(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        normalize=True,
        cache_dir=cache_dir,
    )
    retriever = SemanticRetriever(embedding_model=model)
    retriever.index(sample_chunks)
    return retriever


def test_semantic_search_ranking(semantic_retriever: SemanticRetriever):
    """Test that queries rank the most semantically relevant chunk first."""
    # Query matching chunk 2 (pets)
    results = semantic_retriever.search("Are dogs and animals allowed in the apartment?", top_k=3)
    assert len(results) == 3
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "doc_c002"
    assert "No pets exceeding" in top_chunk.text
    assert score > 0.4

    # Query matching chunk 3 (emergency)
    results_emergency = semantic_retriever.search("Who do I call if there is a water leak or emergency?", top_k=3)
    assert results_emergency[0][0].chunk_id == "doc_c003"
    assert "plumbing emergency" in results_emergency[0][0].text


def test_semantic_search_top_k(semantic_retriever: SemanticRetriever):
    """Test that top_k restricts the number of returned chunks."""
    results = semantic_retriever.search("rent payment schedule", top_k=1)
    assert len(results) == 1
    assert results[0][0].chunk_id == "doc_c001"

    # top_k larger than collection returns all available chunks
    results_all = semantic_retriever.search("lease rules", top_k=10)
    assert len(results_all) == 3


def test_semantic_search_empty_query(semantic_retriever: SemanticRetriever):
    """Test that an empty query returns an empty list."""
    results = semantic_retriever.search("", top_k=5)
    assert results == []
    results_whitespace = semantic_retriever.search("   ", top_k=5)
    assert results_whitespace == []


def test_semantic_search_empty_index(tmp_path_factory):
    """Test search on an unindexed SemanticRetriever."""
    cache_dir = tmp_path_factory.mktemp("empty_sem_cache")
    model = EmbeddingModel(model_name="all-MiniLM-L6-v2", device="cpu", cache_dir=cache_dir)
    empty_retriever = SemanticRetriever(embedding_model=model)
    results = empty_retriever.search("hello", top_k=3)
    assert results == []
