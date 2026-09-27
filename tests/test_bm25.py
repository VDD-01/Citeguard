"""Unit tests for BM25 lexical retrieval (Phase 5).

Tests:
- Tokenization behavior and punctuation handling
- Exact keyword search ranking accuracy
- Top-k result limits
- Non-matching query terms
- Empty query and empty index handling
"""

import pytest
from src.citeguard.models.chunk import DocumentChunk
from src.citeguard.retrieval.bm25_retriever import BM25Retriever


@pytest.fixture
def bm25_chunks() -> list[DocumentChunk]:
    """Provide synthetic legal document chunks with distinct terminology."""
    return [
        DocumentChunk(
            chunk_id="chunk_lease",
            document_id="lease.pdf",
            text="The standard residential lease requires payment of rent on the first of each calendar month.",
            page_start=1,
            page_end=1,
        ),
        DocumentChunk(
            chunk_id="chunk_eviction",
            document_id="eviction_rules.pdf",
            text="Under the statutory eviction moratorium, no writ of restitution may be executed without 14 days notice.",
            page_start=5,
            page_end=5,
        ),
        DocumentChunk(
            chunk_id="chunk_deposit",
            document_id="security_deposit.pdf",
            text="The landlord must return the security deposit within 30 days accompanied by an itemized list of deductions.",
            page_start=2,
            page_end=2,
        ),
    ]


def test_bm25_tokenize():
    """Test tokenization helper."""
    text = "Section 4.1(a): Landlord's Right-of-Entry & Notice!"
    tokens = BM25Retriever.tokenize(text)
    assert "section" in tokens
    assert "4" in tokens
    assert "1" in tokens
    assert "landlord" in tokens
    assert "notice" in tokens
    # Punctuation should be removed
    assert ":" not in tokens
    assert "!" not in tokens


def test_bm25_search_exact_match(bm25_chunks):
    """Test that BM25 accurately surfaces chunks containing exact keyword matches."""
    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.index(bm25_chunks)

    # Query matching eviction moratorium chunk
    results = retriever.search("eviction moratorium writ of restitution", top_k=2)
    assert len(results) == 2
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "chunk_eviction"
    assert score > 0
    assert "eviction moratorium" in top_chunk.text

    # Query matching security deposit chunk
    deposit_results = retriever.search("itemized list of deductions deposit", top_k=1)
    assert len(deposit_results) == 1
    assert deposit_results[0][0].chunk_id == "chunk_deposit"


def test_bm25_search_top_k(bm25_chunks):
    """Test top-k capping and ranking order."""
    retriever = BM25Retriever()
    retriever.index(bm25_chunks)

    results = retriever.search("lease rent", top_k=1)
    assert len(results) == 1
    assert results[0][0].chunk_id == "chunk_lease"

    # top_k larger than corpus returns all indexed chunks
    results_all = retriever.search("the", top_k=10)
    assert len(results_all) == 3


def test_bm25_empty_query_and_empty_index():
    """Test handling of empty query or unindexed state."""
    retriever = BM25Retriever()
    # Unindexed
    assert retriever.search("rent", top_k=3) == []

    # Indexed with empty query
    retriever.index([DocumentChunk(chunk_id="c1", document_id="d1", text="Sample", page_start=1, page_end=1)])
    assert retriever.search("", top_k=3) == []
    assert retriever.search("   ", top_k=3) == []
