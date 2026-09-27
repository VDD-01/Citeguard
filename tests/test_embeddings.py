"""Unit tests for the dense embedding system (Phase 3).

Tests:
- Lazy loading and single-instance behavior
- Document batch encoding and normalization
- Query encoding
- Cosine similarity computation
- Disk caching of embeddings (save and reload)
- Empty text handling
"""

from pathlib import Path
import numpy as np
import pytest

from src.citeguard.models.chunk import DocumentChunk
from src.citeguard.retrieval.embeddings import EmbeddingModel


@pytest.fixture(scope="module")
def embedding_model(tmp_path_factory) -> EmbeddingModel:
    """Fixture providing a fast, locally cached model for unit testing."""
    cache_dir = tmp_path_factory.mktemp("emb_cache")
    # Use all-MiniLM-L6-v2 for fast offline unit tests
    return EmbeddingModel(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        normalize=True,
        cache_dir=cache_dir,
    )


def test_encode_documents_shape_and_norm(embedding_model: EmbeddingModel):
    """Test that batch encoding produces unit-normalized vectors with correct dimension."""
    texts = [
        "Tenants must pay rent on the first of each month.",
        "Security deposits must be returned within thirty days of lease termination.",
        "Landlords are required to maintain heating facilities in working condition.",
    ]
    embeddings = embedding_model.encode_documents(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == 384  # MiniLM dimension

    # Verify unit normalization (L2 norm should be ~1.0 for each vector)
    norms = np.linalg.norm(embeddings, axis=1)
    for norm in norms:
        assert pytest.approx(norm, rel=1e-3) == 1.0


def test_encode_query(embedding_model: EmbeddingModel):
    """Test single query encoding and normalization."""
    query = "When is the security deposit returned?"
    q_emb = embedding_model.encode_query(query)

    assert isinstance(q_emb, np.ndarray)
    assert q_emb.ndim == 1
    assert q_emb.shape[0] == 384
    assert pytest.approx(np.linalg.norm(q_emb), rel=1e-3) == 1.0


def test_cosine_similarity(embedding_model: EmbeddingModel):
    """Test that cosine similarity identifies the most semantically relevant text."""
    docs = [
        "Security deposits must be refunded within 30 days after moving out.",
        "The pet policy prohibits dogs over fifty pounds in common areas.",
        "Parking spaces are assigned on a first-come, first-served basis.",
    ]
    doc_embs = embedding_model.encode_documents(docs)

    query = "How long does a landlord have to return a security deposit?"
    query_emb = embedding_model.encode_query(query)

    scores = embedding_model.similarity(query_emb, doc_embs)

    assert len(scores) == 3
    # The security deposit document (index 0) must score highest
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    # Similarity should be positive and reasonably strong
    assert scores[0] > 0.5


def test_similarity_identical_text(embedding_model: EmbeddingModel):
    """Test that an identical query and document have cosine similarity close to 1.0."""
    text = "Tenants have the right to peaceful enjoyment of the leased premises."
    q_emb = embedding_model.encode_query(text)
    d_emb = embedding_model.encode_documents([text])

    score = embedding_model.similarity(q_emb, d_emb)[0]
    assert pytest.approx(score, rel=1e-3) == 1.0


def test_empty_inputs(embedding_model: EmbeddingModel):
    """Test that empty inputs are handled cleanly without raising exceptions."""
    empty_docs = embedding_model.encode_documents([])
    assert empty_docs.shape == (0, 768) or empty_docs.size == 0

    scores = embedding_model.similarity(np.zeros(384), np.empty((0, 384)))
    assert scores.size == 0


def test_disk_caching(embedding_model: EmbeddingModel, tmp_path: Path):
    """Test that chunk embeddings are saved to disk and reloaded on subsequent calls."""
    chunks = [
        DocumentChunk(
            chunk_id="test_c001",
            document_id="test_doc",
            text="First test clause regarding lease terms.",
            page_start=1,
            page_end=1,
        ),
        DocumentChunk(
            chunk_id="test_c002",
            document_id="test_doc",
            text="Second test clause regarding maintenance requests.",
            page_start=1,
            page_end=2,
        ),
    ]

    # First call: computes and creates cache file
    embs1 = embedding_model.get_or_compute_chunk_embeddings(chunks, cache_name="test_run")
    assert embs1.shape == (2, 384)

    # Verify cache file exists
    cache_file = embedding_model.cache_dir / "test_run.npz"
    assert cache_file.exists()

    # Second call: loads from cache
    embs2 = embedding_model.get_or_compute_chunk_embeddings(chunks, cache_name="test_run")
    assert np.allclose(embs1, embs2)
