"""Unit tests for evidence deduplication and sufficiency checks (Phase 7).

Tests:
- Elimination of exact duplicate chunks
- Elimination of near-duplicate overlapping chunks
- Preservation of diverse evidence chunks
- Generation context chunk truncation (up to 4 chunks)
- Evidence sufficiency threshold evaluation and standardized refusal trigger
"""

import pytest
from src.citeguard.models.chunk import DocumentChunk
from src.citeguard.models.retrieval import RetrievedChunk
from src.citeguard.retrieval.deduplicator import EvidenceDeduplicator, check_evidence_sufficiency


def make_retrieved_chunk(chunk_id: str, text: str, hybrid_score: float) -> RetrievedChunk:
    """Helper to construct a RetrievedChunk."""
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc.pdf",
        text=text,
        page_start=1,
        page_end=1,
    )
    return RetrievedChunk(
        chunk=chunk,
        semantic_score=hybrid_score,
        lexical_score=hybrid_score,
        normalized_semantic_score=hybrid_score,
        normalized_lexical_score=hybrid_score,
        hybrid_score=hybrid_score,
    )


def test_deduplicator_exact_duplicates():
    """Test that identical text chunks are collapsed into the highest-scoring candidate."""
    c1 = make_retrieved_chunk("c1", "Tenants must provide written notice of repairs.", 0.90)
    c2 = make_retrieved_chunk("c2", "Tenants must provide written notice of repairs.", 0.70)

    dedup = EvidenceDeduplicator(similarity_threshold=0.85)
    selected = dedup.deduplicate([c1, c2], max_chunks=4)

    assert len(selected) == 1
    assert selected[0].chunk.chunk_id == "c1"
    assert selected[0].hybrid_score == 0.90


def test_deduplicator_near_duplicates():
    """Test that overlapping near-duplicate chunks are removed."""
    # c1 and c2 share almost all text except a small prefix/suffix
    text1 = "Landlords must maintain hot water and heating facilities in sanitary operational condition during winter months."
    text2 = "Notice: Landlords must maintain hot water and heating facilities in sanitary operational condition during winter."

    c1 = make_retrieved_chunk("c1", text1, 0.85)
    c2 = make_retrieved_chunk("c2", text2, 0.80)
    c3 = make_retrieved_chunk("c3", "Parking permits are distributed annually by the leasing office.", 0.60)

    dedup = EvidenceDeduplicator(similarity_threshold=0.75)
    selected = dedup.deduplicate([c1, c2, c3], max_chunks=4)

    # c1 and c3 should be selected, c2 should be discarded as near-duplicate of c1
    assert len(selected) == 2
    assert selected[0].chunk.chunk_id == "c1"
    assert selected[1].chunk.chunk_id == "c3"


def test_deduplicator_max_chunks_cap():
    """Test that deduplicator respects max_chunks (e.g. 4 chunks passed to generator)."""
    candidates = [
        make_retrieved_chunk(f"c{i}", f"Distinct rule number {i} regarding tenant regulations.", 1.0 - i * 0.1)
        for i in range(8)
    ]

    dedup = EvidenceDeduplicator()
    selected = dedup.deduplicate(candidates, max_chunks=4)

    assert len(selected) == 4
    # Highest scores are kept
    assert [c.chunk.chunk_id for c in selected] == ["c0", "c1", "c2", "c3"]


def test_evidence_sufficiency_check_pass():
    """Test that evidence with scores above threshold passes sufficiency check."""
    evidence = [
        make_retrieved_chunk("c1", "Relevant eviction statutes.", 0.65),
        make_retrieved_chunk("c2", "Additional procedural guidance.", 0.40),
    ]
    is_sufficient, refusal = check_evidence_sufficiency(evidence, minimum_score=0.15)
    assert is_sufficient is True
    assert refusal == ""


def test_evidence_sufficiency_check_fail_low_score():
    """Test that weak retrieval scores trigger standardized refusal."""
    evidence = [
        make_retrieved_chunk("c1", "Totally unrelated text about office holidays.", 0.08),
        make_retrieved_chunk("c2", "Another low scoring chunk.", 0.05),
    ]
    is_sufficient, refusal = check_evidence_sufficiency(evidence, minimum_score=0.15)
    assert is_sufficient is False
    assert "The document does not contain sufficient information" in refusal


def test_evidence_sufficiency_check_empty():
    """Test that empty retrieval triggers standardized refusal."""
    is_sufficient, refusal = check_evidence_sufficiency([], minimum_score=0.15)
    assert is_sufficient is False
    assert "The document does not contain sufficient information" in refusal
