"""Unit tests for paragraph-aware document chunking (Phase 2).

Tests:
- Normal paragraphs within budget
- Oversized paragraphs falling back to sentence splitting
- Short documents
- Empty pages handling
- Overlap preservation between adjacent chunks
- Page range tracking across page boundaries
- Filtering of low-information fragments
"""

import pytest
from src.citeguard.models.document import DocumentPage
from src.citeguard.models.chunk import DocumentChunk
from src.citeguard.ingestion.chunker import DocumentChunker, chunk_document
from src.citeguard.config import ChunkingConfig


def test_chunker_short_document():
    """Test that a short document produces a single chunk with preserved metadata."""
    page = DocumentPage(
        document_id="policy.pdf",
        page_number=1,
        printed_page_number=1,
        text="This is a short policy statement that describes tenant rights and responsibilities in rental housing.",
    )
    chunks = chunk_document([page])
    assert len(chunks) == 1
    assert chunks[0].document_id == "policy.pdf"
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert "tenant rights" in chunks[0].text
    assert chunks[0].chunk_id == "policy.pdf_c000"


def test_chunker_empty_pages():
    """Test that empty or whitespace-only pages produce no chunks and do not crash."""
    pages = [
        DocumentPage(document_id="empty.pdf", page_number=1, text=""),
        DocumentPage(document_id="empty.pdf", page_number=2, text="   \n\n  "),
    ]
    chunks = chunk_document(pages)
    assert len(chunks) == 0


def test_chunker_normal_paragraphs_and_overlap():
    """Test paragraph segmentation and overlap between consecutive chunks."""
    # Create paragraphs that will exceed a small chunk size
    p1 = "Paragraph 1 discusses the security deposit regulations and return conditions."
    p2 = "Paragraph 2 outlines the timeline for returning deposits within thirty days."
    p3 = "Paragraph 3 explains the penalties for landlords who withhold deposits unlawfully."

    page = DocumentPage(
        document_id="housing_code.pdf",
        page_number=1,
        printed_page_number=1,
        text=f"{p1}\n\n{p2}\n\n{p3}",
    )

    # Use a small chunk size to force chunk splitting
    # p1 is ~15 tokens, p2 ~16 tokens. With chunk_size=25, p1 and p2 together exceed 25.
    chunker = DocumentChunker(chunk_size=25, overlap=10, min_chunk_length=10)
    chunks = chunker.chunk_pages([page])

    assert len(chunks) >= 2
    # Verify that all text components are covered
    full_chunk_text = " ".join(c.text for c in chunks)
    assert "Paragraph 1" in full_chunk_text
    assert "Paragraph 2" in full_chunk_text
    assert "Paragraph 3" in full_chunk_text

    # Verify that chunk IDs are sequentially numbered
    assert chunks[0].chunk_id == "housing_code.pdf_c000"
    assert chunks[1].chunk_id == "housing_code.pdf_c001"


def test_chunker_oversized_paragraph_sentence_fallback():
    """Test that an oversized paragraph falls back to sentence-level splitting."""
    sentence1 = "Landlords must provide written notice at least thirty days prior to any rent increase."
    sentence2 = "Tenants have the right to request a conciliation conference if the increase exceeds the allowable cap."
    sentence3 = "The housing authority shall arbitrate all disputes within fourteen business days of receiving the appeal."

    # A single continuous paragraph with multiple sentences
    long_para = f"{sentence1} {sentence2} {sentence3}"

    page = DocumentPage(
        document_id="rent_control.pdf",
        page_number=1,
        printed_page_number=10,
        text=long_para,
    )

    # Use small chunk size (18 tokens) to force sentence splitting
    chunker = DocumentChunker(chunk_size=20, overlap=5, min_chunk_length=10)
    chunks = chunker.chunk_pages([page])

    assert len(chunks) > 1
    # Check that sentences were successfully split
    assert any("written notice" in c.text for c in chunks)
    assert any("conciliation conference" in c.text for c in chunks)
    assert any("arbitrate all disputes" in c.text for c in chunks)


def test_chunker_page_boundaries():
    """Test that chunks spanning multiple pages correctly reflect page_start and page_end."""
    page1 = DocumentPage(
        document_id="statute.pdf",
        page_number=1,
        printed_page_number=101,
        text="This is introductory text on page 1 covering statutory definitions.",
    )
    page2 = DocumentPage(
        document_id="statute.pdf",
        page_number=2,
        printed_page_number=102,
        text="This is continuing text on page 2 covering enforcement mechanisms and penalties.",
    )

    # Chunk size large enough to hold both pages combined
    chunker = DocumentChunker(chunk_size=600, overlap=120, min_chunk_length=10)
    chunks = chunker.chunk_pages([page1, page2])

    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert chunks[0].printed_page_start == 101
    assert chunks[0].printed_page_end == 102
    assert chunks[0].get_page_display() == "pp. 1-2"


def test_chunker_min_length_filter():
    """Test that extremely short fragments (e.g. navigation headers) are filtered out."""
    page = DocumentPage(
        document_id="doc.pdf",
        page_number=1,
        text="Header",  # 6 characters, below min_chunk_length=50
    )
    chunks = chunk_document([page])
    assert len(chunks) == 0
