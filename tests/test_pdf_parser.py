"""Unit tests for PDF parsing and DocumentPage data model (Phase 1).

Uses synthetic PDFs dynamically created in memory or temporary files with PyMuPDF
to ensure reproducible, zero-external-dependency offline testing.
"""

from pathlib import Path
import pytest
import pymupdf

from src.citeguard.models.document import DocumentPage
from src.citeguard.ingestion.pdf_parser import PDFParser, parse_pdf


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a temporary multi-page PDF document for testing."""
    pdf_path = tmp_path / "test_document.pdf"
    doc = pymupdf.open()

    # Page 1: Standard text with printed page number
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((50, 72), "CiteGuard-RAG is a validation-centered AI system for question answering.")
    p1.insert_text((50, 800), "Page 1")

    # Page 2: Second section with another printed page number
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((50, 72), "In legal QA, answers must remain strictly faithful to source documents.")
    p2.insert_text((50, 800), "Page 2")

    # Page 3: Blank/empty page
    doc.new_page(width=595, height=842)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_document_page_model():
    """Test DocumentPage Pydantic model initialization and methods."""
    page = DocumentPage(
        document_id="doc_01.pdf",
        page_number=1,
        printed_page_number=5,
        text="Sample legal text.",
    )
    assert page.document_id == "doc_01.pdf"
    assert page.page_number == 1
    assert page.printed_page_number == 5
    assert page.text == "Sample legal text."
    assert "Page 5" in page.get_display_page()

    # Fallback when printed_page_number is None
    page2 = DocumentPage(
        document_id="doc_02.pdf",
        page_number=3,
        text="Another legal snippet.",
    )
    assert page2.get_display_page() == "Page 3"


def test_parse_pdf_success(sample_pdf: Path):
    """Test that parse_pdf correctly extracts text page by page."""
    pages = parse_pdf(sample_pdf)

    # Must return 3 pages matching the PDF structure
    assert len(pages) == 3

    # Check page 1
    assert pages[0].page_number == 1
    assert "CiteGuard-RAG is a validation-centered AI system" in pages[0].text
    assert pages[0].document_id == sample_pdf.name
    assert pages[0].printed_page_number == 1

    # Check page 2
    assert pages[1].page_number == 2
    assert "legal QA" in pages[1].text
    assert pages[1].printed_page_number == 2

    # Check page 3 (empty page handling)
    assert pages[2].page_number == 3
    assert pages[2].text == ""


def test_parse_pdf_custom_document_id(sample_pdf: Path):
    """Test custom document_id override."""
    pages = parse_pdf(sample_pdf, document_id="housing_law_manual")
    assert len(pages) == 3
    for p in pages:
        assert p.document_id == "housing_law_manual"


def test_parse_pdf_nonexistent_file(tmp_path: Path):
    """Test that missing files raise FileNotFoundError."""
    missing_file = tmp_path / "non_existent.pdf"
    with pytest.raises(FileNotFoundError):
        parse_pdf(missing_file)


def test_parse_pdf_corrupted_file(tmp_path: Path):
    """Test that corrupted/invalid PDF files raise ValueError."""
    corrupt_file = tmp_path / "corrupted.pdf"
    corrupt_file.write_bytes(b"This is not a real PDF document.")
    with pytest.raises(ValueError, match="Could not read PDF"):
        parse_pdf(corrupt_file)


def test_printed_page_number_regex():
    """Test heuristic printed page number detection regex."""
    parser = PDFParser()

    assert parser._extract_printed_page_number("Header\nSome text\nPage 14 of 50") == 14
    assert parser._extract_printed_page_number("Header\nSome text\nPage 42") == 42
    assert parser._extract_printed_page_number("Title\n- 8 -") == 8
    assert parser._extract_printed_page_number("Just plain text with no page numbers") is None
