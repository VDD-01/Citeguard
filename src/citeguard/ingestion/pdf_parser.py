"""Page-aware PDF parser for CiteGuard-RAG.

Extracts text from PDF documents page by page, preserving PDF canonical page numbers
and attempting to recover printed page numbers from headers/footers.

Matches Section 2.2 of Barua et al. (2026):
P_i = {text_i, pdf_page_i, printed_page_i}
"""

from pathlib import Path
from typing import List, Optional, Union
import re
import pymupdf

from src.citeguard.models.document import DocumentPage
from src.citeguard.utils.logging import setup_logger
from src.citeguard.utils.text import normalize_text

logger = setup_logger(__name__)


class PDFParser:
    """Extracts page-aware text from PDF documents using PyMuPDF.

    Ensures that document provenance (document identity, canonical page numbers,
    and printed page numbers) is retained without concatenating text across pages.
    """

    def __init__(self):
        """Initialize the PDF parser."""
        pass

    @staticmethod
    def _extract_printed_page_number(page_text: str) -> Optional[int]:
        """Heuristically inspect header and footer lines to detect printed page numbers.

        Looks for common patterns such as:
        - "Page 12" or "Page 12 of 45"
        - "- 12 -"
        - Standalone numbers at the very start or end of the page text.

        Args:
            page_text: Raw or normalized page text.

        Returns:
            Optional[int]: Detected printed page number, or None if not found.
        """
        if not page_text or not page_text.strip():
            return None

        lines = [line.strip() for line in page_text.strip().split("\n") if line.strip()]
        if not lines:
            return None

        # Inspect first 2 lines (header) and last 2 lines (footer)
        candidate_lines = lines[:2] + lines[-2:]

        # Pattern 1: "Page X of Y" or "Page X"
        page_pattern = re.compile(r"^\s*(?:page|pg\.?)\s*(\d+)(?:\s*(?:of|/)\s*\d+)?\s*$", re.IGNORECASE)
        for line in candidate_lines:
            match = page_pattern.match(line)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass

        # Pattern 2: "- X -" or "[X]"
        bracket_pattern = re.compile(r"^[-–—\[\(\s]*(\d+)[-–—\]\)\s]*$")
        for line in candidate_lines:
            match = bracket_pattern.match(line)
            if match:
                try:
                    num = int(match.group(1))
                    if 0 < num < 10000:
                        return num
                except ValueError:
                    pass

        return None

    def parse(
        self,
        pdf_path: Union[str, Path],
        document_id: Optional[str] = None,
    ) -> List[DocumentPage]:
        """Parse a PDF document into a sequence of page-aware DocumentPage objects.

        Args:
            pdf_path: Path to the input PDF file.
            document_id: Optional identifier for the document. Defaults to the filename.

        Returns:
            List[DocumentPage]: Extracted pages with page numbers and text.

        Raises:
            FileNotFoundError: If the specified PDF path does not exist.
            ValueError: If the file is not a valid or readable PDF.
        """
        path = Path(pdf_path)
        if not path.exists():
            logger.error(f"PDF file not found: {path}")
            raise FileNotFoundError(f"PDF file not found at: {path}")

        if not path.is_file():
            logger.error(f"Provided path is not a file: {path}")
            raise ValueError(f"Expected a file path, got: {path}")

        doc_id = document_id if document_id else path.name
        pages: List[DocumentPage] = []

        try:
            doc = pymupdf.open(str(path))
        except Exception as e:
            logger.error(f"Failed to open PDF '{path}': {str(e)}")
            raise ValueError(f"Could not read PDF '{path}'. It may be corrupted or encrypted: {e}")

        total_pages = len(doc)
        logger.info(f"Parsing PDF '{doc_id}' ({total_pages} pages)...")

        try:
            for page_index in range(total_pages):
                page_number = page_index + 1  # 1-indexed canonical PDF page
                try:
                    page = doc[page_index]
                    raw_text = page.get_text("text") or ""
                    clean_text = normalize_text(raw_text)

                    printed_page = self._extract_printed_page_number(raw_text)
                    if printed_page is None:
                        # Fallback to canonical page number
                        printed_page = page_number

                    if not clean_text:
                        logger.warning(
                            f"Page {page_number} of '{doc_id}' is empty or contains no extractable text."
                        )

                    doc_page = DocumentPage(
                        document_id=doc_id,
                        page_number=page_number,
                        printed_page_number=printed_page,
                        text=clean_text,
                    )
                    pages.append(doc_page)

                except Exception as page_err:
                    # Robustness: log page extraction errors without crashing entire document processing
                    logger.error(
                        f"Error extracting text from page {page_number} of '{doc_id}': {str(page_err)}"
                    )
                    # Insert empty page placeholder to preserve sequential page numbering
                    pages.append(
                        DocumentPage(
                            document_id=doc_id,
                            page_number=page_number,
                            printed_page_number=page_number,
                            text="",
                        )
                    )
        finally:
            doc.close()

        logger.info(f"Successfully extracted {len(pages)} pages from '{doc_id}'.")
        return pages


def parse_pdf(
    pdf_path: Union[str, Path],
    document_id: Optional[str] = None,
) -> List[DocumentPage]:
    """Convenience function to parse a PDF file into DocumentPage objects.

    Args:
        pdf_path: Path to the PDF file.
        document_id: Optional document ID (defaults to filename).

    Returns:
        List[DocumentPage]: Extracted document pages.
    """
    parser = PDFParser()
    return parser.parse(pdf_path=pdf_path, document_id=document_id)
