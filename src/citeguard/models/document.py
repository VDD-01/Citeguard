"""Document data models representing extracted pages.

Matches Section 2.2 of Barua et al. (2026):
P_i = {text_i, pdf_page_i, printed_page_i}
"""

from typing import Optional, Union
from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """Represents a single extracted page from a source document.

    Attributes:
        document_id: Unique identifier for the document (e.g. filename).
        page_number: 1-indexed canonical PDF page number.
        printed_page_number: Optional human-visible page number extracted from
            headers/footers. If unidentifiable, defaults to page_number.
        text: Extracted and normalized text content of the page.
    """
    document_id: str = Field(..., description="Unique identifier for the document (e.g., filename or hash)")
    page_number: int = Field(..., ge=1, description="1-indexed physical PDF page number")
    printed_page_number: Optional[Union[int, str]] = Field(
        default=None,
        description="Human-visible page number on the document (fallback to page_number)",
    )
    text: str = Field(..., description="Normalized text content extracted from this page")

    def get_display_page(self) -> str:
        """Return the most user-friendly page representation."""
        if self.printed_page_number is not None:
            return f"Page {self.printed_page_number} (PDF page {self.page_number})"
        return f"Page {self.page_number}"
