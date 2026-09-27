"""Chunk data models representing segmented document units.

Matches Section 2.2 of Barua et al. (2026):
C_j = {text_j, chunk_id_j, pdf_range_j, printed_range_j}
"""

from typing import Optional, Union
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a discrete chunk of text derived from one or more pages.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        document_id: Identifier of the parent document.
        text: Normalized chunk text.
        page_start: 1-indexed starting PDF page.
        page_end: 1-indexed ending PDF page.
        printed_page_start: Starting human-visible page number if available.
        printed_page_end: Ending human-visible page number if available.
    """
    chunk_id: str = Field(..., description="Unique ID for the chunk (e.g. doc_p1_c0)")
    document_id: str = Field(..., description="Parent document identifier")
    text: str = Field(..., description="Text content of the chunk")
    page_start: int = Field(..., ge=1, description="1-indexed starting PDF page number")
    page_end: int = Field(..., ge=1, description="1-indexed ending PDF page number")
    printed_page_start: Optional[Union[int, str]] = Field(
        default=None,
        description="Starting human-visible page number",
    )
    printed_page_end: Optional[Union[int, str]] = Field(
        default=None,
        description="Ending human-visible page number",
    )

    def get_page_display(self) -> str:
        """Format the page range for citations and display."""
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"
