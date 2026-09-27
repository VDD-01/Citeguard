"""Paragraph-aware document chunking module for CiteGuard-RAG.

Segments parsed DocumentPage objects into discrete DocumentChunk units using
a paragraph-first, sentence-fallback strategy that respects token budgets
and preserves page provenance.

Matches Section 2.2 of Barua et al. (2026):
C_j = {text_j, chunk_id_j, pdf_range_j, printed_range_j}
Default: chunk_size = 600 tokens, chunk_overlap = 120 tokens.
"""

from typing import List, Optional, Tuple
import re

from src.citeguard.config import ChunkingConfig, load_config
from src.citeguard.models.chunk import DocumentChunk
from src.citeguard.models.document import DocumentPage
from src.citeguard.utils.logging import setup_logger
from src.citeguard.utils.text import estimate_tokens, split_into_sentences, normalize_text

logger = setup_logger(__name__)


class DocumentChunker:
    """Paragraph-aware chunker that preserves document provenance and token budgets."""

    def __init__(
        self,
        chunk_size: int = 600,
        overlap: int = 120,
        min_chunk_length: int = 50,
    ):
        """Initialize the document chunker.

        Args:
            chunk_size: Target maximum token budget per chunk (default 600).
            overlap: Target token overlap between adjacent chunks (default 120).
            min_chunk_length: Minimum character length to filter out noise fragments.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_length = min_chunk_length

    def _split_into_atomic_units(self, text: str) -> List[str]:
        """Split text into paragraphs, falling back to sentences if paragraphs are oversized.

        Args:
            text: Raw or normalized text.

        Returns:
            List[str]: Units of text (paragraphs or sentences) small enough to fit within budget.
        """
        raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        units: List[str] = []

        for p in raw_paragraphs:
            p_tokens = estimate_tokens(p)
            if p_tokens <= self.chunk_size:
                units.append(p)
            else:
                # Fallback to sentence-level decomposition for oversized paragraphs
                sentences = split_into_sentences(p)
                for s in sentences:
                    s_tokens = estimate_tokens(s)
                    if s_tokens <= self.chunk_size:
                        units.append(s)
                    else:
                        # Fallback for an exceptionally long sentence with no punctuation:
                        # split by word groups
                        words = s.split()
                        words_per_batch = int(self.chunk_size / 1.33)
                        for i in range(0, len(words), words_per_batch):
                            sub_text = " ".join(words[i : i + words_per_batch])
                            if sub_text:
                                units.append(sub_text)

        return units

    def chunk_pages(
        self,
        pages: List[DocumentPage],
    ) -> List[DocumentChunk]:
        """Convert a list of DocumentPage objects into a list of DocumentChunk objects.

        Args:
            pages: List of DocumentPage objects from the PDF parser.

        Returns:
            List[DocumentChunk]: Segmented chunks with retained page ranges and provenance.
        """
        if not pages:
            return []

        doc_id = pages[0].document_id

        # Collect atomic text units across pages with their associated page metadata
        # Each entry is (unit_text, pdf_page_number, printed_page_number)
        indexed_units: List[Tuple[str, int, Optional[int | str]]] = []

        for page in pages:
            if not page.text or not page.text.strip():
                continue

            page_units = self._split_into_atomic_units(page.text)
            for u in page_units:
                indexed_units.append((u, page.page_number, page.printed_page_number))

        if not indexed_units:
            logger.warning(f"No usable text units found across {len(pages)} pages of '{doc_id}'.")
            return []

        chunks: List[DocumentChunk] = []
        current_units: List[Tuple[str, int, Optional[int | str]]] = []
        current_tokens = 0
        chunk_idx = 0

        i = 0
        while i < len(indexed_units):
            unit_text, pdf_page, printed_page = indexed_units[i]
            unit_tokens = estimate_tokens(unit_text)

            # Check if adding this unit exceeds the chunk size budget
            if current_tokens + unit_tokens > self.chunk_size and current_units:
                # Finalize and emit current chunk
                chunk = self._build_chunk(
                    doc_id=doc_id,
                    chunk_idx=chunk_idx,
                    units=current_units,
                )
                if chunk is not None:
                    chunks.append(chunk)
                    chunk_idx += 1

                # Calculate overlap: retain units from the tail of current_units up to target overlap
                overlap_units: List[Tuple[str, int, Optional[int | str]]] = []
                accumulated_overlap = 0
                for u_text, p_num, pr_num in reversed(current_units):
                    u_tok = estimate_tokens(u_text)
                    if accumulated_overlap + u_tok <= self.overlap:
                        overlap_units.insert(0, (u_text, p_num, pr_num))
                        accumulated_overlap += u_tok
                    else:
                        break

                current_units = overlap_units
                current_tokens = accumulated_overlap

            # Add unit to current accumulator
            current_units.append((unit_text, pdf_page, printed_page))
            current_tokens += unit_tokens
            i += 1

        # Emit the final remaining chunk if any
        if current_units:
            chunk = self._build_chunk(
                doc_id=doc_id,
                chunk_idx=chunk_idx,
                units=current_units,
            )
            if chunk is not None:
                chunks.append(chunk)

        logger.info(f"Chunked document '{doc_id}' into {len(chunks)} chunks.")
        return chunks

    def _build_chunk(
        self,
        doc_id: str,
        chunk_idx: int,
        units: List[Tuple[str, int, Optional[int | str]]],
    ) -> Optional[DocumentChunk]:
        """Construct a DocumentChunk from accumulated units, filtering out noise.

        Args:
            doc_id: Parent document identifier.
            chunk_idx: Zero-based sequence number for the chunk.
            units: List of (unit_text, pdf_page, printed_page) tuples.

        Returns:
            Optional[DocumentChunk]: The created chunk, or None if filtered out.
        """
        if not units:
            return None

        text = "\n\n".join(u[0] for u in units).strip()

        # Filter out low-information fragments (navigation markers, standalone headers, etc.)
        if len(text) < self.min_chunk_length:
            logger.debug(f"Filtering out short chunk ({len(text)} chars): '{text[:30]}...'")
            return None

        pdf_pages = [u[1] for u in units]
        page_start = min(pdf_pages)
        page_end = max(pdf_pages)

        printed_pages = [u[2] for u in units if u[2] is not None]
        printed_start = printed_pages[0] if printed_pages else None
        printed_end = printed_pages[-1] if printed_pages else None

        chunk_id = f"{doc_id}_c{chunk_idx:03d}"

        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            text=text,
            page_start=page_start,
            page_end=page_end,
            printed_page_start=printed_start,
            printed_page_end=printed_end,
        )


def chunk_document(
    pages: List[DocumentPage],
    config: Optional[ChunkingConfig] = None,
) -> List[DocumentChunk]:
    """Convenience function to chunk a list of DocumentPage objects.

    Args:
        pages: List of DocumentPage objects.
        config: Optional ChunkingConfig. If None, loads from config.yaml.

    Returns:
        List[DocumentChunk]: Generated chunks.
    """
    if config is None:
        app_cfg = load_config()
        config = app_cfg.chunking

    chunker = DocumentChunker(
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        min_chunk_length=config.min_chunk_length,
    )
    return chunker.chunk_pages(pages)
