"""Evidence deduplication and sufficiency checking module for CiteGuard-RAG.

Filters near-duplicate chunks resulting from overlapping chunk boundaries,
selects the top generation context chunks (up to 4 chunks), and performs
a preliminary heuristic check to determine whether retrieved evidence is sufficient
to attempt answer generation.

Matches Section 2.4 and Section 2.5 of Barua et al. (2026).
"""

from typing import List, Optional, Set, Tuple
import re

from src.citeguard.models.retrieval import RetrievedChunk
from src.citeguard.utils.logging import setup_logger

logger = setup_logger(__name__)

DEFAULT_REFUSAL = (
    "The document does not contain sufficient information to answer this question."
)


class EvidenceDeduplicator:
    """Deduplicates retrieved chunks to enhance context diversity and selects top-k generation evidence."""

    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize the deduplicator.

        Args:
            similarity_threshold: Word-token Jaccard similarity threshold above which
                two chunks are considered near-duplicates (default 0.85).
        """
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Convert text into a set of normalized word tokens."""
        return set(re.findall(r"\b\w+\b", text.lower()))

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard token similarity between two text snippets.

        Jaccard(A, B) = |A ∩ B| / |A ∪ B|

        Args:
            text1: First text snippet.
            text2: Second text snippet.

        Returns:
            float: Jaccard similarity in [0.0, 1.0].
        """
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        return float(intersection / union) if union > 0 else 0.0

    def deduplicate(
        self,
        candidates: List[RetrievedChunk],
        max_chunks: int = 4,
    ) -> List[RetrievedChunk]:
        """Deduplicate candidates while preserving the highest-scoring chunk and selecting top-k.

        Candidates are expected to be ordered by hybrid_score descending.
        Redundant chunks that have Jaccard similarity >= similarity_threshold
        with an already-selected chunk are discarded.

        Args:
            candidates: Ranked list of RetrievedChunk objects.
            max_chunks: Maximum number of generation context chunks to return (paper default: 4).

        Returns:
            List[RetrievedChunk]: Deduplicated generation evidence.
        """
        if not candidates:
            return []

        selected: List[RetrievedChunk] = []

        for cand in candidates:
            if len(selected) >= max_chunks:
                break

            # Check if this candidate is too similar to any previously selected chunk
            is_duplicate = False
            for kept in selected:
                sim = self.compute_similarity(cand.chunk.text, kept.chunk.text)
                if sim >= self.similarity_threshold:
                    logger.debug(
                        f"Deduplicating chunk '{cand.chunk.chunk_id}' (Jaccard similarity {sim:.3f} "
                        f"with higher-scoring '{kept.chunk.chunk_id}')."
                    )
                    is_duplicate = True
                    break

            if not is_duplicate:
                selected.append(cand)

        logger.info(
            f"Deduplicated {len(candidates)} retrieval candidates down to "
            f"{len(selected)} generation context chunks."
        )
        return selected


def check_evidence_sufficiency(
    evidence: List[RetrievedChunk],
    minimum_score: float = 0.15,
    refusal_message: str = DEFAULT_REFUSAL,
) -> Tuple[bool, str]:
    """Heuristic check to determine if retrieved evidence is sufficient to answer the query.

    The system must not generate an answer if retrieval failed to surface relevant text.
    If evidence is insufficient, CiteGuard-RAG triggers an early standardized refusal.

    Args:
        evidence: List of selected generation chunks.
        minimum_score: Minimum hybrid score required for the top evidence chunk.
        refusal_message: Standardized refusal string.

    Returns:
        Tuple[bool, str]: (is_sufficient, refusal_reason_or_empty)
    """
    if not evidence:
        logger.warning("Evidence sufficiency check failed: zero chunks retrieved.")
        return False, refusal_message

    top_score = max(c.hybrid_score for c in evidence)
    if top_score < minimum_score:
        logger.warning(
            f"Evidence sufficiency check failed: top score {top_score:.4f} is below "
            f"threshold {minimum_score:.4f}."
        )
        return False, refusal_message

    logger.debug(f"Evidence sufficiency check passed (top score: {top_score:.4f} >= {minimum_score:.4f}).")
    return True, ""
