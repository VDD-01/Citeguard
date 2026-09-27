"""Text processing utilities for CiteGuard-RAG.

Handles unicode normalization, whitespace cleaning, token estimation,
and sentence segmentation.
"""

import re
import unicodedata
from typing import List


def normalize_text(text: str) -> str:
    """Normalize text by converting unicode characters and collapsing whitespace.

    Args:
        text: Raw input string.

    Returns:
        Cleaned, normalized string.
    """
    if not text:
        return ""
    # Normalize unicode (NFKC replaces compatibility characters with standard ones)
    text = unicodedata.normalize("NFKC", text)
    # Replace non-breaking spaces and exotic whitespace with standard space
    text = re.sub(r"[\r\t\f\v ]+", " ", text)
    # Replace multiple newlines with at most two newlines (paragraph boundary)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text chunk.

    In English, 1 token is approximately 4 characters or ~0.75 words.
    This estimator uses word count * 1.33 as a fast, robust approximation
    consistent across platforms.

    Args:
        text: Text to estimate.

    Returns:
        Estimated integer token count.
    """
    if not text:
        return 0
    words = text.split()
    return max(1, int(len(words) * 1.33))


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving citations and abbreviations.

    Args:
        text: Input paragraph or answer text.

    Returns:
        List of non-empty sentence strings.
    """
    if not text:
        return []

    # Clean leading/trailing whitespace
    text = text.strip()

    # Split on period, exclamation, or question mark followed by whitespace and capital letter
    # or followed by an inline citation e.g. "claim [Source 1]. Next claim."
    pattern = r"(?<=[.!?])\s+(?=[A-Z0-9\[])"
    raw_sentences = re.split(pattern, text)

    sentences = []
    for s in raw_sentences:
        cleaned = s.strip()
        if cleaned:
            sentences.append(cleaned)

    return sentences
