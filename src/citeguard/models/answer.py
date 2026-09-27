"""Data models for citations, candidate answers, and answer sentences.

Matches Section 2.5 of Barua et al. (2026).
"""

from typing import List
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Represents a citation referencing a source evidence block.

    Attributes:
        source_id: Human-readable source identifier e.g. "Source 1" or "source_1".
        chunk_id: Internal chunk ID corresponding to this source.
        document_id: Source document ID.
        page_number: Source page number.
    """
    source_id: str = Field(..., description="Source tag e.g. 'Source 1'")
    chunk_id: str = Field(..., description="Underlying DocumentChunk ID")
    document_id: str = Field(..., description="Source document ID")
    page_number: int = Field(..., description="Source page number")


class AnswerSentence(BaseModel):
    """Represents an individual sentence in a generated answer and its associated citations.

    Attributes:
        text: Sentence text.
        citations: List of source identifiers cited by this sentence (e.g. ['Source 1']).
    """
    text: str = Field(..., description="Content of the sentence")
    citations: List[str] = Field(default_factory=list, description="Extracted citation tags")


class CandidateAnswer(BaseModel):
    """Represents a full generated candidate answer before or after validation.

    Attributes:
        raw_text: Full raw response string from the generator.
        sentences: List of decomposed AnswerSentence objects.
        is_refusal: True if the answer is a standardized abstention.
    """
    raw_text: str = Field(..., description="Full raw answer string")
    sentences: List[AnswerSentence] = Field(default_factory=list, description="Parsed sentences with citations")
    is_refusal: bool = Field(default=False, description="Whether this candidate is an abstention/refusal")
