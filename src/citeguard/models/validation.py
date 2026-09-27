"""Data models for sentence-level and answer-level grounding validation.

Matches Section 2.6 and Algorithm 1 of Barua et al. (2026):
A sentence passes grounding if: citation_valid AND (lexical_valid OR semantic_valid).
A response is flagged as hallucinated if it contains one or more unsupported substantive claims.
"""

from typing import List
from pydantic import BaseModel, Field
from src.citeguard.models.answer import AnswerSentence


class SentenceValidationResult(BaseModel):
    """Validation report for an individual sentence against cited evidence.

    Attributes:
        sentence: The evaluated sentence text.
        citation_valid: Whether all cited source IDs exist in the retrieved evidence set.
        lexical_score: Token overlap score |T_s ∩ T_e| / |T_s| in [0.0, 1.0].
        lexical_valid: Whether lexical_score >= lexical_threshold (default 0.10).
        semantic_score: Cosine similarity between sentence and cited evidence embeddings.
        semantic_valid: Whether semantic_score >= semantic_threshold (default 0.40).
        evidence_supported: citation_valid AND (lexical_valid OR semantic_valid).
        reason: Human-readable diagnostic explaining why validation passed or failed.
    """
    sentence: str = Field(..., description="Sentence text evaluated")
    citation_valid: bool = Field(..., description="Whether citations exist in retrieved evidence")
    lexical_score: float = Field(default=0.0, description="Token overlap ratio")
    lexical_valid: bool = Field(default=False, description="Passed lexical threshold (0.10)")
    semantic_score: float = Field(default=0.0, description="Cosine similarity score")
    semantic_valid: bool = Field(default=False, description="Passed semantic threshold (0.40)")
    evidence_supported: bool = Field(..., description="citation_valid AND (lexical_valid OR semantic_valid)")
    reason: str = Field(default="", description="Diagnostic message")


class AnswerValidationResult(BaseModel):
    """Aggregate validation report for the complete candidate answer.

    Attributes:
        total_sentences: Total number of evaluated sentences.
        supported_sentences: Count of sentences where evidence_supported is True.
        unsupported_sentences: Count of sentences where evidence_supported is False.
        supported_sentence_ratio: (total - unsupported) / total.
        all_citations_valid: True if every citation in every sentence is valid.
        validation_passed: True if candidate meets acceptance criteria.
        hallucination_detected: True if any substantive sentence is unsupported (u > 0).
        sentence_results: Detailed per-sentence validation records.
        feedback: Structured feedback string passed to generator if regeneration is required.
    """
    total_sentences: int = Field(default=0, description="Total sentences analyzed")
    supported_sentences: int = Field(default=0, description="Number of fully grounded sentences")
    unsupported_sentences: int = Field(default=0, description="Number of unsupported sentences")
    supported_sentence_ratio: float = Field(default=0.0, description="(n - u) / n")
    all_citations_valid: bool = Field(default=True, description="Whether all citations are valid")
    validation_passed: bool = Field(default=False, description="Overall validation acceptance")
    hallucination_detected: bool = Field(default=False, description="Detected ungrounded claim")
    sentence_results: List[SentenceValidationResult] = Field(default_factory=list)
    feedback: str = Field(default="", description="Actionable correction feedback")
