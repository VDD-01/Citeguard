"""
Text Chunker Module for CiteGuard Project

This module handles splitting text into manageable chunks for processing.
"""

from typing import List, Optional


class TextChunker:
    """Class to split text into chunks based on various strategies."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_by_size(self, text: str) -> List[str]:
        """
        Split text into chunks based on character count.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        words = text.split()
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                current_chunk += (word + " ") if current_chunk else word
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks based on sentence boundaries.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks (sentences or groups of sentences)
        """
        import re
        
        # Find all sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        sentence_count = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += (sentence + " ") if current_chunk else sentence
                sentence_count += 1
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                start_idx = max(0, len(current_chunk) - self.overlap)
                current_chunk = current_chunk[start_idx:] + " " + sentence
                sentence_count = 1
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text into chunks based on paragraph boundaries.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks (paragraphs or groups of paragraphs)
        """
        # Split by double newlines (paragraph breaks)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        paragraph_count = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += (para + "\n\n") if current_chunk else para
                paragraph_count += 1
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                start_idx = max(0, len(current_chunk) - self.overlap)
                current_chunk = current_chunk[start_idx:] + "\n\n" + para
                paragraph_count = 1
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks


def chunk_text(text: str, strategy: str = 'size', **kwargs) -> List[str]:
    """
    Convenience function to chunk text using specified strategy.
    
    Args:
        text: The text to chunk
        strategy: Chunking strategy ('size', 'sentences', or 'paragraphs')
        **kwargs: Additional arguments for the chunker
        
    Returns:
        List of text chunks
    """
    chunker = TextChunker(**kwargs)
    
    if strategy == 'size':
        return chunker.chunk_by_size(text)
    elif strategy == 'sentences':
        return chunker.chunk_by_sentences(text)
    elif strategy == 'paragraphs':
        return chunker.chunk_by_paragraphs(text)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
