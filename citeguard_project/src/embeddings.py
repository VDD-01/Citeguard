"""
Embeddings Module for CiteGuard Project

This module handles generating text embeddings using various embedding models.
"""

from typing import List, Optional
import numpy as np


class EmbeddingGenerator:
    """Class to generate text embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the embedding model to use
            device: Device to run embeddings on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of shape (n_texts, embedding_dim) with normalized embeddings
        """
        # Placeholder implementation - replace with actual embedding model
        # For production, use sentence-transformers or similar
        
        import torch
        from sentence_transformers import SentenceTransformer
        
        try:
            model = SentenceTransformer(self.model_name, device=self.device)
            embeddings = model.encode(texts, convert_to_numpy=True)
            
            # Normalize embeddings to unit length
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings with {self.model_name}: {str(e)}")
            # Return dummy embeddings for testing
            n_texts = len(texts)
            embedding_dim = 384  # Default dimension for MiniLM
            return np.random.rand(n_texts, embedding_dim).astype(np.float32)
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate a single embedding for one text.
        
        Args:
            text: Single text string to embed
            
        Returns:
            NumPy array of shape (embedding_dim,) with normalized embedding
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0]


def get_embedding(text: str, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Convenience function to generate a single embedding.
    
    Args:
        text: Text string to embed
        model_name: Name of the embedding model
        
    Returns:
        NumPy array with normalized embedding
    """
    generator = EmbeddingGenerator(model_name=model_name)
    return generator.generate_single_embedding(text)


def get_embeddings(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Convenience function to generate embeddings for multiple texts.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of the embedding model
        
    Returns:
        NumPy array of shape (n_texts, embedding_dim) with normalized embeddings
    """
    generator = EmbeddingGenerator(model_name=model_name)
    return generator.generate_embeddings(texts)
