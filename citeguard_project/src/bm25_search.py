"""
BM25 Search Module for CiteGuard Project

This module implements BM25 (Best Match 25) algorithm for text retrieval.
"""

from typing import List, Dict, Tuple
import re


class BM25Index:
    """Class to build and query a BM25 index."""
    
    def __init__(self):
        self.idf_table = {}
        self.doc_freqs = {}
        self.token_counts = {}
        self.num_docs = 0
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of lowercase tokens
        """
        # Convert to lowercase and extract alphanumeric tokens
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return tokens
    
    def build_index(self, documents: List[str], doc_ids: Optional[List[int]] = None):
        """
        Build a BM25 index from a list of documents.
        
        Args:
            documents: List of document texts
            doc_ids: Optional list of document IDs (defaults to 0, 1, 2, ...)
        """
        if doc_ids is None:
            doc_ids = list(range(len(documents)))
            
        self.num_docs = len(documents)
        
        # Calculate document frequencies and token counts
        for i, doc in enumerate(documents):
            tokens = self.tokenize(doc)
            unique_tokens = set(tokens)
            
            for token in unique_tokens:
                if token not in self.doc_freqs:
                    self.doc_freqs[token] = 0
                self.doc_freqs[token] += 1
                
                # Track total token counts per document
                if i not in self.token_counts:
                    self.token_counts[i] = {}
                self.token_counts[i][token] = self.token_counts[i].get(token, 0) + 1
        
        # Calculate IDF values
        for token, freq in self.doc_freqs.items():
            self.idf_table[token] = np.log((self.num_docs - freq + 0.5) / (freq + 0.5))
    
    def query(self, query: str, k1: float = 1.5, b: float = 0.75) -> List[Tuple[int, float]]:
        """
        Query the BM25 index.
        
        Args:
            query: Query text to search for
            k1: BM25 parameter (typically 1.2-2.0)
            b: BM25 parameter (typically 0.75)
            
        Returns:
            List of (doc_id, score) tuples sorted by relevance score descending
        """
        query_tokens = self.tokenize(query)
        
        scores = {}
        
        for token in query_tokens:
            if token not in self.idf_table:
                continue
                
            # Calculate average document length
            avg_len = np.mean([len(self.tokenize(doc)) for doc in documents])
            
            for i, tokens in enumerate(self.token_counts):
                if token not in tokens:
                    continue
                    
                tf = tokens[token]
                idf = self.idf_table[token]
                
                # BM25 scoring formula
                score = (idf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(tokens) / avg_len))
                
                if i not in scores:
                    scores[i] = 0.0
                scores[i] += score
        
        # Sort by score descending and return top results
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(doc_id, score) for doc_id, score in sorted_results]


def bm25_search(documents: List[str], query: str, k1: float = 1.5, b: float = 0.75) -> List[Tuple[int, float]]:
    """
    Convenience function to perform BM25 search.
    
    Args:
        documents: List of document texts to search in
        query: Query text to search for
        k1: BM25 parameter (typically 1.2-2.0)
        b: BM25 parameter (typically 0.75)
        
    Returns:
        List of (doc_index, score) tuples sorted by relevance score descending
    """
    index = BM25Index()
    index.build_index(documents)
    return index.query(query, k1=k1, b=b)
