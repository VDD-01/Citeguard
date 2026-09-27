"""
PDF Parser Module for CiteGuard Project

This module handles parsing and extracting text from PDF documents.
"""

import PyPDF2
from typing import List, Dict, Optional


class PDFParser:
    """Class to parse PDF documents and extract text content."""
    
    def __init__(self):
        self.metadata = {}
        
    def parse_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Parse a PDF file and extract its content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                # Extract metadata
                self.metadata = {
                    'title': reader.metadata.get('/Title', ''),
                    'author': reader.metadata.get('/Author', ''),
                    'subject': reader.metadata.get('/Subject', ''),
                    'creator': reader.metadata.get('/Creator', ''),
                    'producer': reader.metadata.get('/Producer', ''),
                    'creation_date': reader.metadata.get('/CreationDate', ''),
                }
                
                # Extract text from all pages
                text_content = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                        
                return {
                    'text': '\n\n'.join(text_content),
                    'metadata': self.metadata,
                    'num_pages': len(reader.pages)
                }
        except Exception as e:
            raise ValueError(f"Error parsing PDF: {str(e)}")
    
    def get_metadata(self) -> Dict[str, any]:
        """Return the extracted metadata."""
        return self.metadata


def parse_pdf_file(pdf_path: str) -> Dict[str, any]:
    """
    Convenience function to parse a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing extracted text and metadata
    """
    parser = PDFParser()
    return parser.parse_pdf(pdf_path)
