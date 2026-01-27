"""Utility functions for tender checking."""
from typing import Optional
from PyPDF2 import PdfReader
import io


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Optional[str]:
    """
    Extract text from PDF bytes.
    
    Args:
        pdf_bytes: PDF file as bytes
        
    Returns:
        Extracted text as string, or None if error
    """
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        
        text_parts = []
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num} ---\n{page_text}")
            except Exception as e:
                print(f"Warning: Error extracting text from page {page_num}: {str(e)}")
                continue
        
        if not text_parts:
            return None
        
        full_text = "\n\n".join(text_parts)
        return full_text
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None


def extract_text_from_pdf_file(file_path: str) -> Optional[str]:
    """
    Extract text from PDF file path.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted text as string, or None if error
    """
    try:
        with open(file_path, 'rb') as f:
            pdf_bytes = f.read()
        return extract_text_from_pdf_bytes(pdf_bytes)
    except Exception as e:
        print(f"Error reading PDF file: {str(e)}")
        return None

