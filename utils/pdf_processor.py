"""PDF processing utilities."""
import streamlit as st
from PyPDF2 import PdfReader
from typing import Optional
import io


def extract_text_from_pdf(uploaded_file) -> Optional[str]:
    """
    Extract text from an uploaded PDF file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Extracted text as string, or None if error
    """
    try:
        # Read the PDF file
        pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
        
        # Extract text from all pages
        text_parts = []
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num} ---\n{page_text}")
            except Exception as e:
                st.warning(f"Error extracting text from page {page_num}: {str(e)}")
                continue
        
        if not text_parts:
            st.error("No text could be extracted from the PDF.")
            return None
        
        full_text = "\n\n".join(text_parts)
        return full_text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

