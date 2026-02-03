"""PDF processing utilities."""
import os
import streamlit as st
from PyPDF2 import PdfReader
from typing import Optional
import io


def extract_text_from_pdf_with_llamaparse(
    uploaded_file,
    api_key: Optional[str] = None,
    tier: str = "agentic_plus",
) -> Optional[str]:
    """
    Extract text from PDF using LlamaParse (LlamaCloud) for layout-aware parsing.
    Best for documents with tables, multi-column layout, and complex structure.
    Requires LLAMA_CLOUD_API_KEY to be set (or pass api_key).

    Args:
        uploaded_file: Streamlit uploaded file object or file-like with .read() and .name
        api_key: LlamaCloud API key; if None, uses env LLAMA_CLOUD_API_KEY
        tier: Parsing tier - "cost_effective", "agentic", "agentic_plus", or "fast"

    Returns:
        Extracted markdown/text as string, or None if error or API key missing.
    """
    key = api_key or os.getenv("LLAMA_CLOUD_API_KEY", "").strip()
    if not key:
        return None

    try:
        from llama_cloud import LlamaCloud
    except ImportError:
        return None

    pdf_bytes = uploaded_file.read()
    if not pdf_bytes:
        return None
    filename = getattr(uploaded_file, "name", "document.pdf") or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        filename = "document.pdf"

    try:
        client = LlamaCloud(api_key=key, timeout=300.0)
        # File upload: API accepts (filename, content, media_type)
        file_obj = client.files.create(
            file=(filename, pdf_bytes, "application/pdf"),
            purpose="parse",
        )
        # Parse and wait for result (blocking)
        result = client.parsing.parse(
            file_id=file_obj.id,
            tier=tier,
            version="latest",
            expand=["markdown", "text"],
        )
    except Exception:
        return None

    # Build full document from pages (prefer markdown for structure)
    parts = []
    if hasattr(result, "markdown") and result.markdown and hasattr(result.markdown, "pages"):
        for p in result.markdown.pages:
            if hasattr(p, "markdown") and p.markdown:
                parts.append(p.markdown)
    if not parts and hasattr(result, "text") and result.text and hasattr(result.text, "pages"):
        for p in result.text.pages:
            if hasattr(p, "text") and p.text:
                parts.append(p.text)
    if not parts:
        return None
    return "\n\n".join(parts)


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

