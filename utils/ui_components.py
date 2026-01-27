"""UI components for displaying results and metadata."""
import streamlit as st
import pandas as pd
import json
from typing import List, Dict, Any, Optional
from utils.logging import log


def enrich_citation_with_doc_info(citation: Dict[str, Any], chunks_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Enrich citation with document name by matching source text to chunks."""
    log("F", "ui_components.py", "enrich_citation_with_doc_info called", {
        "chunks_metadata_type": type(chunks_metadata).__name__, 
        "chunks_metadata_is_none": chunks_metadata is None, 
        "chunks_metadata_len": len(chunks_metadata) if chunks_metadata else 0
    })
    
    if chunks_metadata is None:
        log("F", "ui_components.py", "chunks_metadata is None, using empty list", {})
        chunks_metadata = []
    
    source_text = citation.get("source_text", "").strip()
    if not source_text:
        return citation
        
    source_text_lower = source_text.lower()
    doc_ref = citation.get("document_reference", "")
    
    best_match = None
    
    # Try to find the chunk that contains the source text
    for chunk in chunks_metadata:
        chunk_content = chunk.get('content', '').lower()
        # If verbatim in chunk, it's a strong match
        if source_text_lower in chunk_content or (len(source_text_lower) > 20 and source_text_lower[:20] in chunk_content):
            best_match = chunk
            break
    
    if best_match:
        file_name = best_match.get('file_name') or best_match.get('original_filename')
        page_number = best_match.get('page_number')
        if file_name:
            # Enhance document reference with actual file name
            doc_ref = f"{file_name}" + (f" (page {page_number})" if page_number else "")
    
    citation_copy = citation.copy()
    citation_copy["document_name"] = doc_ref
    return citation_copy


def display_chunks_metadata(chunks_metadata: List[Dict[str, Any]], search_method: str = "semantic"):
    """Display chunk metadata in a nice table format."""
    if not chunks_metadata:
        st.info("No chunks retrieved.")
        return
    
    st.subheader("📚 Retrieved Chunks")
    
    # Create a DataFrame for display
    display_data = []
    for i, chunk in enumerate(chunks_metadata, 1):
        file_name = chunk.get('file_name') or chunk.get('original_filename') or 'Unknown'
        page_num = chunk.get('page_number')
        page_display = page_num if page_num is not None else 'N/A'
        
        row_data = {
            'Rank': i,
            'Document': file_name,
            'Page': page_display,
        }
        
        # Add search-specific metrics
        if search_method == "hybrid":
            rrf_score = chunk.get('rrf_score')
            vector_rank = chunk.get('vector_rank')
            text_rank = chunk.get('text_rank')
            
            row_data['RRF Score'] = f"{rrf_score:.4f}" if rrf_score is not None else 'N/A'
            row_data['Vector Rank'] = vector_rank if vector_rank is not None else 'N/A'
            row_data['Text Rank'] = text_rank if text_rank is not None else 'N/A'
        else:
            similarity = chunk.get('similarity')
            row_data['Similarity'] = f"{similarity:.4f}" if similarity is not None else 'N/A'
        
        display_data.append(row_data)
    
    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Chunks", len(chunks_metadata))
    with col2:
        unique_files = len(set(chunk.get('file_name') or chunk.get('original_filename') or 'Unknown' for chunk in chunks_metadata))
        st.metric("Unique Documents", unique_files)
    with col3:
        if search_method == "hybrid":
            avg_rrf = sum(chunk.get('rrf_score', 0) or 0 for chunk in chunks_metadata) / len(chunks_metadata) if chunks_metadata else 0
            st.metric("Avg RRF Score", f"{avg_rrf:.4f}")
        else:
            avg_sim = sum(chunk.get('similarity', 0) or 0 for chunk in chunks_metadata) / len(chunks_metadata) if chunks_metadata else 0
            st.metric("Avg Similarity", f"{avg_sim:.4f}")


def display_result(result: Dict[str, Any], chunks_metadata: List[Dict[str, Any]] = None, key_suffix: str = ""):
    """Display a single requirement result."""
    # Status badge
    status = result.get("status", "UNKNOWN")
    if status == "FULFILLED":
        st.success(f"✅ Status: {status}")
    elif status == "PARTIALLY_FULFILLED":
        st.warning(f"⚠️ Status: {status}")
    elif status == "NOT_FULFILLED":
        st.error(f"❌ Status: {status}")
    else:
        st.info(f"Status: {status}")
    
    # Relevance Score
    raw_score = result.get("relevance_score", 0)
    log("D", "ui_components.py", "Processing relevance_score", {"raw_score": raw_score, "type": type(raw_score).__name__})
    
    try:
        score = int(raw_score) if raw_score is not None else 0
        log("D", "ui_components.py", "relevance_score converted successfully", {"score": score})
    except (ValueError, TypeError) as e:
        log("D", "ui_components.py", "relevance_score conversion failed", {"error": str(e), "raw_score": raw_score})
        score = 0
    
    st.metric("Relevance Score", f"{score}/10")
    
    # Justification
    st.subheader("Justification")
    st.write(result.get("justification", "No justification provided."))
    
    # Citations
    citations = result.get("citations", [])
    if citations:
        st.subheader("Citations")
        for i, citation in enumerate(citations, 1):
            # Enrich citation with document info if chunks metadata available
            if chunks_metadata:
                citation = enrich_citation_with_doc_info(citation, chunks_metadata)
            
            doc_name = citation.get("document_name") or citation.get("document_reference", "N/A")
            
            with st.expander(f"Citation {i}: {doc_name}"):
                st.write("**Document:**")
                st.info(doc_name)
                st.write("**Source Text:**")
                st.code(citation.get("source_text", ""))
                if citation.get("document_reference") and citation.get("document_reference") != doc_name:
                    st.write("**Original Reference:**")
                    st.caption(citation.get("document_reference", "N/A"))
                
                # Find matching chunk and show full content option
                if chunks_metadata:
                    source_text = citation.get("source_text", "").strip().lower()
                    matching_chunk = None
                    for chunk in chunks_metadata:
                        chunk_content = chunk.get('content', '').lower()
                        # Try to find chunk containing the source text
                        if source_text and (source_text in chunk_content or 
                                          (len(source_text) > 20 and source_text[:20] in chunk_content)):
                            matching_chunk = chunk
                            break
                    
                    if matching_chunk:
                        full_content = matching_chunk.get('content', '')
                        if full_content:
                            with st.expander("📄 View Full Chunk Content", expanded=False):
                                st.text_area(
                                    "Full chunk content:",
                                    value=full_content,
                                    height=300,
                                    disabled=True,
                                    key=f"full_chunk_{i}_{key_suffix}"
                                )
    
    # JSON Output
    st.markdown("---")
    with st.expander("📄 View Raw JSON Output", expanded=False):
        # Create enriched result with document names in citations
        enriched_result = result.copy()
        if chunks_metadata and citations:
            enriched_citations = []
            for citation in citations:
                enriched_citation = enrich_citation_with_doc_info(citation, chunks_metadata)
                enriched_citations.append(enriched_citation)
            enriched_result["citations"] = enriched_citations
        
        st.json(enriched_result)
        
        # Copy button
        result_json = json.dumps(enriched_result, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=result_json,
            file_name="requirement_result.json",
            mime="application/json",
            key=f"json_download_from_view_{key_suffix}"
        )

