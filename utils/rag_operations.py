"""RAG (Retrieval-Augmented Generation) operations for document search."""
import streamlit as st
from typing import List, Dict, Any, Tuple


def get_merged_chunks(
    query: str, 
    project_id: str, 
    top_k: int = 5,
    search_method: str = "semantic"
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Get merged chunks using RAG search. Returns (merged_text, chunks_metadata).
    
    Args:
        query: Search query string
        project_id: Project ID to filter chunks
        top_k: Number of chunks to retrieve
        search_method: "semantic" or "hybrid" (default: "semantic")
    
    Returns:
        Tuple of (merged_text, chunks_metadata)
    """
    try:
        if search_method == "hybrid":
            # Use hybrid search with RRF
            relevant_chunks = st.session_state.supabase_client.get_chunks_by_hybrid_search(
                project_id=project_id,
                query=query,
                match_count=top_k
            )
        else:
            # Use semantic (vector-only) search
            relevant_chunks = st.session_state.supabase_client.search_chunks_by_project_id(
                project_id=project_id,
                query=query,
                top_k=top_k
            )
        
        # Merge with file names and page numbers: "chunk-1 (file: doc.pdf, page: 5): <content>..."
        merged_parts = []
        for i, chunk in enumerate(relevant_chunks):
            file_name = chunk.get('file_name') or chunk.get('original_filename') or 'Unknown'
            page_num = chunk.get('page_number')
            page_info = f", page: {page_num}" if page_num else ""
            chunk_header = f"chunk-{i+1} (file: {file_name}{page_info})"
            merged_parts.append(f"{chunk_header}: {chunk.get('content', '')}")
        
        merged = "\n".join(merged_parts)
        return merged, relevant_chunks
    except Exception as e:
        st.error(f"Error fetching chunks for query '{query}': {str(e)}")
        return "", []

