"""AI processing functions for requirement analysis."""
import streamlit as st
import json
import traceback
from typing import List, Dict, Any, Tuple, Optional
from utils.logging import log
from utils.rag_operations import get_merged_chunks


def process_breakdown(requirement: str, prompt_template: str) -> str:
    """Process a requirement breakdown using AI. Returns raw text response."""
    # Format the prompt - support both {{query}} and {{requirement_text}} placeholders
    prompt = prompt_template.replace("{{query}}", requirement)
    prompt = prompt.replace("{{requirement_text}}", requirement)
    
    # Call AI
    try:
        response = st.session_state.ai_client.client.chat.completions.create(
            model=st.session_state.ai_client.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Senior Contract Strategist and Requirements Engineer. Always return valid JSON when requested."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing breakdown: {str(e)}"


def extract_requirements_from_breakdown(breakdown_result: str) -> List[str]:
    """
    Extract individual requirement statements from breakdown JSON result.
    
    Args:
        breakdown_result: JSON string from breakdown processing
        
    Returns:
        List of compliance verification statements (requirement strings)
    """
    try:
        breakdown_data = json.loads(breakdown_result)
        
        # Extract requirements from the JSON structure
        requirements = []
        if "requirements" in breakdown_data:
            for req in breakdown_data["requirements"]:
                # Use compliance_verification_statement if available, otherwise construct from other fields
                if isinstance(req, dict):
                    if "compliance_verification_statement" in req:
                        requirements.append(req["compliance_verification_statement"])
                    elif "detailed_requirement" in req:
                        requirements.append(req["detailed_requirement"])
                    elif "specific_action" in req:
                        # Construct from available fields
                        parts = []
                        if "responsible_entity" in req:
                            parts.append(req["responsible_entity"])
                        if "specific_action" in req:
                            parts.append(req["specific_action"])
                        if "detailed_requirement" in req:
                            parts.append(req["detailed_requirement"])
                        if parts:
                            requirements.append(" ".join(parts))
                elif isinstance(req, str):
                    requirements.append(req)
        
        # If no requirements found, try to extract from other possible structures
        if not requirements:
            # Try alternative structure
            for key, value in breakdown_data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "compliance_verification_statement" in item:
                            requirements.append(item["compliance_verification_statement"])
                        elif isinstance(item, str):
                            requirements.append(item)
        
        return requirements if requirements else [breakdown_result]  # Fallback to original if parsing fails
    except json.JSONDecodeError:
        # If not JSON, try to extract requirements from text
        # Look for numbered lists or bullet points
        lines = breakdown_result.split('\n')
        requirements = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and markdown formatting
            if line and not line.startswith('#') and not line.startswith('*') and not line.startswith('-'):
                # Remove common prefixes
                for prefix in ['REQ-', 'Requirement', 'Req', '1.', '2.', '3.', '4.', '5.']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                if line:
                    requirements.append(line)
        
        return requirements if requirements else [breakdown_result]
    except Exception as e:
        log("C", "ai_processing.py", "Error extracting requirements from breakdown", {"error": str(e)})
        return [breakdown_result]  # Fallback to original text


def process_requirement(
    requirement: str,
    project_id: str,
    prompt_template: str,
    top_k: int = 5,
    search_method: str = "semantic"
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Process a single requirement using RAG and AI. Returns (result, chunks_metadata).
    
    Args:
        requirement: Requirement text to verify
        project_id: Project ID to filter chunks
        prompt_template: Prompt template to use
        top_k: Number of chunks to retrieve
        search_method: "semantic" or "hybrid" (default: "semantic")
    """
    # Get chunks using RAG
    document_chunks_text, chunks_metadata = get_merged_chunks(
        requirement, 
        project_id, 
        top_k,
        search_method=search_method
    )
    return process_requirement_with_chunks(requirement, document_chunks_text, chunks_metadata, prompt_template)


def process_requirement_with_chunks(
    requirement: str,
    document_chunks_text: str,
    chunks_metadata: List[Dict[str, Any]],
    prompt_template: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Helper to process requirement with pre-fetched chunks."""
    if not document_chunks_text:
        return {
            "status": "NOT_FULFILLED",
            "relevance_score": 0,
            "justification": "No relevant documents found to verify this requirement.",
            "citations": []
        }, []
    
    # Format the prompt
    prompt = prompt_template.replace("{{requirement_text}}", requirement)
    prompt = prompt.replace("{{documentChunkTexts}}", document_chunks_text)
    
    # Call AI
    try:
        response = st.session_state.ai_client.client.chat.completions.create(
            model=st.session_state.ai_client.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Construction Compliance Auditor. Always return valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        # #region agent log
        response_content = response.choices[0].message.content
        log("C", "ai_processing.py", "AI response received", {"content_length": len(response_content), "content_preview": response_content[:200]})
        # #endregion
        
        # #region agent log
        try:
            result = json.loads(response_content)
            log("C", "ai_processing.py", "JSON parsing successful", {"result_keys": list(result.keys()) if isinstance(result, dict) else None})
        except json.JSONDecodeError as e:
            log("C", "ai_processing.py", "JSON parsing failed", {"error": str(e), "content_preview": response_content[:500]})
            raise
        # #endregion
        
        return result, chunks_metadata
    except Exception as e:
        # #region agent log
        log("C", "ai_processing.py", "Exception in process_requirement_with_chunks", {"error": str(e), "traceback": traceback.format_exc()})
        # #endregion
        st.error(f"Error processing requirement: {str(e)}")
        return {
            "status": "ERROR",
            "relevance_score": 0,
            "justification": f"Error: {str(e)}",
            "citations": []
        }, chunks_metadata


def process_multiple_requirements(
    requirements: List[str],
    project_id: str,
    prompt_template: str,
    top_k: int = 5,
    search_method: str = "semantic"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process multiple requirements and combine results.
    
    Args:
        requirements: List of requirement strings to verify
        project_id: Project ID to filter chunks
        prompt_template: Prompt template to use
        top_k: Number of chunks to retrieve per requirement
        search_method: "semantic" or "hybrid" (default: "semantic")
    
    Returns:
        Tuple of (combined_result, all_chunks_metadata)
    """
    if not requirements:
        return {
            "status": "NOT_FULFILLED",
            "relevance_score": 0,
            "justification": "No requirements provided.",
            "citations": []
        }, []
    
    # Format requirements for the prompt
    requirements_text = "\n".join([f"{i+1}. {req}" for i, req in enumerate(requirements)])
    
    # Get chunks using the first requirement (or combine queries)
    # For multiple requirements, we'll use the first one for initial search
    # The prompt will handle multiple requirements
    combined_query = " ".join(requirements[:3])  # Use first 3 for search
    document_chunks_text, chunks_metadata = get_merged_chunks(
        combined_query,
        project_id,
        top_k * 2,  # Get more chunks for multiple requirements
        search_method=search_method
    )
    
    # Process with combined requirements
    return process_requirement_with_chunks(requirements_text, document_chunks_text, chunks_metadata, prompt_template)

