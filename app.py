import streamlit as st
import pandas as pd
import json
from typing import List, Dict, Any, Optional, Tuple
from clients.supabase_client import SupabaseClient
from clients.ai_client import AIClient
from config.config import TOP_K_CHUNKS, OPENAI_MODEL
import traceback
from datetime import datetime

# #region agent log
LOG_PATH = "/Users/ilyasyeskenov/Desktop/req_check/.cursor/debug.log"
def _log(hypothesis_id: str, location: str, message: str, data: dict = None):
    try:
        log_entry = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": hypothesis_id,
            "timestamp": int(datetime.now().timestamp() * 1000),
            "location": location,
            "message": message,
            "data": data or {}
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except:
        pass
# #endregion

# Default prompt template
DEFAULT_PROMPT = """You are an expert Construction Compliance Auditor and Quality Assurance Specialist. Your role is to rigorously analyze technical documentation to verify if specific project requirements have been met.

**Your Goal:**
Analyze the provided supporting documents to determine if the specific Requirement (provided below) is fulfilled. You must provide a justification based *strictly* on the evidence found in the text.

**Input Context:**
- **Requirement to Verify:** "{{requirement_text}}"
- **Supporting Documents:**
{{documentChunkTexts}}

**Step-by-Step Reasoning Process:**
1.  **Analyze the Requirement:** Break down the specific requirement into its constituent conditions (e.g., specific materials, dimensions, safety standards, certifications, or tolerances).
2.  **Scan for Evidence:** Search the Supporting Documents for exact keywords, synonyms, or technical specifications that match the requirement's conditions.
3.  **Evaluate Completeness:** Determine if the evidence covers the *entirety* of the requirement or only parts of it.
4.  **Formulate Justification:** Construct an argument linking the text in the documents to the requirement conditions.

**Output Instructions:**
Provide your response in a valid JSON format with the following structure:

```json
{
  "status": "FULFILLED" | "PARTIALLY_FULFILLED" | "NOT_FULFILLED",
  "relevance_score": <integer_0_to_10>,
  "justification": "<A detailed explanation of how the document satisfies the requirement. If partially fulfilled, explicitly explain what is missing.>",
  "citations": [
    {
      "source_text": "<The exact verbatim quote from the document used as evidence>",
      "document_reference": "<The name, page number, or ID of the specific document chunk if available>"
    }
    ]
}
```"""

# Default breakdown prompt template
DEFAULT_BREAKDOWN_PROMPT = """Break down the provided requirement into individual sub-requirements if it is complex.
A requirement is considered complex if it involves multiple actions, systems, equipment, or processes.
If the requirement is simple (e.g., involves a single action or system), return the original requirement as is.

Instructions:
Generate up to 5 sub-requirements, ensuring each is distinct and relevant to the original requirement.
Each sub-requirement must be self-contained and include all key systems, equipment, locations, and processes mentioned in the original requirement, even if this causes repetition.
Exclude references to actors (e.g., “the Contractor,” “the operator,” “staff”). Only include systems, equipment, locations, and processes.
Do not reword the original requirement excessively; preserve its terminology and original technical meaning.
If the requirement does not specify systems, equipment, locations, or processes, note this clearly and proceed with the breakdown based on available information.

Example Requirement:
(S320.2B.16.1) (c) All SAMS alarms can be monitored through the MCS workstation. The geographic location of an activated alarm is clearly shown on a station layout display.

Sub-requirements Generated:
All SAMS alarms can be monitored through the MCS workstation.
The geographic location of an activated SAMS alarm is clearly shown on a station layout display.

Example Complex Requirement:
(S450.1A.22.3) The HVAC system must maintain temperature control within ±2°C and log performance data to the BMS, while the control panel displays real-time status updates.

Sub-requirements Generated:
The HVAC system maintains temperature control within ±2°C.
The HVAC system logs performance data to the BMS.
The HVAC system control panel displays real-time status updates of the HVAC system within the BMS environment.

User Query: {{query}}"""


# Initialize session state
# #region agent log
if 'supabase_client' not in st.session_state or not hasattr(st.session_state.supabase_client, 'save_prompt'):
    _log("A", "app.py:74", "Initializing SupabaseClient", {})
    try:
        st.session_state.supabase_client = SupabaseClient()
        _log("A", "app.py:76", "SupabaseClient initialized successfully", {})
    except Exception as e:
        _log("A", "app.py:76", "SupabaseClient initialization failed", {"error": str(e), "traceback": traceback.format_exc()})
        raise
# #endregion

# #region agent log
if 'ai_client' not in st.session_state:
    _log("A", "app.py:79", "Initializing AIClient", {})
    try:
        st.session_state.ai_client = AIClient()
        _log("A", "app.py:81", "AIClient initialized successfully", {})
    except Exception as e:
        _log("A", "app.py:81", "AIClient initialization failed", {"error": str(e), "traceback": traceback.format_exc()})
        raise
# #endregion

if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = DEFAULT_PROMPT

if 'current_breakdown_prompt' not in st.session_state:
    st.session_state.current_breakdown_prompt = DEFAULT_BREAKDOWN_PROMPT

if 'saved_prompts' not in st.session_state:
    # Initialize with default
    st.session_state.saved_prompts = {"Default": DEFAULT_PROMPT}
    # Try to load requirement prompts from Supabase on first run
    try:
        # Load prompts with type "requirement" or no type (for backward compatibility)
        db_requirement_prompts = st.session_state.supabase_client.get_all_prompts(prompt_type="requirement")
        db_all_prompts = st.session_state.supabase_client.get_all_prompts()  # Get all for backward compatibility
        
        # Add requirement prompts
        for p in db_requirement_prompts:
            st.session_state.saved_prompts[p['name']] = p['prompt_text']
        
        # For backward compatibility: add prompts without type that look like requirement prompts
        for p in db_all_prompts:
            prompt_type = p.get('prompt_type')
            if prompt_type is None:  # No type assigned (old prompts)
                # Use heuristic: if it has {{requirement_text}}, it's a requirement prompt
                if "{{requirement_text}}" in p.get('prompt_text', ''):
                    if p['name'] not in st.session_state.saved_prompts:
                        st.session_state.saved_prompts[p['name']] = p['prompt_text']
    except Exception as e:
        st.sidebar.error(f"Error loading prompts: {e}")

if 'saved_breakdown_prompts' not in st.session_state:
    st.session_state.saved_breakdown_prompts = {"Default Breakdown": DEFAULT_BREAKDOWN_PROMPT}
    # Load breakdown prompts from Supabase
    try:
        db_breakdown_prompts = st.session_state.supabase_client.get_all_prompts(prompt_type="breakdown")
        db_all_prompts = st.session_state.supabase_client.get_all_prompts()  # Get all for backward compatibility
        
        # Add breakdown prompts
        for p in db_breakdown_prompts:
            st.session_state.saved_breakdown_prompts[p['name']] = p['prompt_text']
        
        # For backward compatibility: add prompts without type that look like breakdown prompts
        for p in db_all_prompts:
            prompt_type = p.get('prompt_type')
            if prompt_type is None:  # No type assigned (old prompts)
                # Use heuristic: if it has {{query}} but not {{requirement_text}}, it's a breakdown prompt
                prompt_text = p.get('prompt_text', '')
                if "{{query}}" in prompt_text and "{{requirement_text}}" not in prompt_text:
                    if p['name'] not in st.session_state.saved_breakdown_prompts:
                        st.session_state.saved_breakdown_prompts[p['name']] = prompt_text
    except Exception as e:
        st.sidebar.error(f"Error loading breakdown prompts: {e}")
else:
    # Refresh breakdown prompts from Supabase to ensure they're in sync
    try:
        db_breakdown_prompts = st.session_state.supabase_client.get_all_prompts(prompt_type="breakdown")
        for p in db_breakdown_prompts:
            st.session_state.saved_breakdown_prompts[p['name']] = p['prompt_text']
    except Exception as e:
        pass  # Silently fail on refresh to avoid disrupting user experience

# Ensure 'Default' is always there
if "Default" not in st.session_state.saved_prompts:
    st.session_state.saved_prompts["Default"] = DEFAULT_PROMPT
if "Default Breakdown" not in st.session_state.saved_breakdown_prompts:
    st.session_state.saved_breakdown_prompts["Default Breakdown"] = DEFAULT_BREAKDOWN_PROMPT


def get_merged_chunks(query: str, project_id: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    """Get merged chunks using RAG search. Returns (merged_text, chunks_metadata)."""
    try:
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


def process_breakdown(requirement: str, prompt_template: str) -> str:
    """Process a requirement breakdown using AI."""
    # Format the prompt
    prompt = prompt_template.replace("{{query}}", requirement)
    
    # Call AI
    try:
        response = st.session_state.ai_client.client.chat.completions.create(
            model=st.session_state.ai_client.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in breaking down technical requirements."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing breakdown: {str(e)}"


def process_requirement(
    requirement: str,
    project_id: str,
    prompt_template: str,
    top_k: int = 5
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Process a single requirement using RAG and AI. Returns (result, chunks_metadata)."""
    # Get chunks using RAG
    document_chunks_text, chunks_metadata = get_merged_chunks(requirement, project_id, top_k)
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
        _log("C", "app.py:218", "AI response received", {"content_length": len(response_content), "content_preview": response_content[:200]})
        # #endregion
        
        # #region agent log
        try:
            result = json.loads(response_content)
            _log("C", "app.py:220", "JSON parsing successful", {"result_keys": list(result.keys()) if isinstance(result, dict) else None})
        except json.JSONDecodeError as e:
            _log("C", "app.py:220", "JSON parsing failed", {"error": str(e), "content_preview": response_content[:500]})
            raise
        # #endregion
        
        return result, chunks_metadata
    except Exception as e:
        # #region agent log
        _log("C", "app.py:227", "Exception in process_requirement_with_chunks", {"error": str(e), "traceback": traceback.format_exc()})
        # #endregion
        st.error(f"Error processing requirement: {str(e)}")
        return {
            "status": "ERROR",
            "relevance_score": 0,
            "justification": f"Error: {str(e)}",
            "citations": []
        }, chunks_metadata


def enrich_citation_with_doc_info(citation: Dict[str, Any], chunks_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Enrich citation with document name by matching source text to chunks."""
    # #region agent log
    _log("F", "app.py:232", "enrich_citation_with_doc_info called", {"chunks_metadata_type": type(chunks_metadata).__name__, "chunks_metadata_is_none": chunks_metadata is None, "chunks_metadata_len": len(chunks_metadata) if chunks_metadata else 0})
    # #endregion
    
    # #region agent log
    if chunks_metadata is None:
        _log("F", "app.py:235", "chunks_metadata is None, using empty list", {})
        chunks_metadata = []
    # #endregion
    
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
    # #region agent log
    raw_score = result.get("relevance_score", 0)
    _log("D", "app.py:277", "Processing relevance_score", {"raw_score": raw_score, "type": type(raw_score).__name__})
    # #endregion
    
    # #region agent log
    try:
        score = int(raw_score) if raw_score is not None else 0
        _log("D", "app.py:280", "relevance_score converted successfully", {"score": score})
    except (ValueError, TypeError) as e:
        _log("D", "app.py:280", "relevance_score conversion failed", {"error": str(e), "raw_score": raw_score})
        score = 0
    # #endregion
    
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


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Requirement Checker",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 Requirement Checker")
    st.markdown("---")
    
    # Project ID input (global setting) - default from testing notebook
    DEFAULT_PROJECT_ID = "93d3a25b-d15d-4689-affb-d027bbc422e7"
    project_id = st.sidebar.text_input(
        "Project ID",
        value=DEFAULT_PROJECT_ID,
        help="Enter the project_id to check requirements against documents in Supabase",
        key="project_id_input"
    )
    
    # Set top_k to default value of 5
    top_k = 5
    
    # Reset button in sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset App Session", help="Clear all session state and reload"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
    
    if not project_id:
        st.info("👈 Please enter a Project ID in the sidebar to continue.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Requirement", "📊 CSV Batch Processing", "✂️ Requirement Breakdown"])
    
    with tab1:
        show_single_requirement_tab(project_id, top_k)
    
    with tab2:
        show_csv_batch_tab(project_id, top_k)

    with tab3:
        show_breakdown_tab(project_id)


def show_breakdown_tab(project_id: str):
    """Show requirement breakdown testing tab."""
    st.header("Requirement Breakdown Testing")
    st.markdown("Test the requirement breakdown prompt and see how it splits complex requirements.")

    # Callback for reset
    def reset_breakdown():
        st.session_state.current_breakdown_prompt = DEFAULT_BREAKDOWN_PROMPT
        st.session_state.breakdown_prompt_editor = DEFAULT_BREAKDOWN_PROMPT

    # Callback for load
    def load_breakdown():
        # #region agent log
        _log("B", "app.py:382", "load_breakdown called", {})
        # #endregion
        if "breakdown_prompt_to_load" in st.session_state:
            selected = st.session_state.breakdown_prompt_to_load
            # #region agent log
            _log("B", "app.py:386", "Attempting to load breakdown prompt", {"selected": selected, "available_keys": list(st.session_state.saved_breakdown_prompts.keys())})
            # #endregion
            try:
                st.session_state.current_breakdown_prompt = st.session_state.saved_breakdown_prompts[selected]
                st.session_state.breakdown_prompt_editor = st.session_state.saved_breakdown_prompts[selected]
                # #region agent log
                _log("B", "app.py:389", "Breakdown prompt loaded successfully", {"selected": selected})
                # #endregion
            except KeyError as e:
                # #region agent log
                _log("B", "app.py:389", "KeyError loading breakdown prompt", {"selected": selected, "error": str(e)})
                # #endregion
                raise

    # Prompt editor with enhanced UI
    st.subheader("📝 Breakdown Prompt Template Editor")
    
    # Enhanced prompt editor with better styling
    col1, col2 = st.columns([3, 1])
    
    with col1:
        breakdown_prompt_template = st.text_area(
            "Edit the breakdown prompt template below:",
            value=st.session_state.current_breakdown_prompt,
            height=450,
            help="💡 Tip: Use {{query}} as placeholder",
            key="breakdown_prompt_editor",
            label_visibility="visible"
        )
    
    with col2:
        st.markdown("### Quick Actions")
        if st.button("🔄 Reset to Default", key="reset_breakdown_btn", use_container_width=True, on_click=reset_breakdown):
            st.rerun()
        
        if st.button("📋 Copy Prompt", key="copy_breakdown_btn", use_container_width=True):
            st.code(breakdown_prompt_template, language=None)
            st.success("Prompt copied! Use Ctrl+C to copy from above.")
        
        st.markdown("---")
        st.markdown("### 💾 Manage Prompts")
        
        # Load logic
        breakdown_names = list(st.session_state.saved_breakdown_prompts.keys())
        # #region agent log
        _log("E", "app.py:420", "Initializing breakdown selectbox", {"options_count": len(breakdown_names), "options": breakdown_names})
        # #endregion
        if breakdown_names:
            st.selectbox("📂 Load Saved Breakdown Prompt", options=breakdown_names, index=0, key="breakdown_prompt_to_load")
        else:
            # #region agent log
            _log("E", "app.py:423", "Empty breakdown_names list, using placeholder", {})
            # #endregion
            st.selectbox("📂 Load Saved Breakdown Prompt", options=["No prompts available"], index=0, key="breakdown_prompt_to_load", disabled=True)
        if st.button("📂 Load Selected", key="load_breakdown_btn", use_container_width=True, on_click=load_breakdown):
            st.rerun()

        # Save logic
        new_breakdown_name = st.text_input("New Breakdown Name", key="new_breakdown_name", placeholder="e.g., Breakdown V2")
        if st.button("💾 Save Current", key="save_breakdown_btn", use_container_width=True):
            if new_breakdown_name:
                success = st.session_state.supabase_client.save_prompt(
                    name=new_breakdown_name, 
                    prompt_text=breakdown_prompt_template,
                    project_id=project_id if project_id else None,
                    prompt_type="breakdown"
                )
                if success:
                    # Add to saved_prompts first (this is what gets saved to Supabase)
                    st.session_state.saved_prompts[new_breakdown_name] = breakdown_prompt_template
                    # Always add to saved_breakdown_prompts since this is explicitly a breakdown prompt
                    st.session_state.saved_breakdown_prompts[new_breakdown_name] = breakdown_prompt_template
                    st.success(f"Saved to database!")
                    st.rerun()
                else:
                    st.error("Failed to save to database.")
            else:
                st.error("Enter a name")
        
        st.markdown("---")
        st.markdown("### Placeholders")
        st.markdown("""
        <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0;'>
            <code style='color: #d63384;'>{{query}}</code>
        </div>
        """, unsafe_allow_html=True)

    # Preview section
    with st.expander("👁️ Preview Breakdown Prompt with Sample Values", expanded=False):
        sample_query = "The system must do X and Y while monitoring Z."
        preview_prompt = breakdown_prompt_template.replace("{{query}}", sample_query)
        st.text_area(
            "Preview:",
            value=preview_prompt[:2000] + ("..." if len(preview_prompt) > 2000 else ""),
            height=200,
            disabled=True,
            key="breakdown_preview"
        )

    # Save to session state
    st.session_state.current_breakdown_prompt = breakdown_prompt_template

    st.markdown("---")
    
    # Requirement input
    st.subheader("Requirement Input & Testing")
    
    # Comparison Mode
    breakdown_prompts_to_compare = st.multiselect(
        "⚖️ Select Saved Breakdown Prompts to Compare",
        options=list(st.session_state.saved_breakdown_prompts.keys()),
        default=[],
        help="Select multiple breakdown prompts to see results side-by-side. If empty, only current prompt will run."
    )
    
    requirement_text = st.text_area(
        "Enter Requirement to Break Down",
        height=100,
        help="Enter a complex requirement to break down into sub-requirements",
        placeholder="Enter a complex requirement here...",
        key="breakdown_input"
    )
    
    if st.button("🚀 Break Down Requirement", type="primary", key="run_breakdown"):
        if not requirement_text:
            st.warning("Please enter a requirement first.")
            return
        
        # Decide which prompts to run
        if breakdown_prompts_to_compare:
            # Run comparison
            active_breakdown_prompts = { name: st.session_state.saved_breakdown_prompts[name] for name in breakdown_prompts_to_compare }
            
            with st.spinner(f"Comparing {len(active_breakdown_prompts)} breakdown prompts..."):
                # Side-by-side columns
                cols = st.columns(len(active_breakdown_prompts))
                for i, (name, template) in enumerate(active_breakdown_prompts.items()):
                    with cols[i]:
                        st.markdown(f"#### Breakdown Prompt: {name}")
                        breakdown_result = process_breakdown(requirement_text, template)
                        
                        st.markdown("### Result")
                        st.info(breakdown_result)
                        
                        # Download button for each comparison result
                        result_text = breakdown_result
                        st.download_button(
                            label="📥 Download Result",
                            data=result_text,
                            file_name=f"breakdown_result_{name.replace(' ', '_')}.txt",
                            mime="text/plain",
                            key=f"breakdown_download_comp_{i}"
                        )
                        
                        # Raw output expander
                        with st.expander("📄 View Raw Output", expanded=False):
                            st.text(breakdown_result)
        else:
            # Run single (current) prompt
            with st.spinner("Breaking down requirement..."):
                breakdown_result = process_breakdown(requirement_text, breakdown_prompt_template)
            
            st.markdown("---")
            st.subheader("Results")
            st.markdown("### Breakdown Result")
            st.info(breakdown_result)
            
            # Download result
            st.download_button(
                label="📥 Download Result",
                data=breakdown_result,
                file_name="breakdown_result.txt",
                mime="text/plain",
                key="main_breakdown_result_download"
            )
            
            # Option to use this breakdown for something else or just view it
            with st.expander("📄 View Raw Output", expanded=False):
                st.text(breakdown_result)


def show_single_requirement_tab(project_id: str, top_k: int):
    """Show single requirement testing tab."""
    st.header("Single Requirement Testing")
    st.markdown("Enter a requirement and customize the prompt to see the AI's analysis.")
    
    # Callback for reset
    def reset_single():
        st.session_state.current_prompt = DEFAULT_PROMPT
        st.session_state.prompt_editor = DEFAULT_PROMPT

    # Callback for load
    def load_single():
        # #region agent log
        _log("B", "app.py:486", "load_single called", {})
        # #endregion
        if "prompt_to_load" in st.session_state:
            selected = st.session_state.prompt_to_load
            # #region agent log
            _log("B", "app.py:489", "Attempting to load prompt", {"selected": selected, "available_keys": list(st.session_state.saved_prompts.keys())})
            # #endregion
            try:
                st.session_state.current_prompt = st.session_state.saved_prompts[selected]
                st.session_state.prompt_editor = st.session_state.saved_prompts[selected]
                # #region agent log
                _log("B", "app.py:492", "Prompt loaded successfully", {"selected": selected})
                # #endregion
            except KeyError as e:
                # #region agent log
                _log("B", "app.py:492", "KeyError loading prompt", {"selected": selected, "error": str(e)})
                # #endregion
                raise
    
    # Prompt editor with enhanced UI
    st.subheader("📝 Prompt Template Editor")
    
    # Enhanced prompt editor with better styling
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt_template = st.text_area(
            "Edit the prompt template below:",
            value=st.session_state.current_prompt,
            height=450,
            help="💡 Tip: Use {{requirement_text}} and {{documentChunkTexts}} as placeholders",
            key="prompt_editor",
            label_visibility="visible"
        )
    
    with col2:
        st.markdown("### Quick Actions")
        if st.button("🔄 Reset to Default", use_container_width=True, on_click=reset_single):
            st.rerun()
        
        if st.button("📋 Copy Prompt", use_container_width=True):
            st.code(prompt_template, language=None)
            st.success("Prompt copied! Use Ctrl+C to copy from above.")
        
        st.markdown("---")
        st.markdown("### 💾 Manage Prompts")
        
        # Load logic
        prompt_names = list(st.session_state.saved_prompts.keys())
        # #region agent log
        _log("E", "app.py:524", "Initializing single prompt selectbox", {"options_count": len(prompt_names), "options": prompt_names})
        # #endregion
        if prompt_names:
            st.selectbox("📂 Load Saved Prompt", options=prompt_names, index=0, key="prompt_to_load")
        else:
            # #region agent log
            _log("E", "app.py:527", "Empty prompt_names list, using placeholder", {})
            # #endregion
            st.selectbox("📂 Load Saved Prompt", options=["No prompts available"], index=0, key="prompt_to_load", disabled=True)
        if st.button("📂 Load Selected", use_container_width=True, on_click=load_single):
            st.rerun()

        # Save logic
        new_prompt_name = st.text_input("New Prompt Name", placeholder="e.g., Reasoning V2")
        if st.button("💾 Save Current", use_container_width=True):
            if new_prompt_name:
                # Save to Supabase
                success = st.session_state.supabase_client.save_prompt(
                    name=new_prompt_name, 
                    prompt_text=prompt_template,
                    project_id=project_id if project_id else None,
                    prompt_type="requirement"
                )
                if success:
                    st.session_state.saved_prompts[new_prompt_name] = prompt_template
                    st.success(f"Saved to database!")
                    st.rerun()
                else:
                    st.error("Failed to save to database.")
            else:
                st.error("Enter a name")
        
        st.markdown("---")
        st.markdown("### Placeholders")
        st.markdown("""
        <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0;'>
            <code style='color: #d63384;'>{{requirement_text}}</code>
        </div>
        <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0;'>
            <code style='color: #856404;'>{{documentChunkTexts}}</code>
        </div>
        """, unsafe_allow_html=True)
    
    # Preview section
    with st.expander("👁️ Preview Prompt with Sample Values", expanded=False):
        sample_requirement = "Sample requirement text here"
        sample_chunks = "chunk-1: Sample document content...\nchunk-2: More sample content..."
        preview_prompt = prompt_template.replace("{{requirement_text}}", sample_requirement)
        preview_prompt = preview_prompt.replace("{{documentChunkTexts}}", sample_chunks)
        st.text_area(
            "Preview:",
            value=preview_prompt[:2000] + ("..." if len(preview_prompt) > 2000 else ""),
            height=200,
            disabled=True,
            key="prompt_preview"
        )
    
    # Save prompt to session state
    st.session_state.current_prompt = prompt_template
    
    st.markdown("---")
    
    # Requirement input
    st.subheader("Requirement Input & Testing")
    
    # Comparison Mode
    prompts_to_compare = st.multiselect(
        "⚖️ Select Saved Prompts to Compare",
        options=list(st.session_state.saved_prompts.keys()),
        default=[],
        help="Select multiple prompts to see results side-by-side. If empty, only current prompt will run."
    )

    requirement_text = st.text_area(
        "Enter Requirement",
        height=100,
        help="Enter the requirement to verify against the documents",
        key="single_requirement_input"
    )
    
    if st.button("🚀 Process Requirement", type="primary", key="process_single"):
        if not requirement_text:
            st.warning("Please enter a requirement first.")
            return
        
        # Decide which prompts to run
        if prompts_to_compare:
            # Run comparison
            active_prompts = { name: st.session_state.saved_prompts[name] for name in prompts_to_compare }
            
            with st.spinner(f"Comparing {len(active_prompts)} prompts..."):
                document_chunks_text, chunks_metadata = get_merged_chunks(requirement_text, project_id, top_k)
                if not document_chunks_text:
                    st.error("No relevant documents found.")
                    return
                
                # Side-by-side columns
                cols = st.columns(len(active_prompts))
                for i, (name, template) in enumerate(active_prompts.items()):
                    with cols[i]:
                        st.markdown(f"#### Prompt: {name}")
                        result, _ = process_requirement_with_chunks(
                            requirement=requirement_text,
                            document_chunks_text=document_chunks_text,
                            chunks_metadata=chunks_metadata,
                            prompt_template=template
                        )
                        display_result(result, chunks_metadata, key_suffix=f"comp_{i}")
        else:
            # Run single (current) prompt
            with st.spinner("Processing requirement..."):
                result, chunks_metadata = process_requirement(
                    requirement=requirement_text,
                    project_id=project_id,
                    prompt_template=prompt_template,
                    top_k=top_k
                )
            
            st.markdown("---")
            st.subheader("Results")
            display_result(result, chunks_metadata, key_suffix="single")
            
            # Download result with enriched citations
            enriched_result = result.copy()
            if chunks_metadata and result.get("citations"):
                enriched_citations = []
                for citation in result.get("citations", []):
                    enriched_citation = enrich_citation_with_doc_info(citation, chunks_metadata)
                    enriched_citations.append(enriched_citation)
                enriched_result["citations"] = enriched_citations
            
            result_json = json.dumps(enriched_result, indent=2)
            st.download_button(
                label="Download Result (JSON)",
                data=result_json,
                file_name="requirement_result.json",
                mime="application/json",
                key="main_single_result_download"
            )


def show_csv_batch_tab(project_id: str, top_k: int):
    """Show CSV batch processing tab."""
    st.header("CSV Batch Processing")
    st.markdown("Upload a CSV file with requirements and process each row.")
    
    # CSV upload
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload a CSV file containing requirements to process"
    )
    
    if not uploaded_file:
        st.info("👆 Please upload a CSV file to continue.")
        return
    
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns.")
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return
    
    # Column selection
    st.markdown("---")
    st.subheader("Column Mapping")
    
    col1, col2 = st.columns(2)
    
    with col1:
        requirement_column = st.selectbox(
            "Select column for requirements/queries",
            options=df.columns.tolist(),
            help="Select which column contains the requirements to check"
        )
    
    with col2:
        # Show sample of the selected column
        if requirement_column:
            st.write("**Sample values:**")
            sample_values = df[requirement_column].dropna().head(3).tolist()
            for val in sample_values:
                st.caption(f"- {str(val)[:100]}...")
    
    # Callback for reset
    def reset_batch():
        st.session_state.current_prompt = DEFAULT_PROMPT
        st.session_state.batch_prompt_editor = DEFAULT_PROMPT

    # Callback for load
    def load_batch():
        # #region agent log
        _log("B", "app.py:709", "load_batch called", {})
        # #endregion
        if "batch_prompt_to_load" in st.session_state:
            selected = st.session_state.batch_prompt_to_load
            # #region agent log
            _log("B", "app.py:712", "Attempting to load batch prompt", {"selected": selected, "available_keys": list(st.session_state.saved_prompts.keys())})
            # #endregion
            try:
                st.session_state.current_prompt = st.session_state.saved_prompts[selected]
                st.session_state.batch_prompt_editor = st.session_state.saved_prompts[selected]
                # #region agent log
                _log("B", "app.py:715", "Batch prompt loaded successfully", {"selected": selected})
                # #endregion
            except KeyError as e:
                # #region agent log
                _log("B", "app.py:715", "KeyError loading batch prompt", {"selected": selected, "error": str(e)})
                # #endregion
                raise
    
    # Prompt editor for batch processing with enhanced UI
    st.markdown("---")
    st.subheader("📝 Prompt Template Editor")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        batch_prompt_template = st.text_area(
            "Edit the prompt template below:",
            value=st.session_state.current_prompt,
            height=350,
            help="💡 Tip: Use {{requirement_text}} and {{documentChunkTexts}} as placeholders",
            key="batch_prompt_editor",
            label_visibility="visible"
        )
    
    with col2:
        st.markdown("### Quick Actions")
        if st.button("🔄 Reset to Default", key="reset_batch", use_container_width=True, on_click=reset_batch):
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 💾 Manage Prompts")
        
        # Load logic
        prompt_names = list(st.session_state.saved_prompts.keys())
        # #region agent log
        _log("E", "app.py:740", "Initializing batch prompt selectbox", {"options_count": len(prompt_names), "options": prompt_names})
        # #endregion
        if prompt_names:
            st.selectbox("📂 Load Saved Prompt", options=prompt_names, index=0, key="batch_prompt_to_load")
        else:
            # #region agent log
            _log("E", "app.py:743", "Empty prompt_names list for batch, using placeholder", {})
            # #endregion
            st.selectbox("📂 Load Saved Prompt", options=["No prompts available"], index=0, key="batch_prompt_to_load", disabled=True)
        if st.button("📂 Load Selected", key="load_batch_btn", use_container_width=True, on_click=load_batch):
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Placeholders")
        st.markdown("""
        <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0;'>
            <code style='color: #d63384;'>{{requirement_text}}</code>
        </div>
        <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0;'>
            <code style='color: #856404;'>{{documentChunkTexts}}</code>
        </div>
        """, unsafe_allow_html=True)
    
    # Process button
    if st.button("Process CSV", type="primary", key="process_csv"):
        if requirement_column not in df.columns:
            st.error("Selected requirement column not found in CSV.")
            return
        
        # Initialize result columns if they don't exist
        result_columns = {
            "result_status": [],
            "result_relevance_score": [],
            "result_justification": [],
            "result_citations": [],
            "result_json": []
        }
        
        # Process each row
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_rows = len(df)
        results_data = []
        chunks_metadata_list = []  # Store chunks metadata for each row
        
        for idx, row in df.iterrows():
            requirement = str(row[requirement_column]) if pd.notna(row[requirement_column]) else ""
            
            if not requirement or requirement.strip() == "":
                # Empty requirement - add empty results
                results_data.append({
                    "status": "SKIPPED",
                    "relevance_score": 0,
                    "justification": "Empty requirement",
                    "citations": []
                })
                chunks_metadata_list.append([])
            else:
                # Process requirement
                # #region agent log
                _log("G", "app.py:795", "Processing CSV row", {"row_idx": idx, "requirement_preview": requirement[:50]})
                # #endregion
                try:
                    result, chunks_metadata = process_requirement(
                        requirement=requirement,
                        project_id=project_id,
                        prompt_template=batch_prompt_template,
                        top_k=top_k
                    )
                    results_data.append(result)
                    chunks_metadata_list.append(chunks_metadata)
                    # #region agent log
                    _log("G", "app.py:804", "CSV row processed successfully", {"row_idx": idx, "status": result.get("status")})
                    # #endregion
                except Exception as e:
                    # #region agent log
                    _log("G", "app.py:807", "CSV row processing failed", {"row_idx": idx, "error": str(e), "traceback": traceback.format_exc()})
                    # #endregion
                    results_data.append({
                        "status": "ERROR",
                        "relevance_score": 0,
                        "justification": f"Error processing row {idx + 1}: {str(e)}",
                        "citations": []
                    })
                    chunks_metadata_list.append([])
            
            # Update progress
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing row {idx + 1}/{total_rows}: {requirement[:50]}...")
        
        # Add results to dataframe
        df_results = df.copy()
        
        # Enrich citations with document names
        enriched_results_data = []
        for i, result in enumerate(results_data):
            enriched_result = result.copy()
            if chunks_metadata_list[i] and result.get("citations"):
                enriched_citations = []
                for citation in result.get("citations", []):
                    enriched_citation = enrich_citation_with_doc_info(citation, chunks_metadata_list[i])
                    enriched_citations.append(enriched_citation)
                enriched_result["citations"] = enriched_citations
            enriched_results_data.append(enriched_result)
        
        # Add result columns
        df_results["result_status"] = [r.get("status", "UNKNOWN") for r in enriched_results_data]
        df_results["result_relevance_score"] = [r.get("relevance_score", 0) for r in enriched_results_data]
        df_results["result_justification"] = [r.get("justification", "") for r in enriched_results_data]
        df_results["result_citations_count"] = [len(r.get("citations", [])) for r in enriched_results_data]
        
        # Format citations as a readable string
        def format_citations(citations_list):
            if not citations_list:
                return ""
            formatted = []
            for i, cit in enumerate(citations_list, 1):
                doc_name = cit.get("document_name") or cit.get("document_reference", "Unknown")
                source_text = cit.get("source_text", "")[:150]  # Truncate for table display
                if len(cit.get("source_text", "")) > 150:
                    source_text += "..."
                formatted.append(f"{i}. [{doc_name}]: {source_text}")
            return "\n".join(formatted)
        
        df_results["result_citations"] = [format_citations(r.get("citations", [])) for r in enriched_results_data]
        df_results["result_json"] = [json.dumps(r) for r in enriched_results_data]
        
        # Store in session state
        st.session_state.processed_df = df_results
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Successfully processed {total_rows} rows!")
        
        # Display results table
        st.markdown("---")
        st.subheader("Results Table")
        st.markdown("All original columns + result columns added to the right")
        
        # Show all original columns first, then result columns (excluding result_json from display)
        display_columns = [col for col in df.columns]  # All original columns
        display_columns.extend([
            col for col in df_results.columns 
            if col.startswith("result_") and col != "result_json"
        ])
        
        st.dataframe(
            df_results[display_columns],
            use_container_width=True,
            height=400
        )
        
        # Expandable view for full JSON - using individual expanders (can't nest expanders)
        st.markdown("---")
        st.subheader("Detailed Results (JSON)")
        st.markdown("Expand any row below to view the full JSON result:")
        
        for idx, row in df_results.iterrows():
            row_label = f"Row {idx + 1}: {str(row[requirement_column])[:50]}..." if pd.notna(row[requirement_column]) else f"Row {idx + 1}"
            with st.expander(row_label, expanded=False):
                result_data = json.loads(row["result_json"])
                st.json(result_data)
        
        # Download results
        st.markdown("---")
        st.subheader("Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_data,
                file_name="processed_requirements.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download the complete CSV with all original columns and result columns"
            )
        
        with col2:
            json_data = df_results.to_dict(orient="records")
            json_string = json.dumps(json_data, indent=2)
            st.download_button(
                label="📥 Download Results JSON",
                data=json_string,
                file_name="processed_requirements.json",
                mime="application/json",
                use_container_width=True,
                help="Download the complete results as JSON"
            )


if __name__ == "__main__":
    main()

