"""Main Streamlit application for Requirement Checker."""
import io
import time
import streamlit as st
import pandas as pd
import json
import traceback
import threading
from typing import List, Dict, Any
from datetime import datetime

# Import utilities
from utils.session_state import initialize_session_state
from utils.prompts import DEFAULT_PROMPT, DEFAULT_BREAKDOWN_PROMPT
from utils.rag_operations import get_merged_chunks
from utils.ai_processing import (
    process_breakdown,
    process_requirement, 
    process_requirement_with_chunks
)
from utils.ui_components import display_chunks_metadata, display_result, enrich_citation_with_doc_info
from utils.pdf_processor import extract_text_from_pdf, extract_text_from_pdf_with_llamaparse
from utils.logging import log
from config.config import OPENAI_MODEL

try:
    from config.config import LLAMA_CLOUD_API_KEY
except ImportError:
    LLAMA_CLOUD_API_KEY = ""

try:
    from config.config import PROJECT_IDS
except ImportError:
    PROJECT_IDS = {
        "Building services": "fda85e04-3a9c-4e6f-8af0-35bfcb1ba4e0",
        "Tender requirement": "1375eed6-8f48-41c2-bd92-444e6acc7721",
        "Tender Requirement-2": "00d06bf0-7572-4f23-aaec-46a9adad5e63",
        "Tender_handbook": "05ac317c-e700-4d5b-a99e-a1a92fc619e5",
    }
try:
    from config.config import PAGEINDEX_API_KEY, PAGEINDEX_CHAT_URL
except (ImportError, AttributeError):
    PAGEINDEX_API_KEY = ""
    PAGEINDEX_CHAT_URL = "https://api.pageindex.ai/chat/completions"
from tender_checker.workflow import TenderCheckWorkflow
from clients.pageindex_client import PageIndexClient
from tender_checker.prompts.agent_prompts import (
    BREAKDOWN_AGENT_PROMPT,
    OMISSION_CHECKER_PROMPT,
    CONTRADICTION_CHECKER_PROMPT,
    ORCHESTRATOR_PROMPT
)

# Initialize session state
initialize_session_state()


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
        log("B", "app.py", "load_breakdown called", {})
        if "breakdown_prompt_to_load" in st.session_state:
            selected = st.session_state.breakdown_prompt_to_load
            log("B", "app.py", "Attempting to load breakdown prompt", {"selected": selected, "available_keys": list(st.session_state.saved_breakdown_prompts.keys())})
            try:
                st.session_state.current_breakdown_prompt = st.session_state.saved_breakdown_prompts[selected]
                st.session_state.breakdown_prompt_editor = st.session_state.saved_breakdown_prompts[selected]
                log("B", "app.py", "Breakdown prompt loaded successfully", {"selected": selected})
            except KeyError as e:
                log("B", "app.py", "KeyError loading breakdown prompt", {"selected": selected, "error": str(e)})
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
            help="💡 Tip: Use {{requirement_text}} or {{query}} as placeholder",
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
        log("E", "app.py", "Initializing breakdown selectbox", {"options_count": len(breakdown_names), "options": breakdown_names})
        if breakdown_names:
            st.selectbox("📂 Load Saved Breakdown Prompt", options=breakdown_names, index=0, key="breakdown_prompt_to_load")
        else:
            log("E", "app.py", "Empty breakdown_names list, using placeholder", {})
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
                    st.session_state.saved_prompts[new_breakdown_name] = breakdown_prompt_template
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
            <code style='color: #d63384;'>{{requirement_text}}</code>
        </div>
        <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0;'>
            <code style='color: #d63384;'>{{query}}</code> (legacy, also supported)
        </div>
        """, unsafe_allow_html=True)

    # Preview section
    with st.expander("👁️ Preview Breakdown Prompt with Sample Values", expanded=False):
        sample_query = "The system must do X and Y while monitoring Z."
        preview_prompt = breakdown_prompt_template.replace("{{requirement_text}}", sample_query)
        preview_prompt = preview_prompt.replace("{{query}}", sample_query)
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
            
            # Try to parse and display as JSON if possible
            try:
                breakdown_json = json.loads(breakdown_result)
                st.json(breakdown_json)
                
                # Extract and display requirements if available
                if "requirements" in breakdown_json:
                    st.markdown("### Extracted Requirements:")
                    for i, req in enumerate(breakdown_json["requirements"], 1):
                        with st.expander(f"Requirement {i}: {req.get('id', f'REQ-{i}')}", expanded=False):
                            if isinstance(req, dict):
                                st.markdown(f"**Responsible Entity:** {req.get('responsible_entity', 'N/A')}")
                                st.markdown(f"**Context:** {req.get('overarching_context', 'N/A')}")
                                st.markdown(f"**Action:** {req.get('specific_action', 'N/A')}")
                                st.markdown(f"**Detailed Requirement:** {req.get('detailed_requirement', 'N/A')}")
                                st.markdown(f"**Success Criteria:** {req.get('success_criteria', 'N/A')}")
                                st.markdown(f"**Compliance Statement:**")
                                st.code(req.get('compliance_verification_statement', 'N/A'))
                            else:
                                st.text(str(req))
            except json.JSONDecodeError:
                # Not JSON, display as text
                st.info(breakdown_result)
            
            # Download result
            st.download_button(
                label="📥 Download Result",
                data=breakdown_result,
                file_name="breakdown_result.json" if breakdown_result.strip().startswith('{') else "breakdown_result.txt",
                mime="application/json" if breakdown_result.strip().startswith('{') else "text/plain",
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
        log("B", "app.py", "load_single called", {})
        if "prompt_to_load" in st.session_state:
            selected = st.session_state.prompt_to_load
            log("B", "app.py", "Attempting to load prompt", {"selected": selected, "available_keys": list(st.session_state.saved_prompts.keys())})
            try:
                st.session_state.current_prompt = st.session_state.saved_prompts[selected]
                st.session_state.prompt_editor = st.session_state.saved_prompts[selected]
                log("B", "app.py", "Prompt loaded successfully", {"selected": selected})
            except KeyError as e:
                log("B", "app.py", "KeyError loading prompt", {"selected": selected, "error": str(e)})
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
        log("E", "app.py", "Initializing single prompt selectbox", {"options_count": len(prompt_names), "options": prompt_names})
        if prompt_names:
            st.selectbox("📂 Load Saved Prompt", options=prompt_names, index=0, key="prompt_to_load")
        else:
            log("E", "app.py", "Empty prompt_names list, using placeholder", {})
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
    
    # PDF Upload option
    uploaded_pdf = st.file_uploader(
        "📄 Upload PDF Document (Optional)",
        type=["pdf"],
        help="Upload a PDF document to extract text from. The extracted text will be used as the requirement to verify."
    )
    
    # Search method selection
    col1, col2 = st.columns([1, 2])
    with col1:
        search_method = st.radio(
            "🔍 Search Method",
            options=["semantic", "hybrid"],
            index=0,
            help="Semantic: Vector similarity only. Hybrid: Combines keyword + semantic search with RRF ranking.",
            horizontal=True
        )
    
    with col2:
        if search_method == "hybrid":
            st.caption("💡 Hybrid search combines keyword matching with semantic understanding for better accuracy")
        else:
            st.caption("💡 Semantic search finds documents by meaning/similarity")
    
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
        help="Enter the requirement to verify against the documents. If PDF is uploaded, this field can be used for additional context.",
        key="single_requirement_input"
    )
    
    if st.button("🚀 Process Requirement", type="primary", key="process_single"):
        # Handle PDF upload
        input_text = requirement_text
        if uploaded_pdf:
            st.info("📄 Processing PDF document...")
            pdf_text = extract_text_from_pdf(uploaded_pdf)
            if pdf_text:
                # Combine PDF text with requirement text if provided
                if requirement_text.strip():
                    input_text = f"{requirement_text}\n\n--- PDF Document Content ---\n\n{pdf_text}"
                else:
                    input_text = pdf_text
                st.success(f"✓ Extracted {len(pdf_text)} characters from PDF")
            else:
                st.error("Failed to extract text from PDF. Please try again.")
            return
        
        if not input_text or not input_text.strip():
            st.warning("Please enter a requirement or upload a PDF document.")
            return
        
        # Use input text directly as requirement
        requirements_text = input_text
        
        # Decide which prompts to run
        if prompts_to_compare:
            # Run comparison
            active_prompts = { name: st.session_state.saved_prompts[name] for name in prompts_to_compare }
            
            with st.spinner(f"Comparing {len(active_prompts)} prompts..."):
                document_chunks_text, chunks_metadata = get_merged_chunks(
                    requirements_text, 
                    project_id, 
                    top_k,
                    search_method=search_method
                )
                if not document_chunks_text:
                    st.error("No relevant documents found.")
                    return
                
                # Display chunk metadata first
                st.markdown("---")
                display_chunks_metadata(chunks_metadata, search_method=search_method)
                st.markdown("---")
                
                # Side-by-side columns
                cols = st.columns(len(active_prompts))
                for i, (name, template) in enumerate(active_prompts.items()):
                    with cols[i]:
                        st.markdown(f"#### Prompt: {name}")
                        result, _ = process_requirement_with_chunks(
                            requirement=requirements_text,
                            document_chunks_text=document_chunks_text,
                            chunks_metadata=chunks_metadata,
                            prompt_template=template
                        )
                        display_result(result, chunks_metadata, key_suffix=f"comp_{i}")
        else:
            # Run single (current) prompt
            with st.spinner("Processing requirement..."):
                result, chunks_metadata = process_requirement(
                    requirement=requirements_text,
                    project_id=project_id,
                    prompt_template=prompt_template,
                    top_k=top_k,
                    search_method=search_method
                )
            
            st.markdown("---")
            # Display chunk metadata before results
            display_chunks_metadata(chunks_metadata, search_method=search_method)
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
        log("B", "app.py", "load_batch called", {})
        if "batch_prompt_to_load" in st.session_state:
            selected = st.session_state.batch_prompt_to_load
            log("B", "app.py", "Attempting to load batch prompt", {"selected": selected, "available_keys": list(st.session_state.saved_prompts.keys())})
            try:
                st.session_state.current_prompt = st.session_state.saved_prompts[selected]
                st.session_state.batch_prompt_editor = st.session_state.saved_prompts[selected]
                log("B", "app.py", "Batch prompt loaded successfully", {"selected": selected})
            except KeyError as e:
                log("B", "app.py", "KeyError loading batch prompt", {"selected": selected, "error": str(e)})
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
        log("E", "app.py", "Initializing batch prompt selectbox", {"options_count": len(prompt_names), "options": prompt_names})
        if prompt_names:
            st.selectbox("📂 Load Saved Prompt", options=prompt_names, index=0, key="batch_prompt_to_load")
        else:
            log("E", "app.py", "Empty prompt_names list for batch, using placeholder", {})
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
    
    # Search method selection for batch processing
    st.markdown("---")
    st.subheader("🔍 Search Configuration")
    col1, col2 = st.columns([1, 2])
    with col1:
        batch_search_method = st.radio(
            "Search Method",
            options=["semantic", "hybrid"],
            index=0,
            help="Semantic: Vector similarity only. Hybrid: Combines keyword + semantic search with RRF ranking.",
            horizontal=True,
            key="batch_search_method"
        )
    with col2:
        if batch_search_method == "hybrid":
            st.caption("💡 Hybrid search combines keyword matching with semantic understanding for better accuracy")
        else:
            st.caption("💡 Semantic search finds documents by meaning/similarity")
    
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
                log("G", "app.py", "Processing CSV row", {"row_idx": idx, "requirement_preview": requirement[:50]})
                try:
                    result, chunks_metadata = process_requirement(
                        requirement=requirement,
                        project_id=project_id,
                        prompt_template=batch_prompt_template,
                        top_k=top_k,
                        search_method=batch_search_method
                    )
                    results_data.append(result)
                    chunks_metadata_list.append(chunks_metadata)
                    log("G", "app.py", "CSV row processed successfully", {"row_idx": idx, "status": result.get("status")})
                except Exception as e:
                    log("G", "app.py", "CSV row processing failed", {"row_idx": idx, "error": str(e), "traceback": traceback.format_exc()})
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


def show_tender_check_tab():
    """Show tender checking tab with multi-agent system."""
    st.header("📄 Tender Submission Checker")
    st.markdown("Multi-agent system for checking tender submissions against reference documents and guidelines.")
    
    # Initialize session state for tender checker prompts
    if "tender_breakdown_prompt" not in st.session_state:
        st.session_state.tender_breakdown_prompt = BREAKDOWN_AGENT_PROMPT
    if "tender_omission_prompt" not in st.session_state:
        st.session_state.tender_omission_prompt = OMISSION_CHECKER_PROMPT
    if "tender_contradiction_prompt" not in st.session_state:
        st.session_state.tender_contradiction_prompt = CONTRADICTION_CHECKER_PROMPT
    if "tender_orchestrator_prompt" not in st.session_state:
        st.session_state.tender_orchestrator_prompt = ORCHESTRATOR_PROMPT

    # Apply any queued prompt loads before rendering widgets so the UI reflects them
    if "tender_breakdown_prompt_to_apply" in st.session_state:
        new_val = st.session_state.tender_breakdown_prompt_to_apply
        st.session_state["tender_breakdown_prompt_editor"] = new_val
        st.session_state.tender_breakdown_prompt = new_val
        del st.session_state.tender_breakdown_prompt_to_apply

    if "tender_omission_prompt_to_apply" in st.session_state:
        new_val = st.session_state.tender_omission_prompt_to_apply
        st.session_state["tender_omission_prompt_editor"] = new_val
        st.session_state.tender_omission_prompt = new_val
        del st.session_state.tender_omission_prompt_to_apply

    if "tender_contradiction_prompt_to_apply" in st.session_state:
        new_val = st.session_state.tender_contradiction_prompt_to_apply
        st.session_state["tender_contradiction_prompt_editor"] = new_val
        st.session_state.tender_contradiction_prompt = new_val
        del st.session_state.tender_contradiction_prompt_to_apply

    if "tender_orchestrator_prompt_to_apply" in st.session_state:
        new_val = st.session_state.tender_orchestrator_prompt_to_apply
        st.session_state["tender_orchestrator_prompt_editor"] = new_val
        st.session_state.tender_orchestrator_prompt = new_val
        del st.session_state.tender_orchestrator_prompt_to_apply
    
    # Evidence source: RAG (Supabase chunks) or PageIndex Legacy Retrieval (no Chat API)
    st.markdown("---")
    st.subheader("📁 Project Configuration")
    evidence_source = st.radio(
        "Evidence source",
        options=["RAG (Supabase chunks)", "PageIndex (Retrieval)"],
        index=0,
        help="RAG: retrieve chunks from Supabase. PageIndex: Legacy Retrieval API (submit query + poll) per requirement per doc → chunks (no Chat API).",
        key="tender_evidence_source"
    )
    use_pageindex_chat = evidence_source == "PageIndex (Retrieval)"

    if use_pageindex_chat:
        if not (PAGEINDEX_API_KEY or "").strip():
            st.warning("Set **PAGEINDEX_API_KEY** in `.env` to use PageIndex Retrieval.")
        st.markdown("**PageIndex document IDs** (from [PageIndex](https://docs.pageindex.ai/endpoints) — upload PDFs to get `doc_id`s)")
        col1, col2 = st.columns(2)
        with col1:
            reference_doc_id = st.text_input(
                "Reference doc_id (omission)",
                value=st.session_state.get("tender_reference_doc_id", "pi-cml3ep35803gd09qzoczvn1kr"),
                help="PageIndex doc_id for the reference document",
                key="tender_reference_doc_id"
            )
        with col2:
            guidelines_doc_id = st.text_input(
                "Guidelines doc_id (contradiction)",
                value=st.session_state.get("tender_guidelines_doc_id", ""),
                help="PageIndex doc_id for guidelines. Leave empty to use the reference doc for both.",
                key="tender_guidelines_doc_id"
            )
        reference_project_id = ""  # not used in PageIndex path
        guidelines_project_id = ""
        top_k = 5  # not used
        max_workers = 5
    else:
        col1, col2 = st.columns(2)
        with col1:
            reference_project = st.selectbox(
                "Reference Documents Project",
                options=list(PROJECT_IDS.keys()),
                index=2,
                help="Project ID for reference documents (used for omission checking)",
                key="tender_reference_project"
            )
            reference_project_id = PROJECT_IDS[reference_project]
        with col2:
            guidelines_project = st.selectbox(
                "Guidelines Project",
                options=list(PROJECT_IDS.keys()),
                index=2,
                help="Project ID for guidelines (used for contradiction checking). If same as reference, leave as is.",
                key="tender_guidelines_project"
            )
            guidelines_project_id = PROJECT_IDS[guidelines_project]
        reference_doc_id = ""
        guidelines_doc_id = ""

    # Advanced settings (RAG only: top_k and workers; PageIndex uses same workers for Chat calls)
    with st.expander("⚙️ Advanced Settings", expanded=False):
        top_k = st.slider(
            "Chunks per Requirement (top_k)",
            min_value=3,
            max_value=15,
            value=5,
            help="Number of document chunks to retrieve per requirement (RAG only)",
            key="tender_top_k",
            disabled=use_pageindex_chat
        )
        max_workers = st.slider(
            "Parallel Workers",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of parallel workers for requirement checking",
            key="tender_max_workers"
        )
        if use_pageindex_chat:
            top_k = 5
    
    st.markdown("---")
    
    # Agent Prompts Configuration
    st.subheader("🤖 Agent Prompts Configuration")
    
    prompt_tabs = st.tabs([
        "1️⃣ Breakdown Agent",
        "2️⃣ Omission Checker",
        "3️⃣ Contradiction Checker",
        "4️⃣ Orchestrator"
    ])
    
    with prompt_tabs[0]:
        st.markdown("**Breakdown Agent** - Extracts requirements from tender document")
        breakdown_prompt = st.text_area(
            "Breakdown Prompt Template",
            value=st.session_state.get("tender_breakdown_prompt_editor", st.session_state.tender_breakdown_prompt),
            height=300,
            help="Use {{tender_text}} as placeholder",
            key="tender_breakdown_prompt_editor"
        )
        st.session_state.tender_breakdown_prompt = breakdown_prompt
        if st.button("🔄 Reset to Default", key="reset_breakdown_tender"):
            st.session_state.tender_breakdown_prompt_editor = BREAKDOWN_AGENT_PROMPT
            st.session_state.tender_breakdown_prompt = BREAKDOWN_AGENT_PROMPT
            st.rerun()
        
        # Manage saved Breakdown prompts (tender-specific, compact UI)
        with st.expander("💾 Manage Breakdown Prompts", expanded=False):
            if "tender_saved_breakdown_prompts" not in st.session_state:
                st.session_state.tender_saved_breakdown_prompts = {"Default Tender Breakdown": BREAKDOWN_AGENT_PROMPT}
                try:
                    db_prompts = st.session_state.supabase_client.get_all_prompts(prompt_type="tender_breakdown")
                    for p in db_prompts:
                        st.session_state.tender_saved_breakdown_prompts[p["name"]] = p["prompt_text"]
                except Exception:
                    pass
            breakdown_names = list(st.session_state.tender_saved_breakdown_prompts.keys())
            if breakdown_names:
                selected_breakdown = st.selectbox(
                    "📂 Load Saved",
                    options=breakdown_names,
                    index=0,
                    key="tender_breakdown_prompt_to_load",
                )
            else:
                selected_breakdown = None
                st.selectbox(
                    "📂 Load Saved",
                    options=["No prompts available"],
                    index=0,
                    key="tender_breakdown_prompt_to_load",
                    disabled=True,
                )
            load_col, update_col = st.columns(2)
            with load_col:
                if st.button("Load Selected", key="load_tender_breakdown_prompt"):
                    if selected_breakdown:
                        # Queue value to apply on next run so widget state updates correctly
                        st.session_state.tender_breakdown_prompt_to_apply = st.session_state.tender_saved_breakdown_prompts[selected_breakdown]
                        st.rerun()
            new_breakdown_name = st.text_input(
                "New name",
                key="new_tender_breakdown_name",
                placeholder="e.g., Tender Breakdown V2",
            )
            with update_col:
                if st.button("Update Selected", key="update_tender_breakdown_prompt"):
                    if selected_breakdown:
                        success = st.session_state.supabase_client.save_prompt(
                            name=selected_breakdown,
                            prompt_text=breakdown_prompt,
                            project_id=reference_project_id if reference_project_id else None,
                            prompt_type="tender_breakdown",
                        )
                        if success:
                            st.session_state.tender_saved_breakdown_prompts[selected_breakdown] = breakdown_prompt
                            st.success("Updated tender breakdown prompt.")
                            st.rerun()
                        else:
                            st.error("Failed to update tender breakdown prompt.")
                    else:
                        st.error("Select a prompt to update.")
            if st.button("Save Current", key="save_tender_breakdown_prompt"):
                if new_breakdown_name:
                    success = st.session_state.supabase_client.save_prompt(
                        name=new_breakdown_name,
                        prompt_text=breakdown_prompt,
                        project_id=reference_project_id if reference_project_id else None,
                        prompt_type="tender_breakdown",
                    )
                    if success:
                        st.session_state.tender_saved_breakdown_prompts[new_breakdown_name] = breakdown_prompt
                        st.success("Saved tender breakdown prompt.")
                        st.rerun()
                    else:
                        st.error("Failed to save tender breakdown prompt.")
                else:
                    st.error("Enter a name for the breakdown prompt.")
    
    with prompt_tabs[1]:
        st.markdown("**Omission Checker** - Checks if requirements are fulfilled")
        omission_prompt = st.text_area(
            "Omission Checker Prompt Template",
            value=st.session_state.get("tender_omission_prompt_editor", st.session_state.tender_omission_prompt),
            height=300,
            help="Use {{requirement_text}}, {{requirement_id}}, {{reference_chunks}} as placeholders",
            key="tender_omission_prompt_editor"
        )
        st.session_state.tender_omission_prompt = omission_prompt
        if st.button("🔄 Reset to Default", key="reset_omission_tender"):
            st.session_state.tender_omission_prompt_editor = OMISSION_CHECKER_PROMPT
            st.session_state.tender_omission_prompt = OMISSION_CHECKER_PROMPT
            st.rerun()
        
        # Manage saved Omission prompts (tender-specific, compact UI)
        with st.expander("💾 Manage Omission Prompts", expanded=False):
            if "tender_saved_omission_prompts" not in st.session_state:
                st.session_state.tender_saved_omission_prompts = {"Default Tender Omission": OMISSION_CHECKER_PROMPT}
                try:
                    db_prompts = st.session_state.supabase_client.get_all_prompts(prompt_type="tender_omission")
                    for p in db_prompts:
                        st.session_state.tender_saved_omission_prompts[p["name"]] = p["prompt_text"]
                except Exception:
                    pass
            omission_names = list(st.session_state.tender_saved_omission_prompts.keys())
            if omission_names:
                selected_omission = st.selectbox(
                    "📂 Load Saved",
                    options=omission_names,
                    index=0,
                    key="tender_omission_prompt_to_load",
                )
            else:
                selected_omission = None
                st.selectbox(
                    "📂 Load Saved",
                    options=["No prompts available"],
                    index=0,
                    key="tender_omission_prompt_to_load",
                    disabled=True,
                )
            load_col, update_col = st.columns(2)
            with load_col:
                if st.button("Load Selected", key="load_tender_omission_prompt"):
                    if selected_omission:
                        st.session_state.tender_omission_prompt_to_apply = st.session_state.tender_saved_omission_prompts[selected_omission]
                        st.rerun()
            new_omission_name = st.text_input(
                "New name",
                key="new_tender_omission_name",
                placeholder="e.g., Tender Omission V2",
            )
            with update_col:
                if st.button("Update Selected", key="update_tender_omission_prompt"):
                    if selected_omission:
                        success = st.session_state.supabase_client.save_prompt(
                            name=selected_omission,
                            prompt_text=omission_prompt,
                            project_id=reference_project_id if reference_project_id else None,
                            prompt_type="tender_omission",
                        )
                        if success:
                            st.session_state.tender_saved_omission_prompts[selected_omission] = omission_prompt
                            st.success("Updated tender omission prompt.")
                            st.rerun()
                        else:
                            st.error("Failed to update tender omission prompt.")
                    else:
                        st.error("Select a prompt to update.")
            if st.button("Save Current", key="save_tender_omission_prompt"):
                if new_omission_name:
                    success = st.session_state.supabase_client.save_prompt(
                        name=new_omission_name,
                        prompt_text=omission_prompt,
                        project_id=reference_project_id if reference_project_id else None,
                        prompt_type="tender_omission",
                    )
                    if success:
                        st.session_state.tender_saved_omission_prompts[new_omission_name] = omission_prompt
                        st.success("Saved tender omission prompt.")
                        st.rerun()
                    else:
                        st.error("Failed to save tender omission prompt.")
                else:
                    st.error("Enter a name for the omission prompt.")
    
    with prompt_tabs[2]:
        st.markdown("**Contradiction Checker** - Checks for contradictions with guidelines")
        contradiction_prompt = st.text_area(
            "Contradiction Checker Prompt Template",
            value=st.session_state.get("tender_contradiction_prompt_editor", st.session_state.tender_contradiction_prompt),
            height=300,
            help="Use {{requirement_text}}, {{requirement_id}}, {{reference_chunks}} as placeholders",
            key="tender_contradiction_prompt_editor"
        )
        st.session_state.tender_contradiction_prompt = contradiction_prompt
        if st.button("🔄 Reset to Default", key="reset_contradiction_tender"):
            st.session_state.tender_contradiction_prompt_editor = CONTRADICTION_CHECKER_PROMPT
            st.session_state.tender_contradiction_prompt = CONTRADICTION_CHECKER_PROMPT
            st.rerun()
        
        # Manage saved Contradiction prompts (tender-specific, compact UI)
        with st.expander("💾 Manage Contradiction Prompts", expanded=False):
            if "tender_saved_contradiction_prompts" not in st.session_state:
                st.session_state.tender_saved_contradiction_prompts = {"Default Tender Contradiction": CONTRADICTION_CHECKER_PROMPT}
                try:
                    db_prompts = st.session_state.supabase_client.get_all_prompts(prompt_type="tender_contradiction")
                    for p in db_prompts:
                        st.session_state.tender_saved_contradiction_prompts[p["name"]] = p["prompt_text"]
                except Exception:
                    pass
            contradiction_names = list(st.session_state.tender_saved_contradiction_prompts.keys())
            if contradiction_names:
                selected_contradiction = st.selectbox(
                    "📂 Load Saved",
                    options=contradiction_names,
                    index=0,
                    key="tender_contradiction_prompt_to_load",
                )
            else:
                selected_contradiction = None
                st.selectbox(
                    "📂 Load Saved",
                    options=["No prompts available"],
                    index=0,
                    key="tender_contradiction_prompt_to_load",
                    disabled=True,
                )
            load_col, update_col = st.columns(2)
            with load_col:
                if st.button("Load Selected", key="load_tender_contradiction_prompt"):
                    if selected_contradiction:
                        st.session_state.tender_contradiction_prompt_to_apply = st.session_state.tender_saved_contradiction_prompts[selected_contradiction]
                        st.rerun()
            new_contradiction_name = st.text_input(
                "New name",
                key="new_tender_contradiction_name",
                placeholder="e.g., Tender Contradiction V2",
            )
            with update_col:
                if st.button("Update Selected", key="update_tender_contradiction_prompt"):
                    if selected_contradiction:
                        success = st.session_state.supabase_client.save_prompt(
                            name=selected_contradiction,
                            prompt_text=contradiction_prompt,
                            project_id=guidelines_project_id if guidelines_project_id else None,
                            prompt_type="tender_contradiction",
                        )
                        if success:
                            st.session_state.tender_saved_contradiction_prompts[selected_contradiction] = contradiction_prompt
                            st.success("Updated tender contradiction prompt.")
                            st.rerun()
                        else:
                            st.error("Failed to update tender contradiction prompt.")
                    else:
                        st.error("Select a prompt to update.")
            if st.button("Save Current", key="save_tender_contradiction_prompt"):
                if new_contradiction_name:
                    success = st.session_state.supabase_client.save_prompt(
                        name=new_contradiction_name,
                        prompt_text=contradiction_prompt,
                        project_id=guidelines_project_id if guidelines_project_id else None,
                        prompt_type="tender_contradiction",
                    )
                    if success:
                        st.session_state.tender_saved_contradiction_prompts[new_contradiction_name] = contradiction_prompt
                        st.success("Saved tender contradiction prompt.")
                        st.rerun()
                    else:
                        st.error("Failed to save tender contradiction prompt.")
                else:
                    st.error("Enter a name for the contradiction prompt.")
    
    with prompt_tabs[3]:
        st.markdown("**Orchestrator** - Synthesizes final compliance report")
        orchestrator_prompt = st.text_area(
            "Orchestrator Prompt Template",
            value=st.session_state.get("tender_orchestrator_prompt_editor", st.session_state.tender_orchestrator_prompt),
            height=300,
            help="Use {{tender_summary}}, {{omission_results}}, {{contradiction_results}} as placeholders",
            key="tender_orchestrator_prompt_editor"
        )
        st.session_state.tender_orchestrator_prompt = orchestrator_prompt
        if st.button("🔄 Reset to Default", key="reset_orchestrator_tender"):
            st.session_state.tender_orchestrator_prompt_editor = ORCHESTRATOR_PROMPT
            st.session_state.tender_orchestrator_prompt = ORCHESTRATOR_PROMPT
            st.rerun()
        
        # Manage saved Orchestrator prompts (tender-specific, compact UI)
        with st.expander("💾 Manage Orchestrator Prompts", expanded=False):
            if "tender_saved_orchestrator_prompts" not in st.session_state:
                st.session_state.tender_saved_orchestrator_prompts = {"Default Tender Orchestrator": ORCHESTRATOR_PROMPT}
                try:
                    db_prompts = st.session_state.supabase_client.get_all_prompts(prompt_type="tender_orchestrator")
                    for p in db_prompts:
                        st.session_state.tender_saved_orchestrator_prompts[p["name"]] = p["prompt_text"]
                except Exception:
                    pass
            orchestrator_names = list(st.session_state.tender_saved_orchestrator_prompts.keys())
            if orchestrator_names:
                selected_orchestrator = st.selectbox(
                    "📂 Load Saved",
                    options=orchestrator_names,
                    index=0,
                    key="tender_orchestrator_prompt_to_load",
                )
            else:
                selected_orchestrator = None
                st.selectbox(
                    "📂 Load Saved",
                    options=["No prompts available"],
                    index=0,
                    key="tender_orchestrator_prompt_to_load",
                    disabled=True,
                )
            load_col, update_col = st.columns(2)
            with load_col:
                if st.button("Load Selected", key="load_tender_orchestrator_prompt"):
                    if selected_orchestrator:
                        st.session_state.tender_orchestrator_prompt_to_apply = st.session_state.tender_saved_orchestrator_prompts[selected_orchestrator]
                        st.rerun()
            new_orchestrator_name = st.text_input(
                "New name",
                key="new_tender_orchestrator_name",
                placeholder="e.g., Tender Orchestrator V2",
            )
            with update_col:
                if st.button("Update Selected", key="update_tender_orchestrator_prompt"):
                    if selected_orchestrator:
                        success = st.session_state.supabase_client.save_prompt(
                            name=selected_orchestrator,
                            prompt_text=orchestrator_prompt,
                            project_id=reference_project_id if reference_project_id else None,
                            prompt_type="tender_orchestrator",
                        )
                        if success:
                            st.session_state.tender_saved_orchestrator_prompts[selected_orchestrator] = orchestrator_prompt
                            st.success("Updated tender orchestrator prompt.")
                            st.rerun()
                        else:
                            st.error("Failed to update tender orchestrator prompt.")
                    else:
                        st.error("Select a prompt to update.")
            if st.button("Save Current", key="save_tender_orchestrator_prompt"):
                if new_orchestrator_name:
                    success = st.session_state.supabase_client.save_prompt(
                        name=new_orchestrator_name,
                        prompt_text=orchestrator_prompt,
                        project_id=reference_project_id if reference_project_id else None,
                        prompt_type="tender_orchestrator",
                    )
                    if success:
                        st.session_state.tender_saved_orchestrator_prompts[new_orchestrator_name] = orchestrator_prompt
                        st.success("Saved tender orchestrator prompt.")
                        st.rerun()
                    else:
                        st.error("Failed to save tender orchestrator prompt.")
                else:
                    st.error("Enter a name for the orchestrator prompt.")
    
    st.markdown("---")
    
    # Document Upload
    st.subheader("📄 Upload Tender Document")
    
    uploaded_pdf = st.file_uploader(
        "Upload Tender PDF",
        type=["pdf"],
        help="Upload the tender submission document to check",
        key="tender_pdf_upload"
    )
    
    if uploaded_pdf:
        st.success(f"✓ PDF uploaded: {uploaded_pdf.name}")
    
    st.markdown("---")
    
    # Live progress: workflow runs in background thread; we poll and show steps
    STEP_LABELS = {
        1: "Breaking down tender",
        2: "Retrieving reference documents",
        3: "Checking requirements",
        4: "Synthesizing results",
    }

    def _render_live_progress(prog: Dict[str, Any]):
        """Render progress bar and step list from progress store (main thread only)."""
        prog = prog or {}
        step_num = prog.get("step_num", 0)
        total_steps = prog.get("total_steps", 4)
        details = prog.get("details") or {}
        progress_pct = int((step_num / total_steps) * 100) if total_steps else 0
        st.progress(min(progress_pct, 100) / 100.0)
        st.markdown("**Steps**")
        for i in range(1, total_steps + 1):
            label = STEP_LABELS.get(i, f"Step {i}")
            if i == step_num and i == 2 and details.get("step"):
                label = details["step"]
            if i < step_num:
                st.markdown(f"- ✅ **Step {i}/{total_steps}:** {label}")
            elif i == step_num:
                extra = ""
                if details.get("completed") is not None and details.get("total_requirements"):
                    extra = f" ({details['completed']}/{details['total_requirements']})"
                if details.get("current_requirement"):
                    extra += f" — *{details['current_requirement']}*"
                st.markdown(f"- 🔄 **Step {i}/{total_steps}:** {label}{extra}")
            else:
                st.markdown(f"- ⏳ **Step {i}/{total_steps}:** {label} (pending)")

    # Process Button: extract PDF, start workflow in thread, rerun to show progress
    if st.button("🚀 Check Tender Submission", type="primary", key="check_tender_btn"):
        if not uploaded_pdf:
            st.error("Please upload a PDF document first.")
            return

        pdf_bytes = uploaded_pdf.read()
        if not pdf_bytes:
            st.error("PDF file is empty.")
            return
        use_llamaparse = bool(LLAMA_CLOUD_API_KEY and LLAMA_CLOUD_API_KEY.strip())
        with st.spinner(
            "Parsing PDF with LlamaParse (layout + tables)..."
            if use_llamaparse
            else "Extracting text from PDF..."
        ):
            tender_text = None
            if use_llamaparse:
                class _PdfBytes:
                    read = lambda self: pdf_bytes
                    name = getattr(uploaded_pdf, "name", "document.pdf") or "document.pdf"
                tender_text = extract_text_from_pdf_with_llamaparse(_PdfBytes())
                if tender_text:
                    st.success(f"✓ LlamaParse: extracted {len(tender_text)} characters (layout-aware)")
            if not tender_text:
                tender_text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
                if tender_text:
                    st.success(f"✓ Extracted {len(tender_text)} characters from PDF")
            if not tender_text:
                st.error("Failed to extract text from PDF. Please try again.")
                return

        # Prompt sanity checks (prevents silent empty extractions)
        if "{{tender_text}}" not in st.session_state.tender_breakdown_prompt:
            st.error("Breakdown prompt is missing the {{tender_text}} placeholder.")
            st.info("Add {{tender_text}} to the Breakdown prompt so the model sees the uploaded document.")
            return
        if "requirements" not in st.session_state.tender_breakdown_prompt.lower():
            st.warning("Breakdown prompt does not mention 'requirements'. This can lead to empty outputs.")

        # Thread-safe progress and result holders (background thread must NOT touch st.session_state)
        progress_store: Dict[str, Any] = {}
        progress_lock = threading.Lock()
        result_holder: Dict[str, Any] = {"done": False, "result": None, "error": None}

        def progress_callback(step_name: str, step_num: int, total_steps: int, details: Dict[str, Any]):
            with progress_lock:
                progress_store.update({
                    "step_name": step_name,
                    "step_num": step_num,
                    "total_steps": total_steps,
                    "details": details,
                })

        pageindex_client = None
        if use_pageindex_chat and (PAGEINDEX_API_KEY or "").strip():
            try:
                pageindex_client = PageIndexClient(
                    api_key=PAGEINDEX_API_KEY.strip(),
                    chat_url=PAGEINDEX_CHAT_URL or "https://api.pageindex.ai/chat/completions",
                )
            except Exception as e:
                st.error(f"Failed to create PageIndex client: {str(e)}")
                return
        if use_pageindex_chat and not pageindex_client:
            st.error("PageIndex (Retrieval) selected but PAGEINDEX_API_KEY is not set. Set it in .env or switch to RAG.")
            return
        if use_pageindex_chat and not (reference_doc_id or "").strip():
            st.error("PageIndex (Retrieval) selected but Reference doc_id is empty. Enter a PageIndex doc_id.")
            return

        try:
            workflow = TenderCheckWorkflow(
                ai_client=st.session_state.ai_client,
                supabase_client=st.session_state.supabase_client,
                breakdown_prompt=st.session_state.tender_breakdown_prompt,
                omission_prompt=st.session_state.tender_omission_prompt,
                contradiction_prompt=st.session_state.tender_contradiction_prompt,
                orchestrator_prompt=st.session_state.tender_orchestrator_prompt,
                progress_callback=progress_callback,
                pageindex_client=pageindex_client,
            )
        except Exception as e:
            st.error(f"Failed to initialize workflow: {str(e)}")
            return

        st.session_state["tender_workflow_running"] = True
        st.session_state["tender_progress_store"] = progress_store
        st.session_state["tender_result_holder"] = result_holder
        _ref_id = reference_project_id
        _guid_id = guidelines_project_id if (not use_pageindex_chat and guidelines_project_id != reference_project_id) else None
        _ref_doc_id = (reference_doc_id or "").strip()
        _guid_doc_id = (guidelines_doc_id or "").strip()

        def run_workflow():
            try:
                result = workflow.run(
                    tender_text=tender_text,
                    project_id=_ref_id or "n/a",
                    guidelines_project_id=_guid_id,
                    top_k=top_k,
                    use_pageindex_chat=use_pageindex_chat,
                    reference_doc_id=_ref_doc_id,
                    guidelines_doc_id=_guid_doc_id,
                )
                result_holder["result"] = result
            except Exception as e:
                result_holder["error"] = str(e)
            finally:
                result_holder["done"] = True

        t = threading.Thread(target=run_workflow, daemon=True)
        t.start()
        st.rerun()

    # Poll: show live progress while workflow runs, then show result when done (main thread only)
    if st.session_state.get("tender_workflow_running"):
        result_holder = st.session_state.get("tender_result_holder") or {}
        if result_holder.get("done"):
            st.session_state["tender_workflow_running"] = False
            err = result_holder.get("error")
            if err:
                st.session_state["tender_workflow_error_display"] = err
            else:
                st.session_state["tender_check_result"] = result_holder.get("result")
            st.session_state.pop("tender_progress_store", None)
            st.session_state.pop("tender_result_holder", None)
            st.rerun()
        else:
            progress_store = st.session_state.get("tender_progress_store") or {}
            prog = dict(progress_store)  # snapshot for display (main thread only)
            _render_live_progress(prog)
            st.caption("Refreshing in 1 second…")
            time.sleep(1)
            st.rerun()

    # Show workflow error from previous run (after rerun)
    if st.session_state.get("tender_workflow_error_display"):
        st.error("Error during tender checking: " + st.session_state["tender_workflow_error_display"])
        with st.expander("Error Details"):
            st.code(st.session_state["tender_workflow_error_display"])
        st.session_state.pop("tender_workflow_error_display", None)

    # Display Results
    if "tender_check_result" in st.session_state:
        result = st.session_state.tender_check_result
        
        st.markdown("---")
        st.subheader("📊 Tender Check Results")
        
        # Overall Status
        final_report = result.get("final_report", {})
        overall_status = final_report.get("overall_status", "UNKNOWN")
        compliance_score = final_report.get("compliance_score", 0.0)
        
        # Status badge
        if overall_status == "COMPLIANT":
            st.success(f"✅ **Overall Status:** {overall_status} | **Compliance Score:** {compliance_score:.2%}")
        elif overall_status == "CONDITIONALLY_COMPLIANT":
            st.warning(f"⚠️ **Overall Status:** {overall_status} | **Compliance Score:** {compliance_score:.2%}")
        else:
            st.error(f"❌ **Overall Status:** {overall_status} | **Compliance Score:** {compliance_score:.2%}")
        
        # Summary
        st.markdown("### Executive Summary")
        st.info(final_report.get("summary", "No summary available"))
        
        # Requirements Breakdown
        requirements = result.get("requirements", [])
        requirements_by_id = {}
        if not requirements:
            st.warning(
                "No requirements were extracted. This usually means the Breakdown prompt is mis-specified. "
                "Ensure it instructs the model to extract requirements from the uploaded tender document "
                "and returns JSON with a `requirements` array."
            )
        if requirements:
            # Build lookup for later sections (e.g. Critical Issues)
            for i, req in enumerate(requirements, 1):
                req_id = req.get("id", f"REQ-{i}")
                requirements_by_id[req_id] = req

            st.markdown("### 📋 Extracted Requirements")
            st.write(f"**Total Requirements:** {len(requirements)}")
            
            # Filter and sort options
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_category = st.selectbox(
                    "Filter by Category",
                    options=["All"] + list(set(req.get("category", "N/A") for req in requirements)),
                    key="req_filter_category"
                )
            with col2:
                sort_by = st.selectbox(
                    "Sort by",
                    options=["ID", "Category"],
                    key="req_sort_by"
                )
            with col3:
                show_all = st.checkbox("Show all requirements", value=True, key="req_show_all")
            
            # Filter and sort requirements
            filtered_requirements = requirements
            if filter_category != "All":
                filtered_requirements = [r for r in filtered_requirements if r.get("category", "N/A") == filter_category]
            
            if sort_by == "Category":
                filtered_requirements = sorted(filtered_requirements, key=lambda x: x.get("category", "N/A"))
            
            # Display requirements - one expander per requirement
            display_count = len(filtered_requirements) if show_all else min(10, len(filtered_requirements))
            
            for i, req in enumerate(filtered_requirements[:display_count], 1):
                req_id = req.get("id", f"REQ-{i}")
                category = req.get("category", "N/A")
                full_text = req.get("requirement_text", "") or ""
                preview = full_text[:150] + "..." if len(full_text) > 150 else full_text
                
                with st.expander(f"**{req_id}** ({category})", expanded=False):
                    st.markdown(f"**Category:** {category}")
                    st.markdown("**Requirement Text:**")
                    st.write(full_text)
                    if req.get("context"):
                        st.markdown(f"**Context:** {req.get('context')}")
            
            if len(filtered_requirements) > display_count and not show_all:
                st.info(f"Showing {display_count} of {len(filtered_requirements)} requirements. Check 'Show all requirements' to see all.")
        
        # Omission Summary
        omission_summary = final_report.get("omission_summary", {})
        if omission_summary:
            st.markdown("### ✅ Omission Check Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", omission_summary.get("total_requirements", 0))
            with col2:
                st.metric("Fulfilled", omission_summary.get("fulfilled", 0), delta=None)
            with col3:
                st.metric("Partially Fulfilled", omission_summary.get("partially_fulfilled", 0))
            with col4:
                st.metric("Not Fulfilled", omission_summary.get("not_fulfilled", 0), delta=None)
            
            missing = omission_summary.get("missing_requirements", [])
            if missing:
                with st.expander("Missing Requirements", expanded=True):
                    for req in missing:
                        st.write(f"- {req}")
        
        # Contradiction Summary
        contradiction_summary = final_report.get("contradiction_summary", {})
        if contradiction_summary:
            st.markdown("### ⚠️ Contradiction Check Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Checked", contradiction_summary.get("total_checked", 0))
            with col2:
                st.metric("Critical", contradiction_summary.get("critical_contradictions", 0), delta=None, delta_color="inverse")
            with col3:
                st.metric("Moderate", contradiction_summary.get("moderate_contradictions", 0), delta=None, delta_color="inverse")
            with col4:
                st.metric("Minor", contradiction_summary.get("minor_contradictions", 0), delta=None, delta_color="inverse")
            
            contradictions = contradiction_summary.get("contradictions", [])
            if contradictions:
                with st.expander("Contradictions Found", expanded=True):
                    for cont in contradictions:
                        st.write(f"- {cont}")
        
        # Critical Issues
        critical_issues = final_report.get("critical_issues", [])
        omission_results = result.get("omission_results", [])
        contradiction_results = result.get("contradiction_results", [])

        # Build quick lookups for more granular details
        omission_by_id = {
            r.get("requirement_id"): r for r in omission_results if r.get("requirement_id")
        }
        contradiction_by_id = {
            r.get("requirement_id"): r for r in contradiction_results if r.get("requirement_id")
        }

        if critical_issues:
            st.markdown("### 🚨 Critical Issues")
            
            # Filter options for critical issues
            col1, col2 = st.columns(2)
            with col1:
                filter_severity = st.selectbox(
                    "Filter by Severity",
                    options=["All", "CRITICAL", "MODERATE", "MINOR"],
                    key="critical_filter_severity"
                )
            with col2:
                filter_issue_type = st.selectbox(
                    "Filter by Type",
                    options=["All", "OMISSION", "CONTRADICTION"],
                    key="critical_filter_type"
                )
            
            # Filter issues
            filtered_issues = critical_issues
            if filter_severity != "All":
                filtered_issues = [i for i in filtered_issues if i.get("severity") == filter_severity]
            if filter_issue_type != "All":
                filtered_issues = [i for i in filtered_issues if i.get("issue_type") == filter_issue_type]
            
            # Sort by severity (CRITICAL > MODERATE > MINOR)
            severity_order = {"CRITICAL": 0, "MODERATE": 1, "MINOR": 2}
            filtered_issues = sorted(
                filtered_issues,
                key=lambda x: (severity_order.get(x.get("severity", ""), 99), x.get("requirement_id", ""))
            )
            
            if not filtered_issues:
                st.info("No issues match the selected filters.")
            else:
                for issue in filtered_issues:
                    severity_color = {
                        "CRITICAL": "🔴",
                        "MODERATE": "🟡",
                        "MINOR": "🟢"
                    }.get(issue.get("severity", ""), "⚪")

                    issue_type = issue.get("issue_type", "N/A")
                    req_id = issue.get("requirement_id", "N/A")

                    st.markdown(f"{severity_color} **{req_id}** - {issue_type}")
                    st.caption(issue.get("description", ""))
                    st.caption(f"**Impact:** {issue.get('impact', 'N/A')}")

                    # Show the full underlying requirement text (if available)
                    requirement = requirements_by_id.get(req_id) if requirements else None
                    if requirement:
                        with st.expander("View full requirement", expanded=False):
                            st.write(requirement.get("requirement_text", ""))

                    # Surface more granular details from omission / contradiction agents
                    if issue_type == "OMISSION":
                        omission = omission_by_id.get(req_id)
                        if omission:
                            st.markdown("**Omission details**")
                            st.write(
                                f"Status: {omission.get('status', 'UNKNOWN')} "
                                f"(confidence: {omission.get('confidence', 0.0):.2f})"
                            )
                            missing = omission.get("missing_elements") or []
                            if missing:
                                st.write("Missing elements:")
                                for item in missing:
                                    st.write(f"- {item}")
                            citations = omission.get("citations") or []
                            if citations:
                                with st.expander("Supporting citations from reference documents", expanded=False):
                                    for cit in citations:
                                        doc_ref = cit.get("document_reference", "Unknown document")
                                        src = cit.get("source_text", "")
                                        st.markdown(f"- *{doc_ref}*: {src}")

                    elif issue_type == "CONTRADICTION":
                        contradiction = contradiction_by_id.get(req_id)
                        if contradiction:
                            st.markdown("**Contradiction details**")
                            st.write(f"Severity: {contradiction.get('severity', 'NO_CONTRADICTION')}")
                            details = contradiction.get("contradiction_details", "")
                            if details:
                                st.write(details)
                            ref_guideline = contradiction.get("reference_guideline")
                            if ref_guideline:
                                st.write(f"Reference guideline: {ref_guideline}")
                            tender_stmt = contradiction.get("tender_statement")
                            if tender_stmt:
                                st.write(f"Tender statement: {tender_stmt}")
                            citations = contradiction.get("citations") or []
                            if citations:
                                with st.expander("Supporting citations from guidelines", expanded=False):
                                    for cit in citations:
                                        doc_ref = cit.get("document_reference", "Unknown document")
                                        src = cit.get("source_text", "")
                                        st.markdown(f"- *{doc_ref}*: {src}")

        # Recommendations
        recommendations = final_report.get("recommendations", [])
        if recommendations:
            st.markdown("### 💡 Recommendations")
            for rec in recommendations:
                priority_icon = {
                    "HIGH": "🔴",
                    "MEDIUM": "🟡",
                    "LOW": "🟢"
                }.get(rec.get("priority", ""), "⚪")
                
                st.markdown(f"{priority_icon} **{rec.get('requirement_id', 'N/A')}** - {rec.get('action', '')}")
        
        # Risk Assessment
        risk_assessment = final_report.get("risk_assessment", "")
        if risk_assessment:
            st.markdown("### 📊 Risk Assessment")
            st.info(risk_assessment)
        
        # Detailed Results
        with st.expander("📄 View Detailed Results (JSON)", expanded=False):
            st.json(result)
        
        # Download Results
        st.markdown("---")
        st.subheader("📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            result_json = json.dumps(result, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=result_json,
                file_name="tender_check_results.json",
                mime="application/json",
                key="download_tender_results_json",
                use_container_width=True
            )
        
        with col2:
            # Generate HTML report
            html_report = generate_html_report(final_report, requirements, critical_issues, omission_results, contradiction_results)
            st.download_button(
                label="📄 Download HTML Report",
                data=html_report,
                file_name="tender_check_report.html",
                mime="text/html",
                key="download_tender_results_html",
                use_container_width=True
            )
        
        with col3:
            # Generate text summary
            text_summary = generate_text_summary(final_report, critical_issues)
            st.download_button(
                label="📝 Download Text Summary",
                data=text_summary,
                file_name="tender_check_summary.txt",
                mime="text/plain",
                key="download_tender_results_text",
                use_container_width=True
            )


def generate_html_report(
    final_report: Dict[str, Any],
    requirements: List[Dict[str, Any]],
    critical_issues: List[Dict[str, Any]],
    omission_results: List[Dict[str, Any]],
    contradiction_results: List[Dict[str, Any]]
) -> str:
    """Generate HTML report for tender check results."""
    from datetime import datetime
    overall_status = final_report.get("overall_status", "UNKNOWN")
    compliance_score = final_report.get("compliance_score", 0.0)
    summary = final_report.get("summary", "No summary available")
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Tender Check Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .status-compliant {{ color: #27ae60; font-weight: bold; }}
        .status-non-compliant {{ color: #e74c3c; font-weight: bold; }}
        .status-conditional {{ color: #f39c12; font-weight: bold; }}
        .issue-critical {{ background-color: #fee; padding: 10px; margin: 10px 0; border-left: 4px solid #e74c3c; }}
        .issue-moderate {{ background-color: #fff8e1; padding: 10px; margin: 10px 0; border-left: 4px solid #f39c12; }}
        .issue-minor {{ background-color: #f1f8e9; padding: 10px; margin: 10px 0; border-left: 4px solid #8bc34a; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        .summary-box {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Tender Compliance Check Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="summary-box">
        <h2>Executive Summary</h2>
        <p><strong>Overall Status:</strong> <span class="status-{overall_status.lower().replace('_', '-')}">{overall_status}</span></p>
        <p><strong>Compliance Score:</strong> {compliance_score:.2%}</p>
        <p>{summary}</p>
    </div>
    
    <h2>Critical Issues</h2>
    <p><strong>Total Issues:</strong> {len(critical_issues)}</p>
"""
    
    for issue in critical_issues:
        severity = issue.get("severity", "UNKNOWN")
        issue_class = f"issue-{severity.lower()}"
        html += f"""
    <div class="{issue_class}">
        <h3>{issue.get('requirement_id', 'N/A')} - {issue.get('issue_type', 'N/A')}</h3>
        <p><strong>Severity:</strong> {severity}</p>
        <p><strong>Description:</strong> {issue.get('description', 'N/A')}</p>
        <p><strong>Impact:</strong> {issue.get('impact', 'N/A')}</p>
    </div>
"""
    
    html += f"""
    <h2>Requirements Summary</h2>
    <p><strong>Total Requirements:</strong> {len(requirements)}</p>
    
    <h2>Omission Check Summary</h2>
    <table>
        <tr>
            <th>Requirement ID</th>
            <th>Status</th>
            <th>Confidence</th>
        </tr>
"""
    
    for om_result in omission_results[:20]:  # Limit to first 20 for readability
        html += f"""
        <tr>
            <td>{om_result.get('requirement_id', 'N/A')}</td>
            <td>{om_result.get('status', 'N/A')}</td>
            <td>{om_result.get('confidence', 0.0):.2%}</td>
        </tr>
"""
    
    html += """
    </table>
    
    <h2>Contradiction Check Summary</h2>
    <table>
        <tr>
            <th>Requirement ID</th>
            <th>Has Contradiction</th>
            <th>Severity</th>
        </tr>
"""
    
    for con_result in contradiction_results[:20]:  # Limit to first 20
        html += f"""
        <tr>
            <td>{con_result.get('requirement_id', 'N/A')}</td>
            <td>{'Yes' if con_result.get('has_contradiction') else 'No'}</td>
            <td>{con_result.get('severity', 'N/A')}</td>
        </tr>
"""
    
    html += """
    </table>
</body>
</html>
"""
    return html


def generate_text_summary(final_report: Dict[str, Any], critical_issues: List[Dict[str, Any]]) -> str:
    """Generate plain text summary report."""
    from datetime import datetime
    overall_status = final_report.get("overall_status", "UNKNOWN")
    compliance_score = final_report.get("compliance_score", 0.0)
    summary = final_report.get("summary", "No summary available")
    
    text = f"""
TENDER COMPLIANCE CHECK REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

EXECUTIVE SUMMARY
=================
Overall Status: {overall_status}
Compliance Score: {compliance_score:.2%}

{summary}

CRITICAL ISSUES
===============
Total Issues: {len(critical_issues)}

"""
    
    for i, issue in enumerate(critical_issues, 1):
        text += f"""
Issue {i}: {issue.get('requirement_id', 'N/A')} - {issue.get('issue_type', 'N/A')}
Severity: {issue.get('severity', 'N/A')}
Description: {issue.get('description', 'N/A')}
Impact: {issue.get('impact', 'N/A')}

"""
    
    return text


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Requirement Checker",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 Requirement Checker")
    st.markdown("---")
    
    # Project ID selection (global setting)
    selected_project = st.sidebar.selectbox(
        "Project",
        options=list(PROJECT_IDS.keys()),
        index=0,  # Default to "Building services"
        help="Select the project to check requirements against documents in Supabase",
        key="project_selection"
    )
    
    # Get the actual project ID from the selection
    project_id = PROJECT_IDS[selected_project]
    
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Single Requirement", 
        "📊 CSV Batch Processing", 
        "✂️ Requirement Breakdown",
        "📄 Tender Checker"
    ])
    
    with tab1:
        show_single_requirement_tab(project_id, top_k)
    
    with tab2:
        show_csv_batch_tab(project_id, top_k)

    with tab3:
        show_breakdown_tab(project_id)
    
    with tab4:
        show_tender_check_tab()


if __name__ == "__main__":
    main()

