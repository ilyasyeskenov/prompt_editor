"""Session state initialization and management."""
import streamlit as st
import traceback
from clients.supabase_client import SupabaseClient
from clients.ai_client import AIClient
from utils.prompts import DEFAULT_PROMPT, DEFAULT_BREAKDOWN_PROMPT
from utils.logging import log


def initialize_session_state():
    """Initialize all session state variables."""
    # Initialize Supabase client
    needs_init = (
        'supabase_client' not in st.session_state or 
        not hasattr(st.session_state.supabase_client, 'save_prompt') or
        not hasattr(st.session_state.supabase_client, 'get_chunks_by_hybrid_search')
    )

    if needs_init:
        log("A", "session_state.py", "Initializing SupabaseClient", {})
        try:
            st.session_state.supabase_client = SupabaseClient()
            log("A", "session_state.py", "SupabaseClient initialized successfully", {})
        except Exception as e:
            log("A", "session_state.py", "SupabaseClient initialization failed", {"error": str(e), "traceback": traceback.format_exc()})
            raise

    # Initialize AI client
    if 'ai_client' not in st.session_state:
        log("A", "session_state.py", "Initializing AIClient", {})
        try:
            st.session_state.ai_client = AIClient()
            log("A", "session_state.py", "AIClient initialized successfully", {})
        except Exception as e:
            log("A", "session_state.py", "AIClient initialization failed", {"error": str(e), "traceback": traceback.format_exc()})
            raise

    # Initialize prompt templates
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = DEFAULT_PROMPT

    if 'current_breakdown_prompt' not in st.session_state:
        st.session_state.current_breakdown_prompt = DEFAULT_BREAKDOWN_PROMPT

    # Load saved prompts from Supabase
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

    # Load breakdown prompts
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

