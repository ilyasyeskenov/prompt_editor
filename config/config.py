"""Configuration settings for the compliance checker."""
import os
from dotenv import load_dotenv

load_dotenv()

# AI Provider: "openai" | "gemini" (default openai = GPT-4o; set AI_PROVIDER=gemini to use Gemini)
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").strip().lower() or "openai"

# OpenAI Configuration (used when AI_PROVIDER=openai, and for embeddings when provider=gemini)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# Gemini Configuration (used when AI_PROVIDER=gemini for chat)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")  # or "gemini-3-flash-preview" when available

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

# LlamaParse (LlamaCloud) - optional; when set, tender checker uses it for PDF extraction
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

# PageIndex - optional; when set, tender checker can use PageIndex Chat API for "answer over doc"
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_CHAT_URL = os.getenv("PAGEINDEX_CHAT_URL", "https://api.pageindex.ai/chat/completions")

# Note: ACCREDITED_LABORATORIES removed - was only used in deleted prompts.py

# Processing Configuration
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
BATCH_SIZE = 3  # Number of requirements to process in parallel
TOP_K_CHUNKS = 5  # Default number of document chunks to retrieve per requirement

# Data Storage
DATA_DIR = "data"
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
PROJECTS_DIR = os.path.join(DATA_DIR, "projects")

# Project IDs: display name -> UUID (used in sidebar and Tender Checker tab)
PROJECT_IDS = {
    "Building services": "fda85e04-3a9c-4e6f-8af0-35bfcb1ba4e0",
    "Tender Requirement (Annex1)": "00d06bf0-7572-4f23-aaec-46a9adad5e63",
    "Tender_handbook": "05ac317c-e700-4d5b-a99e-a1a92fc619e5"
}

