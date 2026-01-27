"""Configuration settings for the compliance checker."""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

# Note: ACCREDITED_LABORATORIES removed - was only used in deleted prompts.py

# Processing Configuration
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
BATCH_SIZE = 3  # Number of requirements to process in parallel
TOP_K_CHUNKS = 8  # Number of document chunks to retrieve per requirement

# Data Storage
DATA_DIR = "data"
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
PROJECTS_DIR = os.path.join(DATA_DIR, "projects")

