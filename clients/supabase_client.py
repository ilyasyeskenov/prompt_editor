"""Supabase client for RAG operations - matches original TypeScript implementation."""
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
from config.config import SUPABASE_URL, SUPABASE_ANON_KEY
from clients.ai_client import AIClient


class SupabaseClient:
    """Supabase client for document chunk operations."""
    
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env file.")
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        self.ai_client = AIClient()
    
    def get_chunks_by_knowledge_id(self, knowledge_id: str) -> str:
        """
        Get all chunks for a knowledge/document ID.
        Matches: lib/supabase.ts getChunksByKnowledgeId
        
        Note: The Python Supabase client may require the schema to be specified
        in the table name or via connection options. If the table is in the 'knowledge'
        schema, you may need to use 'knowledge.document_chunks' as the table name.
        """
        try:
            # Try with schema prefix first (knowledge.document_chunks)
            # If that doesn't work, try without schema (assuming default schema)
            try:
                response = self.client.table('knowledge.document_chunks')\
                    .select('content, chunkr_tasks!inner(knowledge_id)')\
                    .eq('chunkr_tasks.knowledge_id', knowledge_id)\
                    .order('chunk_index')\
                    .execute()
            except:
                # Fallback: try without schema prefix
                response = self.client.table('document_chunks')\
                    .select('content, chunkr_tasks!inner(knowledge_id)')\
                    .eq('chunkr_tasks.knowledge_id', knowledge_id)\
                    .order('chunk_index')\
                    .execute()
            
            if response.data:
                # Extract content from each chunk and join
                chunks = [chunk.get('content', '') for chunk in response.data if chunk.get('content')]
                return '\n'.join(chunks)
            return ''
        except Exception as e:
            raise Exception(f"Error fetching chunks by knowledge_id: {str(e)}")
    
    def get_chunks_by_project_id(
        self,
        project_id: str,
        query_embedding: List[float],
        match_count: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Get chunks by project ID using vector similarity search.
        Matches: lib/supabase.ts getChunksByProjectId
        Uses RPC function: match_document_chunks_meta (in public schema)
        """
        try:
            # RPC function is in public schema, so no schema prefix needed
            response = self.client.rpc(
                'match_document_chunks_meta',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.1,
                    'match_count': match_count,
                    'project_id_param': project_id
                }
            ).execute()
            
            if response.data:
                # Transform to match the expected format
                chunks = []
                for chunk in response.data:
                    chunks.append({
                        'id': chunk.get('id', ''),
                        'content': chunk.get('content', ''),
                        'task_id': chunk.get('task_id', ''),
                        'similarity': chunk.get('similarity', 0.0),
                        'file_name': chunk.get('file_name') or chunk.get('original_filename'),
                        'original_filename': chunk.get('original_filename'),
                        'page_number': chunk.get('page_number'),
                        'metadata': chunk.get('metadata')
                    })
                return chunks
            return []
        except Exception as e:
            raise Exception(f"Error fetching chunks by project_id: {str(e)}")
    
    def search_chunks_by_project_id(
        self,
        project_id: str,
        query: str,
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Search chunks by project ID - generates embedding and searches.
        Convenience method that combines embedding generation and search.
        """
        # Generate embedding for query
        query_embedding = self.ai_client.generate_embedding(query)
        
        # Search using the RPC function
        return self.get_chunks_by_project_id(
            project_id=project_id,
            query_embedding=query_embedding,
            match_count=top_k
        )

    def get_chunks_by_hybrid_search(
        self,
        project_id: str,
        query: str,
        match_count: int = 10,
        full_text_weight: float = 2.0,
        semantic_weight: float = 1.0,
        rrf_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get chunks using database-level hybrid search with RRF.
        
        Args:
            project_id: Project ID to filter chunks (can be UUID or text)
            query: Search query string
            match_count: Number of results to return
            full_text_weight: Weight for full-text search (default: 1.0)
            semantic_weight: Weight for semantic search (default: 1.0)
            rrf_k: RRF smoothing constant (default: 50)
        
        Returns:
            List of chunk dictionaries sorted by RRF score
        """
        try:
            # Generate embedding for query
            query_embedding = self.ai_client.generate_embedding(query)
            
            # Pass project_id as-is (database function should handle TEXT/UUID)
            # If the function still expects UUID, it will error and we'll catch it
            project_id_param = project_id if project_id else None
            
            # Call hybrid_search RPC function (in public schema)
            response = self.client.rpc(
                'hybrid_search',
                {
                    'query_text': query,
                    'query_embedding': query_embedding,
                    'project_id_param': project_id_param,
                    'match_count': match_count,
                    'full_text_weight': full_text_weight,
                    'semantic_weight': semantic_weight,
                    'rrf_k': rrf_k
                }
            ).execute()
            
            if response.data:
                # Transform to match expected format
                chunks = []
                for chunk in response.data:
                    chunks.append({
                        'id': chunk.get('id', ''),
                        'content': chunk.get('content', ''),
                        'task_id': chunk.get('task_id', ''),
                        'file_name': chunk.get('file_name') or chunk.get('original_filename'),
                        'original_filename': chunk.get('original_filename'),
                        'page_number': chunk.get('page_number'),
                        'metadata': chunk.get('metadata'),
                        'rrf_score': chunk.get('rrf_score'),
                        'vector_rank': chunk.get('vector_rank'),
                        'text_rank': chunk.get('text_rank'),
                        'similarity': chunk.get('similarity'),
                        'project_id': chunk.get('project_id'),
                        'organization_id': chunk.get('organization_id'),
                    })
                return chunks
            return []
        except Exception as e:
            raise Exception(f"Error in hybrid search: {str(e)}")

    def get_all_prompts(self, prompt_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all saved prompt templates from Supabase, optionally filtered by type."""
        try:
            query = self.client.table('prompt_templates').select('*')
            if prompt_type:
                try:
                    query = query.eq('prompt_type', prompt_type)
                    response = query.order('created_at').execute()
                    return response.data or []
                except Exception as e:
                    # Column doesn't exist yet, fall back to loading all prompts
                    error_str = str(e)
                    if 'prompt_type' in error_str and 'does not exist' in error_str:
                        print(f"Note: prompt_type column doesn't exist yet, loading all prompts instead")
                        # Fall through to load all prompts without filtering
                    else:
                        raise
            
            # Load all prompts (either no filter requested, or column doesn't exist)
            response = query.order('created_at').execute()
            return response.data or []
        except Exception as e:
            print(f"Error fetching prompts: {str(e)}")
            return []

    def save_prompt(self, name: str, prompt_text: str, project_id: Optional[str] = None, prompt_type: Optional[str] = None) -> bool:
        """Save or update a prompt template in Supabase."""
        try:
            # Check if prompt with this name already exists
            existing = self.client.table('prompt_templates').select('id').eq('name', name).execute()
            
            data = {
                'name': name,
                'prompt_text': prompt_text,
                'project_id': project_id,
                'updated_at': 'now()'
            }
            
            # Try to add prompt_type if provided, but don't fail if column doesn't exist
            if prompt_type:
                try:
                    data['prompt_type'] = prompt_type
                except Exception:
                    pass  # Column might not exist, continue without it
            
            if existing.data:
                # Update - try with prompt_type, fall back without it if column doesn't exist
                try:
                    self.client.table('prompt_templates').update(data).eq('id', existing.data[0]['id']).execute()
                except Exception as e:
                    error_str = str(e)
                    if 'prompt_type' in error_str and 'does not exist' in error_str:
                        # Column doesn't exist, save without prompt_type
                        data_without_type = {k: v for k, v in data.items() if k != 'prompt_type'}
                        self.client.table('prompt_templates').update(data_without_type).eq('id', existing.data[0]['id']).execute()
                    else:
                        raise
            else:
                # Insert - try with prompt_type, fall back without it if column doesn't exist
                try:
                    self.client.table('prompt_templates').insert(data).execute()
                except Exception as e:
                    error_str = str(e)
                    if 'prompt_type' in error_str and 'does not exist' in error_str:
                        # Column doesn't exist, save without prompt_type
                        data_without_type = {k: v for k, v in data.items() if k != 'prompt_type'}
                        self.client.table('prompt_templates').insert(data_without_type).execute()
                    else:
                        raise
            return True
        except Exception as e:
            print(f"Error saving prompt: {str(e)}")
            return False

    # --- PageIndex submission cache (content_hash -> doc_id to avoid re-uploading same PDF) ---
    TABLE_PAGEINDEX_CACHE = "pageindex_submission_cache"

    def get_pageindex_doc_id_by_hash(self, content_hash: str) -> Optional[str]:
        """
        Look up a PageIndex doc_id by SHA-256 content hash of the PDF.
        Returns None if not found or if the cache table is not available.
        """
        if not (content_hash or "").strip():
            return None
        try:
            response = (
                self.client.table(self.TABLE_PAGEINDEX_CACHE)
                .select("doc_id")
                .eq("content_hash", content_hash.strip())
                .limit(1)
                .execute()
            )
            if response.data and len(response.data) > 0:
                return (response.data[0].get("doc_id") or "").strip() or None
            return None
        except Exception:
            return None

    def save_pageindex_submission_cache(
        self,
        content_hash: str,
        doc_id: str,
        filename: Optional[str] = None,
    ) -> bool:
        """
        Store content_hash -> doc_id for future reuse.
        Uses upsert on content_hash so the same hash updates doc_id/created_at.
        Returns True on success, False on failure (e.g. table does not exist).
        """
        if not (content_hash or "").strip() or not (doc_id or "").strip():
            return False
        try:
            row = {
                "content_hash": content_hash.strip(),
                "doc_id": doc_id.strip(),
            }
            if filename is not None:
                row["filename"] = (filename or "").strip() or None
            self.client.table(self.TABLE_PAGEINDEX_CACHE).upsert(
                row, on_conflict="content_hash"
            ).execute()
            return True
        except Exception:
            return False

