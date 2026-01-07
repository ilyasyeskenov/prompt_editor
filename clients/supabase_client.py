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

