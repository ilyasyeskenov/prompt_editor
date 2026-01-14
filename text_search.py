"""
Text Search Backend Functions
Replicates the text-search functionality from the TypeScript implementation.
Uses Supabase client (same as TypeScript), with optional direct PostgreSQL for score retrieval.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Try importing psycopg2 for direct PostgreSQL queries (optional)
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class TextSearchBackend:
    """
    Backend functions for text search functionality.
    Uses Supabase client for basic queries, and direct PostgreSQL for score retrieval.
    """
    
    def __init__(
        self, 
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        database_url: Optional[str] = None
    ):
        """
        Initialize Supabase client.
        
        Args:
            supabase_url: Supabase project URL. If None, reads from NEXT_PUBLIC_SUPABASE_URL env var.
            supabase_key: Supabase anon key. If None, reads from SUPABASE_ANON_KEY env var.
            database_url: Direct PostgreSQL connection string. If None, reads from DATABASE_URL env var.
                          Used for getting text search scores via direct SQL queries.
        """
        self.supabase_url = supabase_url or os.getenv('NEXT_PUBLIC_SUPABASE_URL') or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_ANON_KEY')
        self.database_url = database_url or os.getenv('DATABASE_URL')
        
        if not self.supabase_url:
            raise ValueError("NEXT_PUBLIC_SUPABASE_URL/SUPABASE_URL environment variable or supabase_url parameter is required")
    
        if not self.supabase_key:
            raise ValueError("SUPABASE_ANON_KEY environment variable or supabase_key parameter is required")
        
        # Create Supabase client (same as TypeScript: createClient(supabaseUrl, supabaseAnonKey, {}))
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
    
    def get_chunks_by_text_search(
        self, 
        query: str, 
        project_id: str
    ) -> List[Dict[str, str]]:
        """
        Search document chunks using full-text search within a specific project.
        
        This replicates the TypeScript function:
        getChunksByTextSearch(query: string, projectId: string)
        
        Args:
            query: Search query string
            project_id: Project ID to filter chunks
            
        Returns:
            List of dictionaries with 'content' field
            
        Raises:
            ValueError: If query or project_id is invalid
            Exception: If Supabase query fails
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query parameter is required and must be a string")
        
        if not project_id or not isinstance(project_id, str):
            raise ValueError("Project ID parameter is required and must be a string")
        
        try:
            # Replicates: supabase.schema('knowledge').from('document_chunks')
            #            .select('content').eq('project_id', projectId).textSearch('content', query)
            response = (
                self.supabase
                .schema('knowledge')
                .table('document_chunks')
                .select('content')
                .eq('project_id', project_id)
                .text_search('content', query)
                .execute()
            )
            
            # Supabase Python client returns response.data
            return response.data if response.data else []
            
        except Exception as e:
            raise Exception(f"Supabase query failed: {str(e)}")
    
    def get_chunks_by_text_search_and_project_id(
        self,
        query: str,
        project_id: str,
        limit: int = 5
    ) -> List[Dict]:

            
        """Search chunks by project ID with full metadata."""
        if not query or not isinstance(query, str):
            raise ValueError("Query parameter is required and must be a string")
        
        if not project_id or not isinstance(project_id, str):
            raise ValueError("Project ID parameter is required and must be a string")
        
        try:
            # Try with websearch type - if that doesn't work, fall back to direct SQL
            try:
                response = (
                    self.supabase
                    .schema('knowledge')
                    .table('document_chunks')
                    .select('id, content, task_id, page_number, metadata, chunk_id, chunkr_tasks!inner(file_name, original_filename)')
                    .eq('project_id', project_id)
                    .text_search('content', query, {'type': 'websearch'})  # Try dict syntax
                    .execute()
                )
                data = response.data if response.data else []
                return data[:limit]
            except Exception:
                # Fallback: Use direct PostgreSQL if Supabase client doesn't support websearch
                if PSYCOPG2_AVAILABLE and self.database_url:
                    return self.get_chunks_by_text_search_with_scores(query, project_id, limit)
                else:
                    # Last resort: use plainto_tsquery compatible format
                    # Replace spaces with & for AND operator
                    formatted_query = ' & '.join(query.split())
                    response = (
                        self.supabase
                        .schema('knowledge')
                        .table('document_chunks')
                        .select('id, content, task_id, page_number, metadata, chunk_id, chunkr_tasks!inner(file_name, original_filename)')
                        .eq('project_id', project_id)
                        .text_search('content', formatted_query)
                        .execute()
                    )
                    data = response.data if response.data else []
                    return data[:limit]
                    
        except Exception as e:
            raise Exception(f"Supabase query failed: {str(e)}")
    
    def get_chunks_by_text_search_with_scores(
        self,
        query: str,
        project_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search chunks by project ID with full metadata AND text search scores.
        Uses direct PostgreSQL queries to get ts_rank() scores.
        
        Args:
            query: Search query string
            project_id: Project ID to filter chunks
            limit: Maximum number of results (default: 5)
            
        Returns:
            List of dictionaries with full chunk metadata including 'text_score' field
            
        Raises:
            ValueError: If query or project_id is invalid, or if psycopg2/DATABASE_URL not available
            Exception: If database query fails
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query parameter is required and must be a string")
        
        if not project_id or not isinstance(project_id, str):
            raise ValueError("Project ID parameter is required and must be a string")
        
        if not PSYCOPG2_AVAILABLE:
            raise ValueError("psycopg2 is required for this method. Install it with: pip install psycopg2-binary")
        
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable or database_url parameter is required for this method")
        
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Query with full metadata including file names and text search scores
            # Uses ts_rank() to get relevance scores from PostgreSQL full-text search
            sql = """
                SELECT 
                    dc.id,
                    dc.content,
                    dc.task_id,
                    dc.page_number,
                    dc.metadata,
                    dc.chunk_id,
                    ct.file_name,
                    ct.original_filename,
                    ts_rank(to_tsvector('english', dc.content), websearch_to_tsquery('english', %s)) as text_score
                FROM knowledge.document_chunks dc
                INNER JOIN knowledge.chunkr_tasks ct ON dc.task_id = ct.task_id
                WHERE dc.project_id = %s
                AND to_tsvector('english', dc.content) @@ websearch_to_tsquery('english', %s)
                ORDER BY text_score DESC
                LIMIT %s
            """
            
            cursor.execute(sql, (query, project_id, query, limit))
            results = cursor.fetchall()
            
            # Convert to list of dictionaries
            return [dict(row) for row in results]
            
        except psycopg2.Error as e:
            raise Exception(f"Database query failed: {str(e)}")
        finally:
            if conn:
                cursor.close()
                conn.close()
