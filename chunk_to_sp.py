import os
import re
import json
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
import tiktoken

load_dotenv()

class ChunkrToSupabase:
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        self.supabase_url = supabase_url or os.getenv('NEXT_PUBLIC_SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_ANON_KEY')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase credentials required")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self.openai = OpenAI(api_key=self.openai_api_key)
        self.encoding = tiktoken.encoding_for_model('text-embedding-3-large')
    
    def clean_embed_text(self, text: str) -> str:
        """Clean embed text by removing base64 images and other noise."""
        if not text:
            return ''
        
        cleaned_text = text
        
        # Remove base64 image data URLs
        cleaned_text = re.sub(
            r'data:image/[^;]+;base64,[a-zA-Z0-9+/=]+',
            '[IMAGE REMOVED]',
            cleaned_text
        )
        
        # Remove any remaining <img> tags
        cleaned_text = re.sub(r'<img[^>]*>', '[IMAGE REMOVED]', cleaned_text, flags=re.IGNORECASE)
        
        # Remove sequences of more than 10 consecutive empty table data cells
        cleaned_text = re.sub(
            r'(<td[^>]*>\s*</td>\s*){11,}',
            '[EMPTY_TABLE_CELLS_REMOVED]',
            cleaned_text,
            flags=re.IGNORECASE
        )
        
        # Remove alphanumeric strings longer than 60 characters
        def replace_long_string(match):
            return '[LONG_STRING_REMOVED]' if len(match.group()) > 99 else match.group()
        cleaned_text = re.sub(r'\b[a-zA-Z0-9]{100,}\b', replace_long_string, cleaned_text)
        
        # Check token length and truncate if needed
        tokens = self.encoding.encode(cleaned_text)
        token_length = len(tokens)
        
        if token_length > 8000:
            truncated_tokens = tokens[:8000]
            truncated_text = self.encoding.decode(truncated_tokens)
            cleaned_text = truncated_text
        
        return cleaned_text
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Optional[List[float]]]:
        """Generate embeddings for a batch of chunks using OpenAI."""
        if not chunks:
            return []
        
        results: List[Optional[List[float]]] = []
        BATCH_SIZE = 25
        
        try:
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                
                # Clean text once and cache it
                for chunk in batch:
                    if 'cleaned_embed' not in chunk:
                        chunk['cleaned_embed'] = self.clean_embed_text(chunk.get('embed', ''))
                
                valid_inputs = [
                    chunk['cleaned_embed']
                    for chunk in batch
                    if chunk.get('cleaned_embed', '').strip() != ''
                ]
                
                if valid_inputs:
                    response = self.openai.embeddings.create(
                        model='text-embedding-3-large',
                        input=valid_inputs,
                        dimensions=1024
                    )
                    
                    # Map embeddings back to chunks
                    embeddings_array = [d.embedding for d in response.data]
                    valid_index = 0
                    batch_embeddings = []
                    for chunk in batch:
                        if chunk.get('cleaned_embed', '').strip() != '':
                            batch_embeddings.append(embeddings_array[valid_index] if valid_index < len(embeddings_array) else None)
                            valid_index += 1
                        else:
                            batch_embeddings.append(None)
                    
                    results.extend(batch_embeddings)
                else:
                    results.extend([None] * len(batch))
            
            return results
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [None] * len(chunks)
    
    def parse_task_data_batch(
        self,
        chunk_batch: List[Dict],
        start_index: int,
        task_id: str,
        knowledge: Dict
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process a batch of chunks into structured documents."""
        processed_chunks = []
        all_segments = []
        
        for i, chunk in enumerate(chunk_batch):
            if not chunk.get('segments'):
                continue
            
            chunk_index = start_index + i
            chunk_id = chunk.get('chunk_id') or f'chunk_{chunk_index}'
            first_segment = chunk['segments'][0]
            
            processed_chunks.append({
                'id': chunk_id,
                'task_id': task_id,
                'chunk_index': chunk_index,
                'project_id': knowledge.get('project_id'),
                'organization_id': knowledge.get('organization_id'),
                'file_name': knowledge.get('file_name'),  # Use file_name from knowledge dict
                'content': chunk.get('content'),
                'embed': chunk.get('embed'),
                'primary_page_number': first_segment.get('page_number'),
                'total_segments': len(chunk['segments']),
            })
            
            for segment_index, segment in enumerate(chunk['segments']):
                bbox = segment.get('bbox') or {}
                all_segments.append({
                    'chunk_id': chunk_id,
                    'segment_index': segment_index,
                    'segment_id': segment.get('segment_id'),
                    'segment_type': str(segment.get('segment_type', '')),
                    'page_number': segment.get('page_number'),
                    'bbox': bbox if bbox else None,
                    'content': segment.get('content'),
                    'html': segment.get('html'),
                    'markdown': segment.get('markdown'),
                    'image': segment.get('image'),
                    'page_height': segment.get('page_height'),
                    'page_width': segment.get('page_width'),
                    'confidence': segment.get('confidence') or segment.get('ocr', [{}])[0].get('confidence') if segment.get('ocr') else None,
                })
        
        return processed_chunks, all_segments
    
    def is_base64_image(self, s: Optional[str]) -> bool:
        """Check if a string is a base64 image data URL."""
        if not s:
            return False
        return s.startswith('data:image/') or len(s) > 50000
    
    def truncate_large_content(self, content: Optional[str], max_length: int = 50000) -> Optional[str]:
        """Truncate large content to avoid payload size issues."""
        if not content:
            return None
        if len(content) <= max_length:
            return content
        return content[:max_length] + '... [TRUNCATED]'
    
    def create_chunkr_task_record(
        self,
        task_id: str,
        task_json: Dict,
        knowledge: Dict
    ) -> None:
        """Create a record in chunkr_tasks table before saving chunks."""
        file_info = task_json.get('file_info', {}) or {}
        
        # Only include fields that exist in the schema
        task_record = {
            'task_id': task_id,  # Required for foreign key
        }

        # Add optional fields that might exist
        optional_fields = {
            'status': task_json.get('status', 'Succeeded'),
            'file_name': knowledge.get('file_name'),
            'original_filename': file_info.get('name'),
            'organization_id': knowledge.get('organization_id'),
            'project_id': knowledge.get('project_id'),
            # Removed timestamp fields: 'created_at', 'finished_at', 'started_at'
            # These columns don't exist in the schema
        }

        # Only add optional fields if they have values
        for key, value in optional_fields.items():
            if value is not None:
                task_record[key] = value

        try:
            (
                self.supabase
                .schema('knowledge')
                .table('chunkr_tasks')
                .insert(task_record)
                .execute()
            )
            print(f"✓ Created chunkr_tasks record for {task_id}")
        except Exception as e:
            # If record already exists, that's okay - continue
            error_str = str(e).lower()
            if 'duplicate' in error_str or 'unique' in error_str or 'already exists' in error_str:
                print(f"⚠ Task record already exists for {task_id}, continuing...")
            elif 'column' in error_str and 'not found' in error_str:
                # If a column doesn't exist, try with just task_id
                print(f"⚠ Some columns don't exist, retrying with minimal fields...")
                minimal_record = {'task_id': task_id}
                try:
                    (
                        self.supabase
                        .schema('knowledge')
                        .table('chunkr_tasks')
                        .insert(minimal_record)
                        .execute()
                    )
                    print(f"✓ Created chunkr_tasks record (minimal) for {task_id}")
                except Exception as e2:
                    error_str2 = str(e2).lower()
                    if 'duplicate' in error_str2 or 'unique' in error_str2:
                        print(f"⚠ Task record already exists for {task_id}, continuing...")
                    else:
                        print(f"⚠ Warning: Could not create task record: {e2}")
                        raise
            else:
                # Re-raise if it's a different error
                print(f"⚠ Warning: Could not create task record: {e}")
                raise
    
    def save_chunks_in_batches(
        self,
        processed_chunks: List[Dict],
        task_id: str
    ) -> Tuple[int, Dict[str, str]]:
        """Save chunks to Supabase in batches."""
        batch_size = 50
        saved_count = 0
        chunk_id_map = {}
        
        for i in range(0, len(processed_chunks), batch_size):
            batch = [
                chunk for chunk in processed_chunks[i:i + batch_size]
                if chunk.get('embedding')  # Only save chunks with embeddings
            ]
            
            if not batch:
                continue
            
            chunk_inserts = []
            for chunk in batch:
                cleaned_content = chunk.get('cleaned_embed') or self.clean_embed_text(chunk.get('embed', ''))
                chunk_inserts.append({
                    'chunk_id': chunk['id'],
                    'task_id': task_id,
                    'chunk_index': chunk['chunk_index'],
                    'content': chunk.get('content'),
                    'embed_text': cleaned_content,
                    'embed': chunk.get('embedding'),  # The vector embedding
                    'page_number': chunk.get('primary_page_number'),
                    'chunk_type': 'text',
                    'chunk_length': len(cleaned_content),
                    'project_id': chunk.get('project_id'),
                    'organization_id': chunk.get('organization_id'),
                    'file_name': chunk.get('file_name'),
                    'metadata': {
                        'total_segments': chunk.get('total_segments', 0),
                    },
                })
            
            response = (
                self.supabase
                .schema('knowledge')
                .table('document_chunks')
                .insert(chunk_inserts)
                .execute()
            )
            
            # Map chunk_id to database id
            for row in response.data:
                chunk_id_map[row['chunk_id']] = row['id']
            
            saved_count += len(batch)
            print(f"Saved chunk batch {i // batch_size + 1}/{(len(processed_chunks) + batch_size - 1) // batch_size}")
        
        return saved_count, chunk_id_map
    
    def save_segments_in_batches(
        self,
        processed_segments: List[Dict],
        chunk_id_map: Dict[str, str]
    ) -> int:
        """Save segments to Supabase in batches."""
        batch_size = 50
        saved_count = 0
        
        for i in range(0, len(processed_segments), batch_size):
            batch = processed_segments[i:i + batch_size]
            batch_number = i // batch_size + 1
            total_batches = (len(processed_segments) + batch_size - 1) // batch_size
            
            segment_inserts = []
            for segment in batch:
                # Skip base64 images
                image_url = None if self.is_base64_image(segment.get('image')) else segment.get('image')
                
                bbox = segment.get('bbox')
                bbox_json = json.dumps(bbox) if bbox else None
                
                db_chunk_id = chunk_id_map.get(segment['chunk_id'])
                if not db_chunk_id:
                    continue  # Skip if chunk wasn't saved
                
                segment_inserts.append({
                    'chunk_id': db_chunk_id,
                    'segment_index': segment['segment_index'],
                    'segment_id': segment.get('segment_id'),
                    'content': self.truncate_large_content(segment.get('content')),
                    'segment_type': segment.get('segment_type'),
                    'page_number': segment.get('page_number'),
                    'bbox': bbox_json,
                    'image_url': image_url,
                    'html_content': self.truncate_large_content(segment.get('html')),
                    'markdown_content': self.truncate_large_content(segment.get('markdown')),
                    'ocr_confidence': segment.get('confidence'),
                    'page_height': segment.get('page_height'),
                    'page_width': segment.get('page_width'),
                })
            
            if not segment_inserts:
                continue
            
            payload_size = len(json.dumps(segment_inserts))
            print(f"Inserting segment batch {batch_number}/{total_batches}: {len(segment_inserts)} segments, ~{payload_size // 1024}KB")
            
            try:
                (
                    self.supabase
                    .schema('knowledge')
                    .table('chunk_segments')
                    .insert(segment_inserts)
                    .execute()
                )
                saved_count += len(segment_inserts)
                print(f"Saved segment batch {batch_number}/{total_batches}")
            except Exception as e:
                print(f"Error saving segment batch {batch_number}: {e}")
                raise
        
        return saved_count
    
    def delete_existing_chunks(
        self,
        organization_id: str,
        project_id: Optional[str] = None,
        file_name: Optional[str] = None
    ) -> int:
        """
        Delete existing chunks and segments for the same org/project/file.
        
        IMPORTANT: This will ONLY delete chunks matching the exact file_name.
        If file_name is not provided, no deletion will occur.
        """
        if not file_name:
            print("⚠ No file_name provided. Skipping deletion to prevent accidental data loss.")
            return 0
            
        try:
            # Build query to find chunks to delete - MUST match exact filename
            query = (
                self.supabase
                .schema('knowledge')
                .table('document_chunks')
                .select('id, task_id, file_name')
                .eq('organization_id', organization_id)
                .eq('file_name', file_name)  # REQUIRED: Only delete exact filename match
            )
            
            if project_id:
                query = query.eq('project_id', project_id)
            
            # Get chunk IDs to delete
            response = query.execute()
            chunk_ids = [chunk['id'] for chunk in response.data]
            
            if not chunk_ids:
                print(f"No existing chunks found for file: {file_name}")
                return 0
            
            # Delete segments first (foreign key constraint) - batch delete
            segments_deleted = 0
            if chunk_ids:
                try:
                    # Delete segments in batches using IN clause
                    batch_size = 100
                    for i in range(0, len(chunk_ids), batch_size):
                        batch = chunk_ids[i:i + batch_size]
                        for chunk_id in batch:
                            try:
                                (
                                    self.supabase
                                    .schema('knowledge')
                                    .table('chunk_segments')
                                    .delete()
                                    .eq('chunk_id', chunk_id)
                                    .execute()
                                )
                                segments_deleted += 1
                            except:
                                pass
                except Exception as e:
                    print(f"⚠ Warning deleting segments: {e}")
            
            # Delete chunks - batch delete
            deleted_count = 0
            if chunk_ids:
                try:
                    # Delete chunks in batches
                    batch_size = 100
                    for i in range(0, len(chunk_ids), batch_size):
                        batch = chunk_ids[i:i + batch_size]
                        for chunk_id in batch:
                            try:
                                (
                                    self.supabase
                                    .schema('knowledge')
                                    .table('document_chunks')
                                    .delete()
                                    .eq('id', chunk_id)
                                    .execute()
                                )
                                deleted_count += 1
                            except:
                                pass
                except Exception as e:
                    print(f"⚠ Warning deleting chunks: {e}")
            
            print(f"✓ Deleted {deleted_count} existing chunks and their segments")
            return deleted_count
        except Exception as e:
            print(f"⚠ Error deleting existing chunks: {e}")
            return 0
    
    def save_chunks_to_supabase(
        self,
        task_json: Dict,
        knowledge: Dict,
        chunks_df=None,
        segments_df=None,
        replace_existing: bool = True
    ) -> Dict:
        """
        Main function to save Chunkr results to Supabase.
        
        Args:
            task_json: Full Chunkr task response JSON
            knowledge: Dict with keys: organization_id, project_id (optional), file_name (required for deletion)
            chunks_df: Optional DataFrame (if provided, will use this instead of parsing task_json)
            segments_df: Optional DataFrame (if provided, will use this instead of parsing task_json)
            replace_existing: If True, delete existing chunks for same org/project/file before saving
        
        Returns:
            Dict with saved counts
        """
        task_id = task_json.get('task_id')
        if not task_id:
            raise ValueError("task_id not found in task_json")
        
        # Get file_name from knowledge dict (primary source - user specified)
        # This ensures consistent file_name for both deletion and saving
        file_name_for_delete = knowledge.get('file_name')
        
        # Delete existing chunks if requested - with confirmation
        if replace_existing:
            if not file_name_for_delete:
                print("⚠ Warning: No file_name found. Cannot safely delete existing chunks.")
                print("   Proceeding without deletion. Chunks may be duplicated.")
            else:
                # Show what will be deleted and ask for confirmation
                print("\n" + "="*80)
                print("⚠ DELETION CONFIRMATION REQUIRED")
                print("="*80)
                print(f"File to DELETE and REPLACE: {file_name_for_delete}")
                print(f"Organization ID: {knowledge.get('organization_id')}")
                if knowledge.get('project_id'):
                    print(f"Project ID: {knowledge.get('project_id')}")
                print("\nThis will permanently delete ALL chunks for this exact filename.")
                print("="*80)
                
                # Get confirmation
                confirmation = input("\nType 'yes' or 'y' to confirm deletion, anything else to skip: ").strip().lower()
                
                if confirmation in ['yes', 'y']:
                    print(f"\n✓ Confirmed. Deleting existing chunks for: {file_name_for_delete}")
                    self.delete_existing_chunks(
                        organization_id=knowledge.get('organization_id'),
                        project_id=knowledge.get('project_id'),
                        file_name=file_name_for_delete
                    )
                else:
                    print(f"\n⚠ Deletion cancelled. Proceeding without deleting existing chunks.")
                    print("   Note: This may result in duplicate chunks if the file already exists.")
        
        # Create task record first to satisfy foreign key constraint
        self.create_chunkr_task_record(task_id, task_json, knowledge)
        
        chunks = task_json.get('output', {}).get('chunks', [])
        if not chunks:
            print("No chunks to process")
            return {'chunks_saved': 0, 'segments_saved': 0}
        
        # If DataFrames provided, convert to dict format
        if chunks_df is not None:
            # Convert DataFrame to list of dicts matching Chunkr format
            chunks = []
            for _, row in chunks_df.iterrows():
                chunk = {
                    'chunk_id': row.get('chunk_id'),
                    'content': row.get('content'),
                    'embed': row.get('embed'),
                }
                # If segments_df provided, attach segments
                if segments_df is not None:
                    chunk_segments = segments_df[segments_df['chunk_id'] == row['chunk_id']].to_dict('records')
                    chunk['segments'] = chunk_segments
                chunks.append(chunk)
        
        CHUNK_BATCH_SIZE = 25
        total_chunks_saved = 0
        total_segments_saved = 0
        global_chunk_id_map = {}
        
        print(f"Processing {len(chunks)} chunks in batches of {CHUNK_BATCH_SIZE}")
        
        for batch_start in range(0, len(chunks), CHUNK_BATCH_SIZE):
            batch_end = min(batch_start + CHUNK_BATCH_SIZE, len(chunks))
            chunk_batch = chunks[batch_start:batch_end]
            
            print(f"Processing chunk batch {batch_start // CHUNK_BATCH_SIZE + 1}/{(len(chunks) + CHUNK_BATCH_SIZE - 1) // CHUNK_BATCH_SIZE}")
            
            # Parse batch
            processed_chunks, all_segments = self.parse_task_data_batch(
                chunk_batch,
                batch_start,
                task_id,
                knowledge
            )
            
            if not processed_chunks:
                continue
            
            # Generate embeddings
            embeddings = self.generate_embeddings(processed_chunks)
            for chunk, embedding in zip(processed_chunks, embeddings):
                chunk['embedding'] = embedding
            
            # Save chunks
            chunks_saved, chunk_id_map = self.save_chunks_in_batches(
                processed_chunks,
                task_id
            )
            global_chunk_id_map.update(chunk_id_map)
            
            # Save segments (commented out for now)
            # segments_saved = self.save_segments_in_batches(
            #     all_segments,
            #     chunk_id_map
            # )
            # total_segments_saved += segments_saved
            
            total_chunks_saved += chunks_saved
            
            print(f"Saved batch: {chunks_saved} chunks")  # , {segments_saved} segments (segment saving disabled)
        
        print(f"Total saved: {total_chunks_saved} chunks")  # , {total_segments_saved} segments (segment saving disabled)
        return {
            'chunks_saved': total_chunks_saved,
            'segments_saved': 0,  # segment saving disabled
        }


# Usage example:
if __name__ == "__main__":
    # Assuming you have task_json from parse_pdf_to_dfs
    # and knowledge dict with org/project info
    
    saver = ChunkrToSupabase()
    
    knowledge = {
        'organization_id': 'your-org-id',
        'project_id': 'your-project-id',  # Optional
        'title': 'document.pdf',  # Optional
    }
    
    # Option 1: Use task_json directly
    # task_json, chunks_df, segments_df = parse_pdf_to_dfs("document.pdf")
    # result = saver.save_chunks_to_supabase(task_json, knowledge)
    
    # Option 2: Use DataFrames
    # result = saver.save_chunks_to_supabase(task_json, knowledge, chunks_df, segments_df)
    
    print("Done!")