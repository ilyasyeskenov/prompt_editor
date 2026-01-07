"""Document processing and chunking utilities."""
import os
import uuid
from typing import List, Dict, Optional
from PyPDF2 import PdfReader
import json


class DocumentChunk:
    """Represents a chunk of a document."""
    def __init__(self, id: str, content: str, chunk_index: int, 
                 file_name: str, page_number: Optional[int] = None):
        self.id = id
        self.content = content
        self.chunk_index = chunk_index
        self.file_name = file_name
        self.page_number = page_number
    
    def to_dict(self) -> Dict:
        """Convert chunk to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "file_name": self.file_name,
            "page_number": self.page_number
        }


class DocumentProcessor:
    """Processes documents and creates chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def chunk_text(self, text: str, file_name: str) -> List[DocumentChunk]:
        """Split text into chunks with overlap."""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():  # Only create chunk if it has content
                chunk_id = str(uuid.uuid4())
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    file_name=file_name
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Process a PDF file and return chunks."""
        file_name = os.path.basename(file_path)
        text = self.extract_text_from_pdf(file_path)
        return self.chunk_text(text, file_name)
    
    def process_text_file(self, file_path: str) -> List[DocumentChunk]:
        """Process a text file and return chunks."""
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.chunk_text(text, file_name)
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process a document (PDF or text) and return chunks."""
        if file_path.endswith('.pdf'):
            return self.process_pdf(file_path)
        elif file_path.endswith('.txt'):
            return self.process_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")


def save_chunks(chunks: List[DocumentChunk], output_path: str):
    """Save chunks to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([chunk.to_dict() for chunk in chunks], f, indent=2, ensure_ascii=False)


def load_chunks(file_path: str) -> List[DocumentChunk]:
    """Load chunks from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [DocumentChunk(**chunk_data) for chunk_data in data]

