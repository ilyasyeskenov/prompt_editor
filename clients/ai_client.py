"""OpenAI client for AI operations."""
from openai import OpenAI
from typing import List, Dict, Any
import json
from config.config import OPENAI_API_KEY, OPENAI_MODEL, EMBEDDING_MODEL
from models.schemas import ComplianceReport, Justification
from utils.prompts import compliance_prompt, justification_prompt


class AIClient:
    """Client for OpenAI API operations."""
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Please set it in .env file.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.embedding_model = EMBEDDING_MODEL
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
            dimensions=1024  # Match the original implementation
        )
        return response.data[0].embedding
    
    def generate_compliance_report(self, document_chunks: List[str]) -> ComplianceReport:
        """Generate compliance report from document chunks."""
        # Join chunks into a single string
        documents_text = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(document_chunks)])
        
        # Create the prompt
        prompt = compliance_prompt(documents_text)
        
        # Use structured output with JSON mode
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a compliance expert. Generate structured compliance reports in JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        # Parse the response
        result_text = response.choices[0].message.content
        result_dict = json.loads(result_text)
        
        # Convert to Pydantic model
        return ComplianceReport(**result_dict)
    
    def generate_justification(self, requirement: str, document_chunks: List[Dict[str, Any]]) -> Justification:
        """Generate justification for a requirement based on document chunks."""
        # Format document chunks for the prompt
        chunks_text = json.dumps([
            {
                "id": chunk.get("id", ""),
                "content": chunk.get("content", ""),
                "file_name": chunk.get("file_name", ""),
                "page_number": chunk.get("page_number")
            }
            for chunk in document_chunks
        ], indent=2)
        
        # Create the prompt
        prompt = justification_prompt.replace("{documentChunkTexts}", chunks_text)
        prompt += f"\n\nThe requirement is: {requirement}"
        
        # Use structured output with JSON mode
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert construction engineer. Generate structured justifications in JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        # Parse the response
        result_text = response.choices[0].message.content
        result_dict = json.loads(result_text)
        
        # Convert to Pydantic model
        return Justification(**result_dict)

