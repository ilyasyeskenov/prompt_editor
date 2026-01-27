"""OpenAI client for AI operations."""
from openai import OpenAI
from typing import List
from prompt_editor.config.config import OPENAI_API_KEY, OPENAI_MODEL, EMBEDDING_MODEL


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

