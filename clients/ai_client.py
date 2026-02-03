"""AI client for chat and embeddings; supports OpenAI and Gemini providers."""
from typing import List, Any

from config.config import (
    AI_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)


class _GeminiChatAdapter:
    """Exposes an OpenAI-like chat.completions.create() using the Gemini API."""

    def __init__(self, api_key: str, model: str):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def chat(self) -> "_GeminiChatAdapter":
        return self

    @property
    def completions(self) -> "_GeminiChatAdapter":
        return self

    def create(
        self,
        model: str = None,
        messages: List[dict] = None,
        response_format: dict = None,
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> Any:
        system_content = ""
        user_content = ""
        for m in (messages or []):
            role = m.get("role", "")
            content = m.get("content", "") or ""
            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content
        model_name = model or self._model
        config = {"temperature": temperature}
        if system_content:
            config["system_instruction"] = system_content
        if response_format and response_format.get("type") == "json_object":
            config["response_mime_type"] = "application/json"
        response = self._client.models.generate_content(
            model=model_name,
            contents=user_content,
            config=config,
        )
        text = getattr(response, "text", None) or ""
        if not text and hasattr(response, "candidates") and response.candidates:
            parts = getattr(response.candidates[0].content, "parts", []) or []
            if parts:
                text = getattr(parts[0], "text", "") or ""
        # Return OpenAI-shaped response object
        class _Message:
            def __init__(self, c):
                self.content = c
        class _Choice:
            def __init__(self, msg):
                self.message = msg
        class _Response:
            def __init__(self, t):
                self.choices = [_Choice(_Message(t))]
        return _Response(text)


class AIClient:
    """Client for AI chat and embeddings. Uses OpenAI or Gemini for chat based on AI_PROVIDER."""

    def __init__(self):
        self._provider = (AI_PROVIDER or "gemini").strip().lower()
        self.model = OPENAI_MODEL
        self.embedding_model = EMBEDDING_MODEL

        if self._provider == "gemini":
            if not (GEMINI_API_KEY or "").strip():
                raise ValueError(
                    "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. "
                    "Set it in .env when AI_PROVIDER=gemini."
                )
            self.model = GEMINI_MODEL or "gemini-2.0-flash-exp"
            self.client = _GeminiChatAdapter(api_key=GEMINI_API_KEY.strip(), model=self.model)
            self._openai_for_embeddings = None
        else:
            if not (OPENAI_API_KEY or "").strip():
                raise ValueError("OPENAI_API_KEY is not set. Please set it in .env file.")
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL
            self._openai_for_embeddings = self.client

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text. Uses OpenAI embeddings (required for Supabase RAG)."""
        if self._openai_for_embeddings is None:
            if not (OPENAI_API_KEY or "").strip():
                raise ValueError(
                    "OPENAI_API_KEY is required for embeddings when using Gemini for chat. "
                    "Set OPENAI_API_KEY in .env for RAG/embeddings."
                )
            from openai import OpenAI
            self._openai_for_embeddings = OpenAI(api_key=OPENAI_API_KEY)
        response = self._openai_for_embeddings.embeddings.create(
            model=self.embedding_model,
            input=text,
            dimensions=1024,
        )
        return response.data[0].embedding
