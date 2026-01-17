"""
OpenAI Embedding Provider
"""
from typing import List

try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAIEmbeddings = None

from .base import BaseEmbeddingProvider

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embeddings provider"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "langchain-openai is not installed. "
                "Install it with: pip install langchain-openai"
            )
        self.api_key = api_key
        self.model = model
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model=model
        )
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        return self.embeddings.embed_documents(texts)
    
    @property
    def provider_name(self) -> str:
        return "openai"

