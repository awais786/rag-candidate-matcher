"""
Local HuggingFace Embedding Provider (Free, no API key needed)
"""
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from .base import BaseEmbeddingProvider

class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local HuggingFace embeddings provider (free, offline)"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        return self.embeddings.embed_documents(texts)
    
    @property
    def provider_name(self) -> str:
        return "local"

