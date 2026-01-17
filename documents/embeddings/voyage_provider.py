"""
Voyage AI Embedding Provider
"""
from typing import List
import voyageai
from .base import BaseEmbeddingProvider

class VoyageEmbeddingProvider(BaseEmbeddingProvider):
    """Voyage AI embeddings provider"""
    
    def __init__(self, api_key: str, model: str = "voyage-2"):
        self.api_key = api_key
        self.model = model
        self.client = voyageai.Client(api_key=api_key)
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        result = self.client.embed([text], model=self.model)
        return result.embeddings[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents (batched)"""
        batch_size = 128  # Voyage API max batch size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            result = self.client.embed(batch, model=self.model)
            all_embeddings.extend(result.embeddings)
        
        return all_embeddings
    
    @property
    def provider_name(self) -> str:
        return "voyage"

