"""
Factory to create embedding providers based on configuration
"""
from django.conf import settings
from .base import BaseEmbeddingProvider
from .voyage_provider import VoyageEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider
from .local_provider import LocalEmbeddingProvider

class EmbeddingFactory:
    """Factory class to create embedding providers"""
    
    @staticmethod
    def create_provider() -> BaseEmbeddingProvider:
        """
        Create embedding provider based on settings
        
        Priority:
        1. EMBEDDING_PROVIDER setting (voyage, openai, local)
        2. Check for API keys
        3. Fallback to local
        
        Returns:
            BaseEmbeddingProvider instance
        """
        provider_name = getattr(settings, 'EMBEDDING_PROVIDER', None)
        
        # If not set, auto-detect based on available API keys
        if not provider_name:
            if getattr(settings, 'VOYAGE_API_KEY', None):
                provider_name = 'voyage'
            elif getattr(settings, 'OPENAI_API_KEY', None):
                provider_name = 'openai'
            else:
                provider_name = 'local'
        
        provider_name = provider_name.lower()
        
        # Create provider based on name
        if provider_name == 'voyage':
            api_key = getattr(settings, 'VOYAGE_API_KEY', None)
            model = getattr(settings, 'VOYAGE_MODEL', 'voyage-2')
            if not api_key:
                # Fallback to local if no API key
                import warnings
                warnings.warn(
                    "Voyage API key not found. Falling back to local HuggingFace embeddings."
                )
                # Create local provider
                local_model = getattr(
                    settings, 
                    'LOCAL_EMBEDDING_MODEL', 
                    'sentence-transformers/all-MiniLM-L6-v2'
                )
                return LocalEmbeddingProvider(model_name=local_model)
            else:
                # Try to initialize Voyage, fallback to local if it fails
                try:
                    return VoyageEmbeddingProvider(api_key=api_key, model=model)
                except Exception as e:
                    # If Voyage initialization fails (network error, invalid key, etc.), fallback to local
                    import warnings
                    warnings.warn(
                        f"Failed to initialize Voyage API ({str(e)}). Falling back to local HuggingFace embeddings."
                    )
                    # Create local provider
                    local_model = getattr(
                        settings, 
                        'LOCAL_EMBEDDING_MODEL', 
                        'sentence-transformers/all-MiniLM-L6-v2'
                    )
                    return LocalEmbeddingProvider(model_name=local_model)
        
        elif provider_name == 'openai':
            api_key = getattr(settings, 'OPENAI_API_KEY', None)
            model = getattr(settings, 'OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY in settings."
                )
            return OpenAIEmbeddingProvider(api_key=api_key, model=model)
        
        elif provider_name == 'local':
            model = getattr(
                settings, 
                'LOCAL_EMBEDDING_MODEL', 
                'sentence-transformers/all-MiniLM-L6-v2'
            )
            return LocalEmbeddingProvider(model_name=model)
        
        else:
            raise ValueError(
                f"Unknown embedding provider: {provider_name}. "
                f"Supported: voyage, openai, local"
            )

