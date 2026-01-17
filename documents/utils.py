"""
Utilities for PDF processing and embedding generation using LangChain
"""
import warnings
import re
from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from documents.embeddings.factory import EmbeddingFactory
import chromadb
from django.conf import settings


class PDFProcessor:
    """Handles PDF document loading and text extraction"""
    
    def __init__(self, loader_type="pypdf"):
        """
        Initialize PDF processor
        
        Args:
            loader_type: Either 'pypdf' or 'pdfplumber'
        """
        self.loader_type = loader_type
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load and extract text from PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects with page content and metadata
        """
        if self.loader_type == "pdfplumber":
            loader = PDFPlumberLoader(file_path)
        else:
            loader = PyPDFLoader(file_path)
        
        documents = loader.load()
        return documents
    
    def split_documents(
        self, 
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of Document objects
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunked Document objects
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def extract_structured_sections(self, full_text: str) -> Dict[str, List[str]]:
        """
        Extract structured sections from CV text: Skills, Education, Experience
        
        Args:
            full_text: Full CV text content
            
        Returns:
            Dictionary with sections: {'skills': [...], 'education': [...], 'experience': [...]}
        """
        text_lines = full_text.split('\n')
        full_text_lower = full_text.lower()
        
        sections = {
            'skills': [],
            'education': [],
            'experience': []
        }
        
        # Section keywords to identify sections
        skills_keywords = ['skills', 'technical skills', 'core skills', 'competencies', 
                          'skill set', 'proficiencies', 'technologies', 'tools', 'expertise']
        education_keywords = ['education', 'academic', 'qualifications', 'degree', 
                             'university', 'college', 'bachelor', 'master', 'phd', 'diploma']
        experience_keywords = ['experience', 'work experience', 'employment', 'employment history',
                              'career', 'professional experience', 'work history', 'positions']
        
        # Find section boundaries
        section_indices = {}
        for i, line in enumerate(text_lines):
            line_lower = line.strip().lower()
            
            # Check for skills section
            if any(keyword in line_lower for keyword in skills_keywords) and len(line_lower) < 100:
                if 'skills' not in section_indices:
                    section_indices['skills'] = i
                    continue
            
            # Check for education section
            if any(keyword in line_lower for keyword in education_keywords) and len(line_lower) < 100:
                if 'education' not in section_indices:
                    section_indices['education'] = i
                    continue
            
            # Check for experience section
            if any(keyword in line_lower for keyword in experience_keywords) and len(line_lower) < 100:
                if 'experience' not in section_indices:
                    section_indices['experience'] = i
                    continue
        
        # Extract sections based on found indices
        sorted_sections = sorted(section_indices.items(), key=lambda x: x[1])
        
        for section_idx, (section_name, start_idx) in enumerate(sorted_sections):
            # Determine end index (next section or end of text)
            if section_idx + 1 < len(sorted_sections):
                end_idx = sorted_sections[section_idx + 1][1]
            else:
                end_idx = len(text_lines)
            
            # Extract section text
            section_lines = text_lines[start_idx + 1:end_idx]  # +1 to skip header
            section_text = '\n'.join(section_lines).strip()
            
            if section_text:
                # Split into chunks if too long (max 2000 chars per chunk)
                if len(section_text) > 2000:
                    # Split by double newlines or large blocks
                    chunks = re.split(r'\n\n+', section_text)
                    for chunk in chunks:
                        if chunk.strip():
                            # Further split if still too long
                            if len(chunk) > 2000:
                                # Split by single newline
                                sub_chunks = [c.strip() for c in chunk.split('\n') if c.strip()]
                                sections[section_name].extend(sub_chunks)
                            else:
                                sections[section_name].append(chunk.strip())
                else:
                    sections[section_name].append(section_text)
        
        # If no structured sections found, try to extract skills from entire text
        if not sections['skills']:
            # Extract common skill keywords
            skills_pattern = r'\b(python|java|javascript|react|node|django|flask|sql|mongodb|postgresql|mysql|html|css|git|docker|kubernetes|aws|azure|gcp|linux|unix|c\+\+|c#|\.net|angular|vue|typescript|php|ruby|rails|go|rust|swift|kotlin|scala|machine learning|ml|ai|deep learning|nlp|data science|tensorflow|pytorch|pandas|numpy|selenium|testing|agile|scrum|devops|ci/cd|rest api|graphql|microservices|redis|elasticsearch)\b'
            skills_found = re.findall(skills_pattern, full_text_lower, re.IGNORECASE)
            if skills_found:
                sections['skills'] = list(set([s.title() for s in skills_found]))
        
        # Filter out empty sections
        sections = {k: [v for v in v_list if v and len(v.strip()) > 10] for k, v_list in sections.items()}
        
        return sections


class EmbeddingManager:
    """Manages embeddings generation and storage in ChromaDB"""
    
    def __init__(self):
        """
        Initialize embedding manager with provider from settings
        No hardcoded providers - uses factory pattern
        """
        # Get embedding provider from factory (based on settings)
        self.embedding_provider = EmbeddingFactory.create_provider()
        self.embedding_type = self.embedding_provider.provider_name
        
        # Initialize ChromaDB with new PersistentClient API
        persist_directory = str(Path(settings.BASE_DIR) / "chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Get current embedding dimension
        # Try to get embedding, fallback to local if Voyage/OpenAI fails
        try:
            test_embedding = self.embedding_provider.embed_query("test")
            self.embedding_dimension = len(test_embedding)
        except Exception as e:
            # If Voyage/OpenAI fails at runtime (network error, API error, etc.), fallback to local
            provider_name = getattr(self.embedding_provider, 'provider_name', 'unknown')
            if provider_name in ['voyage', 'openai']:
                warnings.warn(
                    f"{provider_name.capitalize()} API failed during embedding generation ({str(e)}). "
                    f"Falling back to local HuggingFace embeddings.",
                    UserWarning
                )
                # Reinitialize with local provider
                # Temporarily set provider to local
                original_provider = getattr(settings, 'EMBEDDING_PROVIDER', None)
                settings.EMBEDDING_PROVIDER = 'local'
                self.embedding_provider = EmbeddingFactory.create_provider()
                # Restore original setting
                settings.EMBEDDING_PROVIDER = original_provider
                test_embedding = self.embedding_provider.embed_query("test")
                self.embedding_dimension = len(test_embedding)
            else:
                raise
        
        # Get or create collection, handling dimension mismatches
        # ChromaDB uses cosine similarity by default (L2 distance for cosine)
        self.collection = self._get_or_create_collection_with_dimension_check(
            name="candidates"
        )
    
    def _get_or_create_collection_with_dimension_check(self, name: str):
        """
        Get or create collection, checking for dimension compatibility.
        If dimension mismatch, delete old collection and create new one.
        """
        try:
            # Try to get existing collection
            collection = self.chroma_client.get_collection(name=name)
            
            # Check if collection has embeddings
            count = collection.count()
            
            if count > 0:
                # Collection exists with data - check dimension compatibility
                # Try to add a test embedding to detect dimension mismatch
                try:
                    test_embedding = self.embedding_provider.embed_query("test")
                    collection.add(
                        embeddings=[test_embedding],
                        documents=["test"],
                        metadatas=[{"test": True}],
                        ids=["__dimension_test__"]
                    )
                    # If add succeeds, dimensions match - remove test embedding
                    collection.delete(ids=["__dimension_test__"])
                    return collection
                except Exception as e:
                    # Dimension mismatch detected - delete and recreate
                    error_msg = str(e).lower()
                    if "dimension" in error_msg or "embedding" in error_msg:
                        warnings.warn(
                            f"Dimension mismatch detected. Old collection had different dimension. "
                            f"Deleting old collection and creating new one with dimension {self.embedding_dimension}. "
                            f"All existing embeddings will be removed.",
                            UserWarning
                        )
                        self.chroma_client.delete_collection(name=name)
                        collection = self.chroma_client.create_collection(name=name)
                        return collection
                    else:
                        # Different error, re-raise
                        raise
            else:
                # Collection exists but empty - safe to use
                return collection
                
        except Exception:
            # Collection doesn't exist - create it
            # ChromaDB default is L2 distance, but for cosine similarity we can use L2 with normalized embeddings
            # Most embedding models (OpenAI, Voyage, HuggingFace) return L2-normalized embeddings
            # So L2 distance approximates cosine distance for normalized vectors
            collection = self.chroma_client.create_collection(name=name)
            return collection
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            return self.embedding_provider.embed_documents(texts)
        except Exception as e:
            # If Voyage/OpenAI API fails during generation (network error, API error, etc.), fallback to local
            provider_name = getattr(self.embedding_provider, 'provider_name', 'unknown')
            if provider_name in ['voyage', 'openai']:
                warnings.warn(
                    f"{provider_name.capitalize()} API failed during embedding generation ({str(e)}). "
                    f"Falling back to local HuggingFace embeddings for this batch.",
                    UserWarning
                )
                # Reinitialize with local provider
                original_provider = getattr(settings, 'EMBEDDING_PROVIDER', None)
                settings.EMBEDDING_PROVIDER = 'local'
                self.embedding_provider = EmbeddingFactory.create_provider()
                settings.EMBEDDING_PROVIDER = original_provider
                # Update dimension if changed
                test_embedding = self.embedding_provider.embed_query("test")
                self.embedding_dimension = len(test_embedding)
                # Retry with local provider
                return self.embedding_provider.embed_documents(texts)
            else:
                raise
    
    def store_embeddings(
        self, 
        texts: List[str], 
        metadatas: List[Dict],
        ids: List[str]
    ) -> None:
        """
        Store embeddings in ChromaDB
        
        Args:
            texts: List of text strings
            metadatas: List of metadata dictionaries
            ids: List of unique IDs for each text
        """
        embeddings = self.generate_embeddings(texts)
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search for similar documents using embeddings
        
        Args:
            query: Query text
            n_results: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        try:
            query_embedding = self.embedding_provider.embed_query(query)
        except Exception as e:
            # If Voyage/OpenAI API fails during query, fallback to local
            provider_name = getattr(self.embedding_provider, 'provider_name', 'unknown')
            if provider_name in ['voyage', 'openai']:
                warnings.warn(
                    f"{provider_name.capitalize()} API failed during query embedding ({str(e)}). "
                    f"Falling back to local HuggingFace embeddings.",
                    UserWarning
                )
                # Reinitialize with local provider
                original_provider = getattr(settings, 'EMBEDDING_PROVIDER', None)
                settings.EMBEDDING_PROVIDER = 'local'
                self.embedding_provider = EmbeddingFactory.create_provider()
                settings.EMBEDDING_PROVIDER = original_provider
                query_embedding = self.embedding_provider.embed_query(query)
            else:
                raise
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

