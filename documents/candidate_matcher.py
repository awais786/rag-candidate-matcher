"""
Candidate Matching System: Match Job Descriptions against Candidates
"""
from typing import List, Dict, Optional
from documents.utils import EmbeddingManager
from documents.models import Candidate, Job


class CandidateMatcher:
    """Matches job descriptions against candidates using embeddings"""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
    
    def match_jd_against_candidates(
        self, 
        job_description: Optional[str] = None,
        job_id: Optional[int] = None,
        top_n: int = 10,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Match a job description against all processed candidates using cosine similarity
        
        Process:
        1. Get Job Description embedding (from text or from Job model)
        2. Query candidate embeddings in ChromaDB using JD embedding (cosine similarity)
        3. Aggregate similarity scores across all candidate chunks
        4. Rank candidates by overall relevance score
        
        Similarity Calculation:
        - Uses cosine similarity (ChromaDB default)
        - Cosine distance: 0 = identical, 2 = opposite
        - Cosine similarity: 1 - distance (range: [-1, 1], normalized to [0, 1])
        
        Aggregation Strategy:
        - Overall relevance = weighted combination of:
          * 40% weighted average (emphasizes strong matches)
          * 30% max similarity (best matching chunk)
          * 20% top-k average (average of top 3 chunks)
          * 10% overall average (mean across all chunks)
        
        Args:
            job_description: The job description text (required if job_id not provided)
            job_id: ID of Job model (optional, will use stored embedding)
            top_n: Number of top matches to return
            min_similarity: Minimum similarity threshold (0-1, uses cosine similarity)
        
        Returns:
            List of dictionaries with candidate matches and aggregated scores, ranked by relevance
        """
        # Step 1: Get job description and embedding
        if job_id:
            # Use stored Job from database
            try:
                job = Job.objects.get(job_id=job_id)
                if not job.embedding_processed:
                    raise ValueError(f"Job {job_id} has not been processed. Run 'process_jobs' first.")
                job_description = job.description
                # Retrieve stored embedding from ChromaDB
                stored_job = self.embedding_manager.collection.get(ids=[job.embedding_id])
                if stored_job.get('embeddings') and len(stored_job['embeddings']) > 0:
                    jd_embedding = stored_job['embeddings'][0]
                else:
                    # Fallback: generate embedding
                    jd_embedding = self.embedding_manager.embedding_provider.embed_query(job_description)
            except Job.DoesNotExist:
                raise ValueError(f"Job with ID {job_id} not found")
        elif job_description:
            # Generate embedding for provided text
            jd_embedding = self.embedding_manager.embedding_provider.embed_query(job_description)
        else:
            raise ValueError("Either job_description or job_id must be provided")
        
        # Step 2: Query candidate embeddings in ChromaDB using JD embedding
        # Filter to only get candidate chunks (exclude job descriptions)
        results = self.embedding_manager.collection.query(
            query_embeddings=[jd_embedding],
            n_results=top_n * 5,  # Get more results to filter by candidate
            where={"type": {"$ne": "job_description"}}  # Only match candidate chunks
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        # Group results by candidate_id and aggregate similarity scores
        # Using cosine similarity: similarity = 1 - distance
        # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
        candidate_scores = {}
        
        for doc_text, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            candidate_id = metadata.get('candidate_id')
            if not candidate_id:
                continue
            
            # Convert cosine distance to cosine similarity
            # Cosine distance: 0 = identical, 2 = opposite
            # Cosine similarity: 1 = identical, -1 = opposite
            # Formula: similarity = 1 - (distance / 2) for normalized cosine
            # ChromaDB uses L2-normalized cosine, so: similarity = 1 - distance
            cosine_similarity = 1 - distance  # Range: [-1, 1] but typically [0, 1] for similar docs
            
            # Normalize to [0, 1] range (negative similarities become 0)
            normalized_similarity = max(0, cosine_similarity)
            
            if normalized_similarity < min_similarity:
                continue
            
            if candidate_id not in candidate_scores:
                candidate_scores[candidate_id] = {
                    'candidate_id': candidate_id,
                    'candidate_name': metadata.get('candidate_name', 'Unknown'),
                    'chunks': [],
                    'similarities': [],
                    'distances': [],
                    'avg_similarity': 0,
                    'weighted_avg_similarity': 0,
                    'max_similarity': 0,
                    'min_distance': float('inf'),
                    'relevant_content': []
                }
            
            candidate_scores[candidate_id]['chunks'].append(doc_text)
            candidate_scores[candidate_id]['similarities'].append(normalized_similarity)
            candidate_scores[candidate_id]['distances'].append(distance)
            candidate_scores[candidate_id]['min_distance'] = min(
                candidate_scores[candidate_id]['min_distance'], 
                distance
            )
            candidate_scores[candidate_id]['relevant_content'].append({
                'content': doc_text[:200] + '...' if len(doc_text) > 200 else doc_text,
                'similarity': normalized_similarity,
                'distance': distance
            })
        
        # Calculate aggregated similarity scores for each candidate
        matches = []
        for candidate_id, data in candidate_scores.items():
            similarities = data['similarities']
            
            # Multiple aggregation methods for ranking:
            # 1. Average similarity (mean across all chunks)
            avg_sim = sum(similarities) / len(similarities) if similarities else 0
            
            # 2. Weighted average (give more weight to higher similarity chunks)
            # Weight by similarity^2 to emphasize strong matches
            if similarities:
                weights = [s**2 for s in similarities]
                weighted_avg = sum(s * w for s, w in zip(similarities, weights)) / sum(weights)
            else:
                weighted_avg = 0
            
            # 3. Max similarity (best matching chunk)
            max_sim = max(similarities) if similarities else 0
            
            # 4. Top-K average (average of top 3 chunks)
            top_k = min(3, len(similarities))
            top_k_avg = sum(sorted(similarities, reverse=True)[:top_k]) / top_k if similarities else 0
            
            # Overall relevance score: weighted combination
            # 40% weighted average, 30% max similarity, 20% top-k average, 10% overall average
            overall_relevance = (
                0.4 * weighted_avg +
                0.3 * max_sim +
                0.2 * top_k_avg +
                0.1 * avg_sim
            )
            
            data['avg_similarity'] = avg_sim
            data['weighted_avg_similarity'] = weighted_avg
            data['max_similarity'] = max_sim
            data['top_k_avg_similarity'] = top_k_avg
            data['overall_relevance'] = overall_relevance
            data['chunks_count'] = len(similarities)
            
            # Get candidate from Django DB
            try:
                candidate = Candidate.objects.get(candidate_id=candidate_id)
                data['candidate'] = candidate
                data['email'] = candidate.email
                data['cv_file'] = candidate.cv_file.name if candidate.cv_file else None
            except Candidate.DoesNotExist:
                pass
            
            matches.append(data)
        
        # Sort by overall relevance score (descending)
        # This aggregates similarity scores across all candidate chunks
        matches.sort(key=lambda x: x['overall_relevance'], reverse=True)
        
        # Store ChromaDB ranking before LLM scoring
        # The matches are already ranked by ChromaDB, ready for LLM scoring
        ranked_matches = matches[:top_n]
        
        # Add ranking position from ChromaDB
        for idx, match in enumerate(ranked_matches, 1):
            match['chromadb_rank'] = idx
            match['chromadb_relevance'] = match['overall_relevance']
        
        return ranked_matches

