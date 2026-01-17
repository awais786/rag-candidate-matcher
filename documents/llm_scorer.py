"""
LLM-based candidate scoring using OpenAI
"""
from typing import List, Dict, Optional
from documents.models import Candidate, Job
from documents.utils import EmbeddingManager
import json

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None


class LLMCandidateScorer:
    """Scores candidates using OpenAI LLM after ChromaDB ranking"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize LLM scorer
        
        Args:
            api_key: OpenAI API key (optional, uses settings if not provided)
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        from django.conf import settings
        self.api_key = api_key or getattr(settings, 'OPENAI_API_KEY', None)
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in settings.")
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is not installed. Install it with: pip install openai")
        
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
        self.embedding_manager = EmbeddingManager()
    
    def score_candidates_with_llm(
        self,
        job_description: str,
        candidate_matches: List[Dict],
        top_n: int = 10
    ) -> List[Dict]:
        """
        Score candidates using LLM after ChromaDB ranking
        
        Process:
        1. Take top N candidates from ChromaDB ranking
        2. Get full CV text for each candidate
        3. Send JD + CV to OpenAI LLM for scoring
        4. Get LLM score + explanation for each candidate
        5. Re-rank based on LLM scores
        
        Args:
            job_description: Job description text
            candidate_matches: List of candidate matches from ChromaDB (already ranked)
            top_n: Number of candidates to send to LLM (default: 10)
        
        Returns:
            List of candidates with LLM scores and explanations, re-ranked
        """
        if not candidate_matches:
            return []
        
        # Take top N candidates from ChromaDB ranking
        top_candidates = candidate_matches[:top_n]
        
        # Get full CV text for each candidate from ChromaDB
        for candidate_match in top_candidates:
            candidate_id = candidate_match['candidate_id']
            
            # Get all chunks for this candidate
            try:
                candidate_chunks = self.embedding_manager.collection.get(
                    where={"candidate_id": candidate_id}
                )
                
                # Combine all chunks into full CV text
                full_cv_text = '\n\n'.join(candidate_chunks.get('documents', []))
                candidate_match['full_cv_text'] = full_cv_text
                
                # Get candidate from Django DB
                try:
                    candidate = Candidate.objects.get(candidate_id=candidate_id)
                    candidate_match['candidate'] = candidate
                    if not candidate_match.get('email'):
                        candidate_match['email'] = candidate.email
                except Candidate.DoesNotExist:
                    pass
                    
            except Exception as e:
                candidate_match['full_cv_text'] = ''
                candidate_match['llm_error'] = str(e)
        
        # Score each candidate with LLM
        scored_candidates = []
        for candidate_match in top_candidates:
            try:
                llm_result = self._score_single_candidate(
                    job_description=job_description,
                    cv_text=candidate_match.get('full_cv_text', ''),
                    candidate_name=candidate_match.get('candidate_name', 'Unknown'),
                    chromadb_rank=candidate_match.get('overall_relevance', 0)
                )
                
                # Merge LLM results with ChromaDB ranking
                candidate_match.update({
                    'llm_score': llm_result['score'],
                    'llm_explanation': llm_result['explanation'],
                    'llm_ranking_factors': llm_result.get('ranking_factors', {}),
                    'chromadb_relevance': candidate_match.get('overall_relevance', 0),
                    'final_score': (
                        0.5 * llm_result['score'] +  # 50% LLM score
                        0.5 * candidate_match.get('overall_relevance', 0)  # 50% ChromaDB relevance
                    )
                })
                scored_candidates.append(candidate_match)
                
            except Exception as e:
                # If LLM scoring fails, keep ChromaDB ranking
                candidate_match.update({
                    'llm_score': 0,
                    'llm_explanation': f"LLM scoring failed: {str(e)}",
                    'final_score': candidate_match.get('overall_relevance', 0),
                    'llm_error': str(e)
                })
                scored_candidates.append(candidate_match)
        
        # Re-rank by final_score (combination of LLM + ChromaDB)
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        return scored_candidates
    
    def _score_single_candidate(
        self,
        job_description: str,
        cv_text: str,
        candidate_name: str,
        chromadb_rank: float
    ) -> Dict:
        """
        Score a single candidate using OpenAI LLM
        
        Args:
            job_description: Job description text
            cv_text: Full CV text
            candidate_name: Candidate name
            chromadb_rank: ChromaDB relevance score (0-1)
        
        Returns:
            Dictionary with score, explanation, and ranking factors
        """
        prompt = self._build_scoring_prompt(job_description, cv_text, candidate_name, chromadb_rank)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert HR recruiter evaluating candidates for job positions. Provide objective, detailed scoring based on job requirements."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent scoring
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Validate and normalize score
            score = float(result.get('score', 0))
            score = max(0, min(1, score))  # Clamp to [0, 1]
            
            return {
                'score': score,
                'explanation': result.get('explanation', ''),
                'ranking_factors': result.get('ranking_factors', {}),
                'strengths': result.get('strengths', []),
                'weaknesses': result.get('weaknesses', [])
            }
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _build_scoring_prompt(
        self,
        job_description: str,
        cv_text: str,
        candidate_name: str,
        chromadb_rank: float
    ) -> str:
        """
        Build prompt for LLM scoring
        
        Args:
            job_description: Job description text
            cv_text: Full CV text
            candidate_name: Candidate name
            chromadb_rank: ChromaDB relevance score
        
        Returns:
            Prompt string for LLM
        """
        return f"""Evaluate the candidate for this job position.

JOB DESCRIPTION:
{job_description}

CANDIDATE CV:
{cv_text}

CANDIDATE NAME: {candidate_name}

Note: This candidate was ranked {chromadb_rank:.2%} by semantic similarity search (ChromaDB).

Please provide a comprehensive evaluation in JSON format with the following structure:

{{
    "score": <float between 0.0 and 1.0>,  // Overall match score (0 = poor match, 1 = excellent match)
    "explanation": "<detailed explanation of the score>",
    "ranking_factors": {{
        "skills_match": <float>,  // How well skills match (0-1)
        "experience_relevance": <float>,  // Relevance of experience (0-1)
        "education_match": <float>,  // Education match (0-1)
        "overall_fit": <float>  // Overall fit for the role (0-1)
    }},
    "strengths": ["<strength1>", "<strength2>", ...],  // List of candidate strengths
    "weaknesses": ["<weakness1>", "<weakness2>", ...]  // List of candidate weaknesses or gaps
}}

Be thorough and objective. Consider:
- Technical skills required vs. candidate's skills
- Experience level and relevance
- Education requirements
- Overall fit for the role and company culture
- Both what the candidate has and what they're missing"""

