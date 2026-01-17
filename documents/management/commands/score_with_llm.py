"""
Management command to score candidates with OpenAI LLM (final step after vector search)
"""
import json
from django.core.management.base import BaseCommand
from documents.candidate_matcher import CandidateMatcher


class Command(BaseCommand):
    help = "Score candidates with OpenAI LLM (final step after vector search)"
    
    def add_arguments(self, parser):
        parser.add_argument(
            "--job-id",
            type=int,
            help="Job ID to score candidates for (if not using --input-file)",
        )
        parser.add_argument(
            "--input-file",
            type=str,
            help="JSON file with vector search results from 'search_candidates --output json'",
        )
        parser.add_argument(
            "--candidate-ids",
            type=str,
            help="Comma-separated list of candidate IDs (e.g., '1,2,3') from 'search_candidates --output ids'",
        )
        parser.add_argument(
            "--top-n",
            type=int,
            default=10,
            help="Number of top candidates to score with LLM (default: 10, ignored if using --input-file)",
        )
        parser.add_argument(
            "--min-similarity",
            type=float,
            default=0.0,
            help="Minimum similarity threshold from vector search (0-1, default: 0.0, ignored if using --input-file)",
        )
        parser.add_argument(
            "--output",
            type=str,
            choices=['json', 'table', 'summary'],
            default='summary',
            help="Output format (default: summary)",
        )
    
    def handle(self, *args, **options):
        job_id = options.get('job_id')
        input_file = options.get('input_file')
        top_n = options.get('top_n', 10)
        min_similarity = options.get('min_similarity', 0.0)
        output_format = options.get('output', 'summary')
        
        self.stdout.write("=" * 80)
        self.stdout.write(self.style.SUCCESS("🤖 OpenAI LLM Candidate Scoring"))
        self.stdout.write("=" * 80)
        
        matcher = CandidateMatcher()
        
        try:
            # Option 1: Use candidate IDs from previous search_candidates run
            if options.get('candidate_ids'):
                candidate_ids_str = options.get('candidate_ids')
                candidate_ids = [int(id.strip()) for id in candidate_ids_str.split(',') if id.strip()]
                
                if not candidate_ids:
                    self.stdout.write(self.style.ERROR("❌ No valid candidate IDs provided"))
                    return
                
                self.stdout.write(f"\n📋 Using candidate IDs: {', '.join(map(str, candidate_ids))}")
                self.stdout.write("🔍 Fetching vector search results for specified candidates...")
                
                # Get vector search results, then filter to only requested IDs
                if not job_id:
                    self.stdout.write(self.style.ERROR("❌ Job ID required when using --candidate-ids. Please provide --job-id"))
                    return
                
                # Get all candidates from vector search
                chromadb_matches = matcher.match_jd_against_candidates(
                    job_id=job_id,
                    top_n=1000,  # Get many to filter
                    min_similarity=min_similarity
                )
                
                # Filter to only requested candidate IDs
                chromadb_matches = [m for m in chromadb_matches if m['candidate_id'] in candidate_ids]
                chromadb_matches.sort(key=lambda x: candidate_ids.index(x['candidate_id']))  # Keep original order
                
                if not chromadb_matches:
                    self.stdout.write(self.style.WARNING(f"❌ No candidates found with IDs: {candidate_ids}"))
                    return
                
                self.stdout.write(self.style.SUCCESS(f"✅ Found {len(chromadb_matches)} candidates\n"))
            
            # Option 2: Use input file from previous search_candidates run
            elif input_file:
                import os
                if not os.path.exists(input_file):
                    self.stdout.write(self.style.ERROR(f"❌ Input file not found: {input_file}"))
                    return
                
                self.stdout.write(f"\n📂 Loading vector search results from: {input_file}")
                with open(input_file, 'r') as f:
                    input_data = json.load(f)
                
                if not input_data:
                    self.stdout.write(self.style.WARNING("❌ No candidates in input file."))
                    return
                
                # Convert input data back to match format
                chromadb_matches = []
                for item in input_data:
                    chromadb_matches.append({
                        'candidate_id': item['candidate_id'],
                        'candidate_name': item['candidate_name'],
                        'email': item.get('email'),
                        'overall_relevance': item.get('chromadb_relevance', item.get('final_score', 0)),
                        'chromadb_relevance': item.get('chromadb_relevance', item.get('final_score', 0)),
                        'avg_similarity': item.get('avg_similarity', 0),
                        'max_similarity': item.get('max_similarity', 0),
                        'chunks_count': item.get('chunks_count', 0),
                    })
                
                job_id = job_id or input_data[0].get('job_id')  # Try to get from input
                if not job_id:
                    self.stdout.write(self.style.WARNING("⚠️  Job ID not found. Please provide --job-id"))
                
                self.stdout.write(self.style.SUCCESS(f"✅ Loaded {len(chromadb_matches)} candidates from file\n"))
            
            # Option 2: Run vector search now
            else:
                if not job_id:
                    self.stdout.write(self.style.ERROR("❌ Error: Please provide either --job-id or --input-file"))
                    return
                
                self.stdout.write(f"\n📋 Job ID: {job_id}")
                self.stdout.write(f"🔍 Step 1: Vector matching (ChromaDB)...")
                self.stdout.write(f"🎯 Top {top_n} candidates, min similarity: {min_similarity:.0%}\n")
                
                chromadb_matches = matcher.match_jd_against_candidates(
                    job_id=job_id,
                    top_n=top_n,
                    min_similarity=min_similarity
                )
                
                if not chromadb_matches:
                    self.stdout.write(self.style.WARNING("❌ No candidates found from vector search."))
                    return
                
                self.stdout.write(self.style.SUCCESS(f"✅ Found {len(chromadb_matches)} candidates from vector search\n"))
            
            # Step 2: Score with OpenAI LLM
            if not job_id:
                self.stdout.write(self.style.ERROR("❌ Job ID required for LLM scoring. Please provide --job-id"))
                return
            
            self.stdout.write("🤖 Step 2: Scoring with OpenAI LLM...")
            self.stdout.write(f"   Evaluating {len(chromadb_matches)} candidates...\n")
            
            # Use LLM scorer directly with the chromadb_matches
            from documents.llm_scorer import LLMCandidateScorer
            from documents.models import Job
            
            job = Job.objects.get(job_id=job_id)
            llm_scorer = LLMCandidateScorer()
            
            final_matches = llm_scorer.score_candidates_with_llm(
                job_description=job.description,
                candidate_matches=chromadb_matches,
                top_n=len(chromadb_matches)
            )
            
            if not final_matches:
                self.stdout.write(self.style.WARNING("❌ No candidates scored."))
                return
            
            self.stdout.write(self.style.SUCCESS(f"✅ Scored {len(final_matches)} candidates with LLM\n"))
            self.stdout.write("=" * 80)
            
            self._print_results(final_matches, output_format)
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Error: {str(e)}"))
            return
    
    def _print_results(self, matches: list, output_format: str):
        """Print LLM scoring results in specified format"""
        
        if output_format == 'json':
            output = []
            for match in matches:
                item = {
                    'candidate_id': match['candidate_id'],
                    'candidate_name': match['candidate_name'],
                    'email': match.get('email'),
                    'final_score': round(match.get('final_score', 0), 4),
                    'llm_score': round(match.get('llm_score', 0), 4),
                    'chromadb_relevance': round(match.get('chromadb_relevance', 0), 4),
                    'llm_explanation': match.get('llm_explanation', ''),
                    'ranking_factors': match.get('ranking_factors', {}),
                    'strengths': match.get('strengths', []),
                    'weaknesses': match.get('weaknesses', []),
                    'avg_similarity': round(match.get('avg_similarity', 0), 4),
                    'chunks_count': match.get('chunks_count', 0),
                }
                output.append(item)
            print(json.dumps(output, indent=2))
        
        elif output_format == 'table':
            print("\n" + "=" * 140)
            print(f"{'Rank':<6} {'Name':<25} {'Email':<30} {'Final':<10} {'LLM':<10} {'ChromaDB':<12} {'Chunks':<8}")
            print("=" * 140)
            
            for i, match in enumerate(matches, 1):
                name = match['candidate_name'][:23]
                email = (match.get('email') or 'N/A')[:28]
                final_score = match.get('final_score', 0)
                llm_score = match.get('llm_score', 0)
                chromadb_score = match.get('chromadb_relevance', match.get('overall_relevance', 0))
                chunks = match.get('chunks_count', 0)
                
                print(
                    f"{i:<6} "
                    f"{name:<25} "
                    f"{email:<30} "
                    f"{final_score:.2%} "
                    f"{llm_score:.2%} "
                    f"{chromadb_score:.2%} "
                    f"{chunks:<8}"
                )
        
        else:  # summary
            for i, match in enumerate(matches, 1):
                self.stdout.write("\n" + "=" * 80)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"\n{i}. {match['candidate_name']} (ID: {match['candidate_id']})"
                    )
                )
                
                if match.get('email'):
                    self.stdout.write(f"   📧 Email: {match['email']}")
                
                # Final score
                self.stdout.write(f"\n   🏆 Final Score: {match.get('final_score', 0):.2%} (50% LLM + 50% ChromaDB)")
                self.stdout.write(f"   🤖 LLM Score: {match.get('llm_score', 0):.2%}")
                self.stdout.write(f"   📊 ChromaDB Relevance: {match.get('chromadb_relevance', match.get('overall_relevance', 0)):.2%}")
                
                # LLM Explanation
                if match.get('llm_explanation'):
                    explanation = match.get('llm_explanation', '')
                    self.stdout.write(f"\n   💬 LLM Explanation:")
                    self.stdout.write(f"      {explanation[:300]}...")
                
                # Ranking Factors
                if match.get('ranking_factors'):
                    factors = match['ranking_factors']
                    self.stdout.write(f"\n   📋 LLM Ranking Factors:")
                    for factor, value in factors.items():
                        self.stdout.write(f"      • {factor.replace('_', ' ').title()}: {value:.2%}")
                
                # Strengths
                if match.get('strengths'):
                    self.stdout.write(f"\n   ✅ Strengths:")
                    for strength in match['strengths'][:5]:
                        self.stdout.write(f"      • {strength}")
                
                # Weaknesses
                if match.get('weaknesses'):
                    self.stdout.write(f"\n   ⚠️  Weaknesses/Gaps:")
                    for weakness in match['weaknesses'][:5]:
                        self.stdout.write(f"      • {weakness}")
                
                # Vector search metrics
                self.stdout.write(f"\n   📈 Vector Search Metrics:")
                self.stdout.write(f"      • Average Similarity: {match.get('avg_similarity', 0):.2%}")
                if match.get('max_similarity'):
                    self.stdout.write(f"      • Max Similarity: {match.get('max_similarity', 0):.2%}")
                if match.get('chunks_count'):
                    self.stdout.write(f"      • Matching Chunks: {match.get('chunks_count', 0)}")

