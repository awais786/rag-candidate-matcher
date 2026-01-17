"""
Django management command to search for candidates matching a query string
"""
import json
from django.core.management.base import BaseCommand
from documents.candidate_matcher import CandidateMatcher


class Command(BaseCommand):
    help = "Vector search for candidates matching a job description (using embeddings)"
    
    def add_arguments(self, parser):
        parser.add_argument(
            "--query",
            type=str,
            help="Search query (job description, skills, keywords, etc.)",
        )
        parser.add_argument(
            "--job-id",
            type=int,
            help="Use stored Job from database by ID (alternative to --query)",
        )
        parser.add_argument(
            "--top-n",
            type=int,
            default=10,
            help="Number of top matches to return (default: 10)",
        )
        parser.add_argument(
            "--min-similarity",
            type=float,
            default=0.0,
            help="Minimum similarity threshold (0-1, default: 0.0)",
        )
        parser.add_argument(
            "--output",
            type=str,
            choices=['json', 'table', 'summary', 'ids'],
            default='summary',
            help="Output format (default: summary). Use 'ids' to output only candidate IDs as comma-separated list",
        )
    
    def handle(self, *args, **options):
        query = options.get("query")
        job_id = options.get("job_id")
        top_n = options.get("top_n", 10)
        min_similarity = options.get("min_similarity", 0.0)
        output_format = options.get("output", "summary")
        
        if not query and not job_id:
            self.stdout.write(
                self.style.ERROR("Error: Please provide either --query <text> or --job-id <id>")
            )
            return
        
        matcher = CandidateMatcher()
        
        # Vector search only (no LLM)
        if job_id:
            self.stdout.write(f"🔍 Vector search: Job ID {job_id}")
        else:
            self.stdout.write(f"🔍 Vector search: '{query}'")
        self.stdout.write(f"   Top {top_n} results, min similarity: {min_similarity:.0%}\n")
        
        matches = matcher.match_jd_against_candidates(
            job_description=query,
            job_id=job_id,
            top_n=top_n,
            min_similarity=min_similarity
        )
        
        if not matches:
            self.stdout.write(
                self.style.WARNING("❌ No matching candidates found.")
            )
            return
        
        self.stdout.write(self.style.SUCCESS(f"✅ Found {len(matches)} matching candidate(s)\n"))
        
        # Output IDs only if requested
        if output_format == 'ids':
            candidate_ids = [str(match['candidate_id']) for match in matches]
            print(','.join(candidate_ids))
        else:
            self._print_matches(matches, output_format)
    
    def _print_matches(self, matches: list, output_format: str):
        """Print match results in specified format"""
        
        if output_format == 'json':
            output = []
            for match in matches:
                item = {
                    'candidate_id': match['candidate_id'],
                    'candidate_name': match['candidate_name'],
                    'email': match.get('email'),
                    'final_score': round(match.get('final_score', match.get('overall_relevance', 0)), 4),
                    'chromadb_relevance': round(match.get('chromadb_relevance', match.get('overall_relevance', 0)), 4),
                    'avg_similarity': round(match['avg_similarity'], 4),
                    'max_similarity': round(match.get('max_similarity', 0), 4),
                    'chunks_count': match.get('chunks_count', 0),
                }
                output.append(item)
            print(json.dumps(output, indent=2))
        
        elif output_format == 'table':
            print("\n" + "=" * 100)
            print(f"{'Rank':<6} {'Name':<25} {'Email':<30} {'Relevance':<12} {'Max Sim':<12} {'Chunks':<8}")
            print("=" * 100)
            
            for i, match in enumerate(matches, 1):
                name = match['candidate_name'][:23]
                email = (match.get('email') or 'N/A')[:28]
                relevance = match.get('overall_relevance', match.get('avg_similarity', 0))
                max_sim = match.get('max_similarity', 0)
                chunks = match.get('chunks_count', 0)
                
                print(
                    f"{i:<6} "
                    f"{name:<25} "
                    f"{email:<30} "
                    f"{relevance:.2%} "
                    f"{max_sim:.2%} "
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
                
                self.stdout.write(f"   📊 ChromaDB Relevance: {match.get('overall_relevance', match['avg_similarity']):.2%} (Cosine Similarity)")
                self.stdout.write(f"   📈 Average Similarity: {match['avg_similarity']:.2%}")
                
                if match.get('max_similarity'):
                    self.stdout.write(f"   ⭐ Max Similarity: {match['max_similarity']:.2%}")
                if match.get('weighted_avg_similarity'):
                    self.stdout.write(f"   🎯 Weighted Average: {match.get('weighted_avg_similarity', 0):.2%}")
                if match.get('chunks_count'):
                    self.stdout.write(f"   📄 Matching Chunks: {match.get('chunks_count', 0)}")

