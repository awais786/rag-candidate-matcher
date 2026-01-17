"""
Process candidates from Django table in batches
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db.models import Q
from documents.models import Candidate
from documents.utils import PDFProcessor, EmbeddingManager
from pathlib import Path
import time

class Command(BaseCommand):
    help = "Process candidates from Django table - scan DB and generate embeddings in batches"
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=10,
            help='Number of candidates to process per batch (default: 10)'
        )
        parser.add_argument(
            '--resume-from',
            type=int,
            default=0,
            help='Resume processing from candidate ID (skip already processed)'
        )
        parser.add_argument(
            '--force-reprocess',
            action='store_true',
            help='Reprocess candidates that are already processed'
        )
        parser.add_argument(
            '--limit',
            type=int,
            help='Limit number of candidates to process (for testing)'
        )
    
    def handle(self, *args, **options):
        batch_size = options.get('batch_size', 10)
        resume_from = options.get('resume_from', 0)
        force_reprocess = options.get('force_reprocess', False)
        limit = options.get('limit')
        
        # Initialize processors (uses provider from settings - no hardcoding!)
        self.stdout.write("Initializing processors...")
        try:
            pdf_processor = PDFProcessor()
            embedding_manager = EmbeddingManager()
            self.stdout.write(self.style.SUCCESS(
                f"✓ Using embedding provider: {embedding_manager.embedding_type}"
            ))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error initializing: {e}"))
            return
        
        # Query candidates from database
        query = Candidate.objects.all()
        
        if not force_reprocess:
            # Only get unprocessed candidates
            query = query.filter(Q(embedding_processed=False) | Q(embedding_processed__isnull=True))
        
        # Resume from specific ID if specified
        if resume_from > 0:
            query = query.filter(candidate_id__gte=resume_from)
        
        # Order by ID
        query = query.order_by('candidate_id')
        
        # Apply limit if specified
        if limit:
            query = query[:limit]
        
        candidates = list(query)
        total = len(candidates)
        
        if total == 0:
            self.stdout.write(self.style.WARNING("No candidates to process"))
            return
        
        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(f"Found {total} candidate(s) to process")
        self.stdout.write(f"Batch size: {batch_size}")
        self.stdout.write(f"{'='*60}\n")
        
        processed = 0
        errors = 0
        skipped = 0
        
        # Process in batches
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = candidates[batch_start:batch_end]
            
            self.stdout.write(f"\n📦 Processing Batch {batch_start//batch_size + 1}: "
                            f"Candidates {batch_start+1}-{batch_end} of {total}")
            self.stdout.write("-" * 60)
            
            for candidate in batch:
                try:
                    self._process_candidate(
                        candidate, 
                        pdf_processor, 
                        embedding_manager,
                        force_reprocess
                    )
                    processed += 1
                    
                except FileNotFoundError as e:
                    skipped += 1
                    self.stdout.write(self.style.WARNING(
                        f"  ⏭️  Candidate {candidate.candidate_id}: {str(e)}"
                    ))
                    
                except Exception as e:
                    errors += 1
                    self.stdout.write(self.style.ERROR(
                        f"  ❌ Candidate {candidate.candidate_id} ({candidate.name}): {str(e)}"
                    ))
                    continue
            
            # Show progress after each batch
            self.stdout.write(f"\nProgress: {processed + skipped + errors}/{total} "
                            f"({100*(processed + skipped + errors)/total:.1f}%)")
            self.stdout.write(f"  ✅ Processed: {processed} | ⏭️  Skipped: {skipped} | ❌ Errors: {errors}")
            
            # Small delay between batches to avoid API rate limits
            if batch_end < total:
                time.sleep(1)
        
        # Final summary
        self.stdout.write("\n" + "="*60)
        self.stdout.write(self.style.SUCCESS("🎉 Processing Complete!"))
        self.stdout.write(f"{'='*60}")
        self.stdout.write(f"Total candidates: {total}")
        self.stdout.write(f"✅ Successfully processed: {processed}")
        self.stdout.write(f"⏭️  Skipped: {skipped}")
        self.stdout.write(f"❌ Errors: {errors}")
        self.stdout.write("="*60)
    
    def _process_candidate(self, candidate, pdf_processor, embedding_manager, force_reprocess=False):
        """Process a single candidate"""
        
        # Check if already processed
        if candidate.embedding_processed and not force_reprocess:
            self.stdout.write(f"  ⏭️  Candidate {candidate.candidate_id} ({candidate.name}): Already processed")
            return
        
        # Get file path from FileField
        if not candidate.cv_file:
            raise FileNotFoundError(f"CV file not uploaded for candidate {candidate.candidate_id}")
        
        cv_path = Path(candidate.cv_file.path)
        
        # Check if file exists
        if not cv_path.exists():
            raise FileNotFoundError(f"CV file not found: {cv_path}")
        
        self.stdout.write(f"  📄 Processing: {candidate.name} (ID: {candidate.candidate_id})")
        self.stdout.write(f"     File: {cv_path.name}")
        
        # Load and split PDF
        documents = pdf_processor.load_pdf(str(cv_path))
        chunks = pdf_processor.split_documents(documents)
        
        if len(chunks) == 0:
            self.stdout.write(self.style.WARNING(f"     ⚠️  No text extracted from PDF"))
            return
        
        # Prepare data for embeddings
        texts = [chunk.page_content for chunk in chunks]
        metadatas = []
        ids = []
        
        for idx, chunk in enumerate(chunks):
            metadata = {
                'candidate_id': candidate.candidate_id,
                'candidate_name': candidate.name,
                'email': candidate.email or '',
                'cv_file_path': candidate.cv_file.name,
                'chunk_index': idx,
                'page': chunk.metadata.get('page', 0),
                'type': 'candidate_chunk'  # Distinguish from job descriptions
            }
            metadatas.append(metadata)
            ids.append(f"candidate_{candidate.candidate_id}_chunk_{idx}")
        
        # Generate and store embeddings (using provider from settings)
        self.stdout.write(f"     📊 Generating embeddings for {len(texts)} chunks...")
        
        try:
            embedding_manager.store_embeddings(texts, metadatas, ids)
            
            # Update candidate status
            candidate.embedding_processed = True
            candidate.processed_at = timezone.now()
            candidate.chunks_count = len(chunks)
            candidate.save()
            
            self.stdout.write(self.style.SUCCESS(
                f"     ✅ Done: {len(chunks)} chunks stored in ChromaDB"
            ))
            
        except Exception as e:
            # If embedding fails, don't mark as processed
            raise Exception(f"Failed to generate/store embeddings: {str(e)}")

