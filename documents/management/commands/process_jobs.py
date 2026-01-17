"""
Process jobs to generate and store embeddings
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db.models import Q
from documents.models import Job
from documents.utils import EmbeddingManager
import time


class Command(BaseCommand):
    help = "Process jobs from Django table - generate embeddings and store in ChromaDB"
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--job-id',
            type=int,
            help='Process specific job by ID'
        )
        parser.add_argument(
            '--force-reprocess',
            action='store_true',
            help='Reprocess jobs that are already processed'
        )
        parser.add_argument(
            '--limit',
            type=int,
            help='Limit number of jobs to process (for testing)'
        )
    
    def handle(self, *args, **options):
        job_id = options.get('job_id')
        force_reprocess = options.get('force_reprocess', False)
        limit = options.get('limit')
        
        # Initialize embedding manager
        self.stdout.write("Initializing embedding manager...")
        try:
            embedding_manager = EmbeddingManager()
            self.stdout.write(self.style.SUCCESS(
                f"✓ Using embedding provider: {embedding_manager.embedding_type}"
            ))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error initializing: {e}"))
            return
        
        # Get jobs to process
        if job_id:
            jobs = Job.objects.filter(job_id=job_id)
            if not jobs.exists():
                self.stdout.write(self.style.ERROR(f"Job with ID {job_id} not found"))
                return
        else:
            if force_reprocess:
                jobs = Job.objects.all()
            else:
                jobs = Job.objects.filter(embedding_processed=False)
            
            if limit:
                jobs = jobs[:limit]
        
        total_jobs = jobs.count()
        
        if total_jobs == 0:
            self.stdout.write(self.style.WARNING("No jobs to process"))
            return
        
        self.stdout.write(f"\n📋 Processing {total_jobs} job(s)...")
        self.stdout.write("="*60)
        
        processed_count = 0
        failed_count = 0
        
        for job in jobs:
            try:
                self._process_job(job, embedding_manager, force_reprocess)
                processed_count += 1
            except Exception as e:
                failed_count += 1
                self.stdout.write(self.style.ERROR(
                    f"  ❌ Failed: {job.title} (ID: {job.job_id}) - {str(e)}"
                ))
        
        # Summary
        self.stdout.write("\n" + "="*60)
        self.stdout.write(self.style.SUCCESS(f"✅ Processed: {processed_count}"))
        if failed_count > 0:
            self.stdout.write(self.style.ERROR(f"❌ Failed: {failed_count}"))
        self.stdout.write("="*60)
    
    def _process_job(self, job, embedding_manager, force_reprocess=False):
        """Process a single job"""
        
        # Check if already processed
        if job.embedding_processed and not force_reprocess:
            self.stdout.write(f"  ⏭️  Job {job.job_id} ({job.title}): Already processed")
            return
        
        if not job.description:
            raise ValueError(f"Job description is empty for job {job.job_id}")
        
        self.stdout.write(f"  📄 Processing: {job.title} (ID: {job.job_id})")
        
        # Step 1: Generate embedding for Job Description
        self.stdout.write(f"     📊 Generating embedding...")
        jd_embedding = embedding_manager.embedding_provider.embed_query(job.description)
        
        # Step 2: Store embedding in ChromaDB with metadata
        embedding_id = f"job_{job.job_id}"
        
        # Delete existing embedding if reprocessing
        if force_reprocess and job.embedding_id:
            try:
                embedding_manager.collection.delete(ids=[job.embedding_id])
            except:
                pass  # Ignore if not found
        
        # Step 2: Store embedding in ChromaDB with metadata
        # Store in ChromaDB (using same collection as candidates but with job metadata)
        embedding_manager.collection.add(
            embeddings=[jd_embedding],
            documents=[job.description],
            metadatas=[{
                'job_id': job.job_id,
                'job_title': job.title,
                'company': job.company or '',
                'location': job.location or '',
                'type': 'job_description'  # Distinguish from candidate chunks
            }],
            ids=[embedding_id]
        )
        
        # Step 3: Verify embedding was saved
        try:
            stored = embedding_manager.collection.get(ids=[embedding_id])
            if stored['ids'] and len(stored['ids']) > 0:
                embedding_dimension = len(stored['embeddings'][0]) if stored['embeddings'] else 0
                self.stdout.write(f"     ✓ Verified: Embedding saved ({embedding_dimension} dimensions)")
            else:
                raise Exception("Embedding not found after saving")
        except Exception as e:
            raise Exception(f"Failed to verify embedding storage: {str(e)}")
        
        # Step 4: Update job status in Django DB
        job.embedding_processed = True
        job.processed_at = timezone.now()
        job.embedding_id = embedding_id
        job.save()
        
        self.stdout.write(self.style.SUCCESS(
            f"     ✅ Done: Job embedding stored in ChromaDB (ID: {embedding_id})"
        ))
        self.stdout.write(f"     📌 Django DB updated: embedding_processed=True, embedding_id={embedding_id}")

