from django.db import models


class Candidate(models.Model):
    """Simple candidate table"""
    candidate_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    email = models.EmailField(null=True, blank=True)
    cv_file = models.FileField(upload_to="candidates/cvs/", help_text="CV PDF file", null=True, blank=True)
    
    # Processing status
    embedding_processed = models.BooleanField(default=False, db_index=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    chunks_count = models.IntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
        indexes = [
            models.Index(fields=['candidate_id']),
            models.Index(fields=['embedding_processed']),
        ]
    
    def __str__(self):
        return f"{self.name} (ID: {self.candidate_id})"


class Job(models.Model):
    """Job posting with description and embeddings"""
    job_id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=255, help_text="Job title (e.g., Senior Python Developer)")
    description = models.TextField(help_text="Full job description text")
    
    # Optional fields
    company = models.CharField(max_length=255, null=True, blank=True)
    location = models.CharField(max_length=255, null=True, blank=True)
    
    # Processing status
    embedding_processed = models.BooleanField(default=False, db_index=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    embedding_id = models.CharField(
        max_length=255, 
        null=True, 
        blank=True, 
        help_text="Reference ID to embedding in ChromaDB (embeddings NOT stored in Django DB)"
    )  # Reference ID only - actual embeddings stored in ChromaDB
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['job_id']),
            models.Index(fields=['embedding_processed']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"{self.title} (ID: {self.job_id})"
