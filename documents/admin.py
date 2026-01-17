from django.contrib import admin
from .models import Candidate, Job


@admin.register(Candidate)
class CandidateAdmin(admin.ModelAdmin):
    list_display = ['candidate_id', 'name', 'email', 'embedding_processed', 'chunks_count', 'processed_at']
    list_filter = ['embedding_processed', 'created_at']
    search_fields = ['name', 'email']
    readonly_fields = ['candidate_id', 'created_at', 'updated_at']
    
    fieldsets = (
        ("Candidate Information", {
            "fields": ("name", "email", "cv_file")
        }),
        ("Processing Status", {
            "fields": ("embedding_processed", "processed_at", "chunks_count")
        }),
    )


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ['job_id', 'title', 'company', 'embedding_processed', 'processed_at', 'created_at']
    list_filter = ['embedding_processed', 'created_at']
    search_fields = ['title', 'description', 'company']
    readonly_fields = ['job_id', 'embedding_id', 'created_at', 'updated_at']
    
    fieldsets = (
        ("Job Information", {
            "fields": ("title", "description", "company", "location")
        }),
        ("Processing Status", {
            "fields": ("embedding_processed", "processed_at", "embedding_id")
        }),
    )
