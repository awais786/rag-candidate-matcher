# RAG Chatbot - Candidate Matching System

A Django-based RAG (Retrieval-Augmented Generation) system for matching job descriptions with candidate CVs using embeddings and LLM scoring.

## Quick Start

```bash
# 1. Process candidates (CVs)
python3 manage.py process_candidates

# 2. Process jobs (Job descriptions)
python3 manage.py process_jobs

# 3. Vector search - find matching candidates
python3 manage.py search_candidates --job-id 1 --top-n 10

# 4. LLM scoring - final ranking with explanations
IDS=$(python3 manage.py search_candidates --job-id 1 --top-n 10 --output ids)
python3 manage.py score_with_llm --job-id 1 --candidate-ids "$IDS"
```

## Core Commands

### 1. Process Candidates
```bash
python3 manage.py process_candidates [--batch-size 10]
```
- Processes all candidates from Django `Candidate` table
- Generates embeddings for CV files (PDF/TXT)
- Stores embeddings in ChromaDB
- Metadata stored: `candidate_id`, `candidate_name`, `email`, `type: 'candidate_chunk'`

### 2. Process Jobs
```bash
python3 manage.py process_jobs
```
- Processes all jobs from Django `Job` table
- Generates embeddings for job descriptions
- Stores embeddings in ChromaDB
- Metadata stored: `job_id`, `job_title`, `company`, `location`, `type: 'job_description'`

### 3. Vector Search (ChromaDB)
```bash
# Using Job ID (recommended)
python3 manage.py search_candidates --job-id 1 --top-n 10

# Using query text
python3 manage.py search_candidates --query "Senior Python Developer" --top-n 10

# Output formats
--output summary  # Human-readable summary (default)
--output table    # Table format
--output json     # JSON format
--output ids      # Comma-separated candidate IDs (for LLM step)
```

**Options:**
- `--job-id <id>`: Use stored Job from database
- `--query <text>`: Use custom query text
- `--top-n <number>`: Number of results (default: 10)
- `--min-similarity <float>`: Minimum similarity threshold 0-1 (default: 0.0)
- `--output <format>`: Output format (summary/table/json/ids)

### 4. LLM Scoring (OpenAI)
```bash
# Option A: Using candidate IDs from vector search (recommended)
python3 manage.py score_with_llm --job-id 1 --candidate-ids "1,2,3,4,5"

# Option B: Using saved JSON file
python3 manage.py search_candidates --job-id 1 --output json > results.json
python3 manage.py score_with_llm --job-id 1 --input-file results.json

# Option C: Auto-run vector search + LLM scoring
python3 manage.py score_with_llm --job-id 1 --top-n 10
```

**Options:**
- `--job-id <id>`: Job ID (required)
- `--candidate-ids <ids>`: Comma-separated candidate IDs from `search_candidates --output ids`
- `--input-file <path>`: JSON file from `search_candidates --output json`
- `--top-n <number>`: Number of candidates (if not using --candidate-ids or --input-file)
- `--output <format>`: Output format (summary/table/json)

## Complete Workflow Example

```bash
# Step 1: Process all candidates
python3 manage.py process_candidates

# Step 2: Process all jobs
python3 manage.py process_jobs

# Step 3: Vector search - get candidate IDs
IDS=$(python3 manage.py search_candidates --job-id 1 --top-n 10 --output ids)
echo "Found candidates: $IDS"

# Step 4: LLM scoring - final ranking
python3 manage.py score_with_llm --job-id 1 --candidate-ids "$IDS"
```

## Architecture

```
┌─────────────────┐
│  Candidate CVs  │
│  (PDF/TXT)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│ process_candidates │───▶│  ChromaDB    │
└─────────────────┘     │ (Embeddings) │
                        └──────┬───────┘
                               │
┌─────────────────┐            │
│  Job Descriptions│           │
└────────┬────────┘            │
         │                     │
         ▼                     │
┌─────────────────┐            │
│ process_jobs    │────────────┘
└─────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│ search_candidates│───▶│  Top N       │
│ (Vector Search) │     │  Candidates  │
└────────┬────────┘     └──────┬───────┘
         │                     │
         │ (IDs)               │
         ▼                     │
┌─────────────────┐            │
│ score_with_llm  │◀───────────┘
│ (OpenAI LLM)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Ranked   │
│  Candidates     │
└─────────────────┘
```

## Embedding Providers

The system supports multiple embedding providers with automatic fallback:

### Voyage AI (Default)
- High-quality embeddings (1024 dimensions)
- Requires `VOYAGE_API_KEY` in settings
- **Automatic fallback to local if API unavailable**

### OpenAI
- Good quality embeddings
- Requires `OPENAI_API_KEY` in settings

### Local HuggingFace (Fallback)
- Works offline, no API needed
- Default model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Automatic fallback if Voyage/OpenAI fail**

### Configuration

Edit `rag_chatbot/settings.py`:

```python
# Auto-detect (checks Voyage → OpenAI → Local)
EMBEDDING_PROVIDER = None

# Or explicitly set
EMBEDDING_PROVIDER = "voyage"  # or "openai" or "local"

# Voyage AI
VOYAGE_API_KEY = "your-api-key"
VOYAGE_MODEL = "voyage-2"

# OpenAI
OPENAI_API_KEY = "your-api-key"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Local HuggingFace
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### Automatic Fallback

The system automatically falls back to local HuggingFace if:
- Voyage API key is missing
- Voyage API initialization fails
- Network errors occur
- API rate limits/timeouts happen
- Runtime embedding generation fails

You'll see warnings when fallback occurs:
```
UserWarning: Voyage API failed. Falling back to local HuggingFace embeddings.
```

## How It Works

### Vector Search (Step 3)
1. Job description is embedded (or retrieved from ChromaDB if job_id provided)
2. ChromaDB searches candidate embeddings using cosine similarity
3. Similarity scores are aggregated across candidate chunks:
   - 40% weighted average (emphasizes strong matches)
   - 30% max similarity (best matching chunk)
   - 20% top-k average (average of top 3 chunks)
   - 10% overall average
4. Candidates ranked by `overall_relevance` score

### LLM Scoring (Step 4)
1. Top candidates from vector search are sent to OpenAI LLM
2. LLM analyzes job description + candidate CV
3. Returns:
   - LLM score (0-1)
   - Explanation
   - Strengths
   - Weaknesses/gaps
4. Final score combines: `60% LLM score + 40% ChromaDB relevance`

## File Structure

```
rag_chatbot/
├── documents/
│   ├── models.py              # Candidate, Job models
│   ├── utils.py               # PDFProcessor, EmbeddingManager
│   ├── candidate_matcher.py   # match_jd_against_candidates()
│   ├── llm_scorer.py         # LLMCandidateScorer
│   ├── embeddings/
│   │   ├── factory.py        # EmbeddingFactory (Voyage/OpenAI/Local)
│   │   ├── voyage_provider.py
│   │   ├── openai_provider.py
│   │   └── local_provider.py
│   └── management/commands/
│       ├── process_candidates.py
│       ├── process_jobs.py
│       ├── search_candidates.py
│       └── score_with_llm.py
└── rag_chatbot/
    └── settings.py            # Embedding configuration
```

## Django Models

### Candidate
```python
class Candidate(models.Model):
    candidate_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    email = models.EmailField()
    cv_file = models.FileField(upload_to='cvs/')
    embedding_processed = models.BooleanField(default=False)
    # ...
```

### Job
```python
class Job(models.Model):
    job_id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=255)
    description = models.TextField()
    company = models.CharField(max_length=255, null=True, blank=True)
    location = models.CharField(max_length=255, null=True, blank=True)
    embedding_processed = models.BooleanField(default=False)
    embedding_id = models.CharField(max_length=255, null=True, blank=True)
    # ...
```

## Key Features

- **Embedding Provider Flexibility**: Voyage AI / OpenAI / Local HuggingFace with automatic fallback
- **Vector Database**: ChromaDB with local persistence
- **Similarity Matching**: Cosine similarity for semantic search
- **LLM Integration**: OpenAI GPT for nuanced scoring and explanations
- **Optimized Workflow**: Vector search → LLM scoring (no redundant queries)
- **Batch Processing**: Efficient processing of large candidate pools
- **Error Handling**: Automatic fallback to local embeddings if APIs fail

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- Django
- langchain
- chromadb
- sentence-transformers (for local embeddings)
- voyageai (optional, for Voyage API)
- openai (optional, for OpenAI embeddings/LLM)

## Environment Variables

Set in `rag_chatbot/settings.py` or use environment variables:

```bash
export EMBEDDING_PROVIDER=voyage  # or openai or local
export VOYAGE_API_KEY=your-key
export OPENAI_API_KEY=your-key
```

## Troubleshooting

### ChromaDB Dimension Mismatch
If you switch embedding providers (e.g., Voyage → Local), ChromaDB will detect dimension mismatch and automatically recreate the collection with the new dimension.

### API Failures
The system automatically falls back to local HuggingFace if Voyage/OpenAI APIs fail. Check warnings in console output.

### Processing Status
Check `embedding_processed` flag in Candidate/Job models to see which records have been processed.

## License

[Your License Here]

# rag-candidate-matcher
