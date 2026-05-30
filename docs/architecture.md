# Architecture Overview

## System Design

The RAG Evaluation System follows a layered architecture:

### 1. API Layer (FastAPI)
- Handles HTTP requests from the frontend
- Routes: `/api/query`, `/api/ingest`, `/api/evaluate`, `/api/health`
- Manages document upload and processing

### 2. RAG Pipeline (LangChain)
- Orchestrates the retrieval-augmented generation flow
- Uses RetrievalQA chain with custom prompt templates
- MMR retrieval strategy to reduce redundant context

### 3. Vector Store (ChromaDB)
- Persists document embeddings locally
- Uses OpenAI `text-embedding-3-small` for embedding
- Supports MMR and similarity search

### 4. Evaluation Layer (Ragas)
- Scores every query in real-time
- Metrics: faithfulness, answer relevancy, context precision, context recall
- Batch evaluation against a test dataset

### 5. Frontend (Chart.js + HTML)
- Chat UI for querying the system
- Dashboard for visualizing evaluation metrics

## Data Flow

```
User Query → FastAPI → LangChain RAG Chain → ChromaDB (retrieve) → GPT-4o-mini (generate) → Ragas (evaluate) → Response
```
