# Production RAG System with Ragas Evaluation

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-orange)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-purple)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-red)](https://trychroma.com)
[![Ragas](https://img.shields.io/badge/Ragas-0.2-yellow)](https://ragas.io)

Most RAG projects stop at "it answers questions." This one goes further — every
query is scored automatically for hallucination, relevancy, and retrieval quality
using Ragas. The evaluation harness is what makes it production-ready rather than
just a demo.

---

## The Problem

When you deploy a RAG system without evaluation, you have no way to know if the
answers are trustworthy. Users notice bad answers before you do. This project
adds automated scoring to every query so quality issues surface immediately —
before they reach users.

---

## Architecture

Browser (Chat UI)
|
| POST /api/query
v
FastAPI Backend (app/main.py)
|
v
RAG Pipeline (app/rag_pipeline.py)
|
|-- ChromaDB (vector store, persistent)
| text-embedding-3-small embeddings
| MMR retrieval, top-5 chunks
|
|-- OpenAI GPT-4o-mini
| RetrievalQA chain
| context-grounded prompt template
|
v
Ragas Evaluator (app/evaluator.py)
faithfulness / answer relevancy / context precision / context recall
|
v
Response: answer + quality scores

text

---

## Features

- Document ingestion for PDF and TXT files with configurable chunking and overlap
- Semantic search via ChromaDB with OpenAI embeddings
- MMR (Maximal Marginal Relevance) retrieval to reduce redundant context
- Real-time Ragas scoring on every query — not just in batch
- Batch evaluation endpoint that runs the full Ragas suite against a test dataset
- Observability dashboard built with Chart.js showing metric trends
- Sample AI/ML documents auto-indexed on first launch so the system works immediately

---

## Setup

**Requirements:** Python 3.11+, an OpenAI API key

```bash
git clone https://github.com/koteshyelamati/rag-evaluation-system
cd rag-evaluation-system

python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY to .env

uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 in your browser. Sample documents are indexed on
first launch — no manual ingestion needed to start querying.

---

## Project Structure

rag-evaluation-system/
├── app/
│ ├── main.py # FastAPI routes and startup
│ ├── rag_pipeline.py # LangChain RetrievalQA chain
│ ├── evaluator.py # Ragas evaluation (OpenAI as judge LLM)
│ ├── document_loader.py # Ingestion, chunking, embedding
│ └── config.py # Settings via pydantic-settings
├── frontend/
│ └── index.html # Chat UI and evaluation dashboard
├── tests/
│ ├── test_rag_pipeline.py
│ └── test_evaluator.py
├── data/
│ └── sample_docs/
├── requirements.txt
├── .env.example
└── Makefile

text

---

## API Reference

**GET /api/health**
Returns status and current document count.

**POST /api/query**

```json
// request
{ "question": "What is retrieval-augmented generation?" }

// response
{
  "answer": "RAG is a technique that combines...",
  "source_documents": ["chunk 1...", "chunk 2..."],
  "evaluation": {
    "faithfulness": 0.91,
    "answer_relevancy": 0.87
  }
}
```

**POST /api/ingest**

```json
{ "file_paths": ["./data/my_document.pdf"] }
```

**POST /api/evaluate**
Runs full Ragas batch evaluation against the built-in test dataset and returns
aggregate scores for all four metrics.

---

## Ragas Metrics

**Faithfulness** — checks whether every factual claim in the answer is directly
supported by the retrieved context. High faithfulness means the model is not
making things up.

**Answer Relevancy** — measures whether the answer actually addresses the
question. A response can be factually correct but still score low here if it
drifts off topic.

**Context Precision** — checks whether the most relevant document chunks are
ranked at the top of the retrieved results. Poor precision means useful content
is being buried behind noise.

**Context Recall** — measures whether the retrieved context contains all the
information needed to construct a complete answer. Low recall means the retriever
is missing relevant chunks.

Score interpretation: above 0.7 is production-ready, 0.4 to 0.7 is acceptable
but worth monitoring, below 0.4 indicates the pipeline needs tuning.

---

## Configuration

| Variable           | Default     | Description                |
| ------------------ | ----------- | -------------------------- |
| OPENAI_API_KEY     | required    | Your OpenAI API key        |
| CHROMA_PERSIST_DIR | ./chroma_db | Vector store location      |
| COLLECTION_NAME    | rag_docs    | ChromaDB collection name   |
| CHUNK_SIZE         | 1000        | Characters per chunk       |
| CHUNK_OVERLAP      | 200         | Overlap between chunks     |
| TOP_K_RESULTS      | 5           | Chunks retrieved per query |

---

## Running Tests

```bash
pytest tests/ -v
```

All tests use mocked dependencies. No API calls or costs are incurred during
test runs.

---

## Tech Stack

- FastAPI — REST API and static file serving
- LangChain — RAG chain orchestration and prompt management
- ChromaDB — local persistent vector store
- OpenAI — GPT-4o-mini for generation, text-embedding-3-small for embeddings
- Ragas — evaluation metrics for RAG pipelines
- Chart.js — frontend metric visualizations
