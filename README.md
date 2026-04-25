# Production RAG System with Ragas Evaluation (Gemini + LangChain)

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple)
![Gemini](https://img.shields.io/badge/Gemini-2.5--flash-orange)

A production-grade Retrieval-Augmented Generation system that answers questions from a
private document corpus and continuously evaluates answer quality using Ragas metrics.
Ragas evaluation is critical for production AI — it surfaces hallucination, retrieval
quality issues, and answer degradation before users notice them.

## Architecture

```
User → FastAPI → RAG Pipeline → ChromaDB (vector store)
                      ↓                  ↑
               Gemini 2.5 Flash   GoogleGenerativeAI Embeddings
                      ↓
               Ragas Evaluator → Faithfulness / Relevancy scores
```

## Quick Start

```bash
git clone <repo>
cd rag-evaluation-system
cp .env.example .env           # edit and add your GEMINI_API_KEY
make install                   # pip install -r requirements.txt
make run                       # uvicorn on :8000
```

Open http://localhost:8000 for the chat UI.

## API Endpoints

| Method | Endpoint        | Description                           |
| ------ | --------------- | ------------------------------------- |
| GET    | `/`             | Serves the frontend SPA               |
| GET    | `/api/health`   | Status, doc count, pipeline readiness |
| POST   | `/api/query`    | `{question}` → answer + Ragas scores  |
| POST   | `/api/ingest`   | `{file_paths}` → index documents      |
| POST   | `/api/evaluate` | Run full batch evaluation on test set |

## Ragas Metrics

| Metric                | What it measures                                   | Range |
| --------------------- | -------------------------------------------------- | ----- |
| **Faithfulness**      | Is the answer supported by the retrieved context?  | 0–1   |
| **Answer Relevancy**  | Does the answer directly address the question?     | 0–1   |
| **Context Precision** | Are the most relevant chunks ranked first?         | 0–1   |
| **Context Recall**    | Does the retrieved context cover the ground truth? | 0–1   |

Scores ≥ 0.7 = green (good), 0.4–0.7 = yellow (acceptable), < 0.4 = red (needs tuning).

## Running Tests

```bash
pytest tests/ -v
```
