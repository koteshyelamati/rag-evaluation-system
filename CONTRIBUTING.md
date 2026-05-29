# Contributing to rag-evaluation-system

Thank you for your interest in contributing! This guide covers everything you need to get the project running locally, pass the test suite, and submit high-quality pull requests.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Code Style](#code-style)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Bugs](#reporting-bugs)

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Python | 3.11 |
| pip | 23+ |
| Git | 2.40+ |

You will need a **Google Gemini API key** (free tier works) to run the full pipeline. Tests run with mocked dependencies — no API key required for `pytest`.

---

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/koteshyelamati/rag-evaluation-system.git
cd rag-evaluation-system

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Environment Variables

Copy the example file and fill in your API key:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | — | Google Gemini API key |
| `CHROMA_PERSIST_DIR` | No | `./chroma_db` | Directory for the persistent vector store |
| `COLLECTION_NAME` | No | `rag_docs` | ChromaDB collection name |
| `CHUNK_SIZE` | No | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | No | `200` | Overlap between consecutive chunks |
| `TOP_K_RESULTS` | No | `5` | Number of chunks retrieved per query |
| `QUERY_CACHE_MAXSIZE` | No | `256` | Maximum entries in the in-memory query cache |
| `QUERY_CACHE_TTL_SECONDS` | No | `3600` | Cache entry TTL in seconds (0 = no expiry) |

---

## Running the Application

```bash
uvicorn app.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser. Sample AI/ML documents are indexed automatically on first launch.

### Useful API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check + cache stats |
| `POST` | `/api/query` | Ask a question |
| `POST` | `/api/ingest` | Ingest new documents |
| `POST` | `/api/evaluate` | Run batch Ragas evaluation |
| `GET` | `/api/cache/stats` | Query cache hit/miss metrics |
| `POST` | `/api/cache/clear` | Flush the query cache |

---

## Running Tests

All tests use mocked dependencies — **no API keys or network access required**.

```bash
# Run the full suite
pytest tests/ -v

# Run a single test file
pytest tests/test_cache.py -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=term-missing
```

> **Tip:** Install `pytest-cov` first if you want coverage: `pip install pytest-cov`

---

## Project Structure

```
rag-evaluation-system/
├── app/
│   ├── __init__.py
│   ├── cache.py            # Thread-safe LRU query cache
│   ├── config.py           # Pydantic settings (env vars)
│   ├── document_loader.py  # PDF/TXT ingestion, chunking, embedding
│   ├── evaluator.py        # Ragas evaluation (faithfulness, relevancy, …)
│   ├── main.py             # FastAPI routes and app lifespan
│   └── rag_pipeline.py     # LangChain RetrievalQA chain + cache integration
├── frontend/
│   └── index.html          # Chat UI and evaluation dashboard (Chart.js)
├── tests/
│   ├── conftest.py
│   ├── test_cache.py       # QueryCache unit tests
│   ├── test_evaluator.py   # RAGEvaluator unit tests
│   └── test_rag_pipeline.py# RAGPipeline unit tests
├── data/
│   └── sample_docs/        # Auto-indexed sample documents
├── .env.example
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI (lint + pytest)
├── CONTRIBUTING.md         # This file
├── Makefile
├── README.md
└── requirements.txt
```

---

## Code Style

This project follows standard Python conventions:

- **Formatting**: [PEP 8](https://peps.python.org/pep-0008/). Use 4-space indentation.
- **Type hints**: Add type hints to all function signatures.
- **Docstrings**: Use triple-quoted docstrings for public classes and methods.
- **Imports**: Standard library first, then third-party, then local (`app.*`), separated by blank lines.
- **Logging**: Use `logging.getLogger(__name__)` — never `print()` in application code.
- **Tests**: Every new module should have a corresponding `tests/test_<module>.py`. All tests must pass with mocked external dependencies (no live API calls).

---

## Submitting a Pull Request

1. **Fork** the repository and create a feature branch from `main`:
   ```bash
      git checkout -b feat/your-feature-name
         ```
         2. **Write your changes** with tests.
         3. **Ensure all tests pass** locally: `pytest tests/ -v`
         4. **Commit** using a clear, conventional message:
            ```
               feat: add streaming response support to /api/query
                  fix: handle empty context_chunks in evaluate_single
                     test: add edge-case tests for QueryCache TTL expiry
                        docs: update README with new cache env vars
                           ```
                           5. **Push** your branch and open a pull request against `main`.
                           6. Fill in the PR description explaining *what* changed and *why*.

                           ---

                           ## Reporting Bugs

                           Please open a [GitHub Issue](https://github.com/koteshyelamati/rag-evaluation-system/issues) with:

                           - A clear title and description of the problem
                           - Steps to reproduce
                           - Expected vs. actual behaviour
                           - Python version and OS
                           - Relevant log output or error traceback

                           ---

                           *Happy contributing!*
