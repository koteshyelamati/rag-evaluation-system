# Project Roadmap

This document outlines planned improvements and future directions for the RAG Evaluation System.

## Short-term (Next 1-2 Months)

- [ ] **Multi-file upload UI** — Drag-and-drop interface for ingesting documents directly from the browser
- [ ] **Evaluation history** — Persist and display metric trends over time in the dashboard
- [ ] **Streaming responses** — Stream GPT-4o-mini output token-by-token for better UX
- [ ] **Export evaluation report** — Download a PDF/CSV summary of evaluation results

## Medium-term (3-6 Months)

- [ ] **Multi-collection support** — Separate ChromaDB collections for different document domains
- [ ] **Hybrid search** — Combine dense vector search with BM25 sparse retrieval for better recall
- [ ] **Configurable LLM** — Support switching between OpenAI, Anthropic Claude, and local models (Ollama)
- [ ] **Authentication** — Add API key-based auth for multi-user deployment
- [ ] **LangSmith integration** — Full tracing and observability via LangSmith

## Long-term (6+ Months)

- [ ] **Automated prompt optimization** — Use DSPy or LangChain Expression Language to auto-tune prompts based on Ragas scores
- [ ] **A/B testing framework** — Compare two RAG configurations side-by-side with statistical significance testing
- [ ] **Cloud deployment** — One-click AWS/GCP/Azure deployment via Terraform or Pulumi
- [ ] **Kubernetes Helm chart** — Production-ready Helm chart with horizontal pod autoscaling

## Completed

- [x] FastAPI backend with query, ingest, and evaluate endpoints
- [x] ChromaDB vector store with OpenAI embeddings
- [x] Real-time Ragas scoring on every query
- [x] Batch evaluation endpoint
- [x] Chart.js observability dashboard
- [x] MMR retrieval to reduce redundant context
- [x] Sample documents auto-indexed on first launch
