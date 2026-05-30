# Deployment Guide

## Local Development

### Prerequisites
- Python 3.11+
- OpenAI API key

### Steps

1. Clone the repository:
```bash
git clone https://github.com/koteshyelamati/rag-evaluation-system
cd rag-evaluation-system
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

5. Run the server:
```bash
uvicorn app.main:app --reload --port 8000
```

6. Open http://localhost:8000 in your browser.

## Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| OPENAI_API_KEY | Yes | - | Your OpenAI API key |
| CHROMA_PERSIST_DIR | No | ./chroma_db | Vector store path |
| CHUNK_SIZE | No | 1000 | Characters per chunk |
| TOP_K_RESULTS | No | 5 | Chunks per query |
