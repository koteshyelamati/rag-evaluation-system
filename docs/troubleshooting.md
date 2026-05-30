# Troubleshooting Guide

## Common Issues

### 1. OpenAI API Key Error

**Error:** `AuthenticationError: No API key provided`

**Fix:**
- Make sure you have a `.env` file with `OPENAI_API_KEY=your-key-here`
- Verify the key is valid at https://platform.openai.com/api-keys
- Ensure you haven't exceeded your usage quota

### 2. ChromaDB Persistence Issues

**Error:** `Collection not found` or empty responses on restart

**Fix:**
- Check that `CHROMA_PERSIST_DIR` in `.env` points to a writable directory
- If the directory is corrupted, delete it and restart (documents will be re-indexed)
- Ensure you have write permissions on the directory

### 3. Low Faithfulness Scores

**Symptom:** Faithfulness score consistently below 0.4

**Fix:**
- Reduce `CHUNK_SIZE` to keep context more focused
- Increase `CHUNK_OVERLAP` to avoid cutting off context at chunk boundaries
- Review your prompt template in `app/rag_pipeline.py`

### 4. Slow Query Response

**Symptom:** Queries taking more than 10 seconds

**Fix:**
- Reduce `TOP_K_RESULTS` from 5 to 3
- Consider using a faster OpenAI model
- Check your network connection to the OpenAI API

### 5. Port Already in Use

**Error:** `Address already in use: port 8000`

**Fix:**
```bash
# Find and kill the process using port 8000
lsof -i :8000
kill -9 <PID>

# Or use a different port
uvicorn app.main:app --reload --port 8001
```

## Getting Help

Open an issue on GitHub with:
- Your Python version (`python --version`)
- The full error traceback
- Steps to reproduce
