import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)

_rag_pipeline = None
_evaluator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_pipeline, _evaluator
    logger.info("Starting RAG Evaluation System...")

    from app.document_loader import get_document_count, load_sample_data
    from app.rag_pipeline import RAGPipeline
    from app.evaluator import RAGEvaluator

    if settings.GEMINI_API_KEY:
        try:
            doc_count = get_document_count()
            if doc_count == 0:
                logger.info("Vector store is empty — loading sample data...")
                load_sample_data()
        except Exception as exc:
            logger.warning("Could not check/load sample data: %s", exc)

        try:
            _rag_pipeline = RAGPipeline()
            _evaluator = RAGEvaluator()
            logger.info("RAG pipeline and evaluator ready.")
        except Exception as exc:
            logger.error("Failed to initialize pipeline: %s", exc)
    else:
        logger.warning("GEMINI_API_KEY not set — pipeline not initialized.")

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="RAG Evaluation System",
    description="Production RAG with Ragas evaluation — powered by Gemini + LangChain",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    file_paths: list[str]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = Path(__file__).parent.parent / "frontend" / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(content=index.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    from app.document_loader import get_document_count

    doc_count = 0
    try:
        doc_count = get_document_count()
    except Exception:
        pass

    return {
        "status": "healthy",
        "document_count": doc_count,
        "pipeline_ready": _rag_pipeline is not None,
        "evaluator_ready": _evaluator is not None,
    }


@app.post("/api/query")
async def query(request: QueryRequest):
    if not _rag_pipeline:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not ready. Check GEMINI_API_KEY and restart.",
        )

    rag_result = _rag_pipeline.query(request.question)

    eval_scores: dict = {}
    if _evaluator and rag_result["context_chunks"]:
        try:
            eval_scores = _evaluator.evaluate_single(
                question=request.question,
                answer=rag_result["answer"],
                contexts=rag_result["context_chunks"],
            )
        except Exception as exc:
            logger.warning("Evaluation failed for query: %s", exc)

    return {
        "answer": rag_result["answer"],
        "source_documents": rag_result["source_documents"],
        "context_chunks": rag_result["context_chunks"],
        "evaluation": eval_scores,
    }


@app.post("/api/ingest")
async def ingest(request: IngestRequest):
    if not settings.GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured.")

    from app.document_loader import load_and_index_documents

    try:
        count = load_and_index_documents(request.file_paths)
        return {"status": "success", "chunks_indexed": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/evaluate")
async def evaluate():
    if not _evaluator:
        raise HTTPException(
            status_code=503,
            detail="Evaluator not ready. Check GEMINI_API_KEY and restart.",
        )

    try:
        results = _evaluator.evaluate_batch()
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
