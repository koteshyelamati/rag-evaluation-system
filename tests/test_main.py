"""Integration tests for app/main.py FastAPI routes.

All external dependencies (RAGPipeline, RAGEvaluator, document_loader)
are fully mocked so no API keys or ChromaDB are needed.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(pipeline=None, evaluator=None):
      """Return a TestClient with _rag_pipeline and _evaluator injected."""
      import app.main as main_module

    main_module._rag_pipeline = pipeline
    main_module._evaluator = evaluator

    from app.main import app
    return TestClient(app, raise_server_exceptions=True)


def _default_pipeline():
      p = MagicMock()
      p.query.return_value = {
          "answer": "RAG combines retrieval with generation.",
          "source_documents": ["chunk 1", "chunk 2"],
          "context_chunks": ["chunk 1", "chunk 2"],
      }
      p.cache_stats.return_value = {
          "size": 1, "maxsize": 256, "hits": 3, "misses": 1,
          "hit_rate": 0.75, "ttl_seconds": 3600,
      }
      return p


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
      def test_health_returns_200(self):
                with patch("app.main.get_document_count", return_value=5):
                              client = _make_app(pipeline=_default_pipeline())
                              resp = client.get("/api/health")
                          assert resp.status_code == 200

      def test_health_schema(self):
                with patch("app.main.get_document_count", return_value=5):
                              client = _make_app(pipeline=_default_pipeline())
                              data = client.get("/api/health").json()
                          assert data["status"] == "healthy"
                assert "pipeline_ready" in data
                assert "evaluator_ready" in data
                assert "document_count" in data

      def test_health_pipeline_ready_true(self):
                with patch("app.main.get_document_count", return_value=0):
                              client = _make_app(pipeline=_default_pipeline())
                              data = client.get("/api/health").json()
                          assert data["pipeline_ready"] is True

      def test_health_pipeline_ready_false_when_none(self):
                with patch("app.main.get_document_count", return_value=0):
                              client = _make_app(pipeline=None)
                              data = client.get("/api/health").json()
                          assert data["pipeline_ready"] is False

      def test_health_includes_cache_stats(self):
                with patch("app.main.get_document_count", return_value=5):
                              client = _make_app(pipeline=_default_pipeline())
                              data = client.get("/api/health").json()
                          assert "cache" in data
                assert data["cache"]["hits"] == 3


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
      def test_query_returns_200(self):
                client = _make_app(pipeline=_default_pipeline())
                resp = client.post("/api/query", json={"question": "What is RAG?"})
                assert resp.status_code == 200

      def test_query_response_schema(self):
                client = _make_app(pipeline=_default_pipeline())
                data = client.post("/api/query", json={"question": "What is RAG?"}).json()
                assert "answer" in data
                assert "source_documents" in data
                assert "evaluation" in data

      def test_query_answer_content(self):
                client = _make_app(pipeline=_default_pipeline())
                data = client.post("/api/query", json={"question": "What is RAG?"}).json()
                assert "RAG" in data["answer"]

      def test_query_503_when_pipeline_not_ready(self):
                client = _make_app(pipeline=None)
                resp = client.post("/api/query", json={"question": "hello"})
                assert resp.status_code == 503

      def test_query_missing_question_returns_422(self):
                client = _make_app(pipeline=_default_pipeline())
                resp = client.post("/api/query", json={})
                assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/cache/stats
# ---------------------------------------------------------------------------

class TestCacheStatsEndpoint:
      def test_cache_stats_returns_200(self):
                client = _make_app(pipeline=_default_pipeline())
                resp = client.get("/api/cache/stats")
                assert resp.status_code == 200

      def test_cache_stats_schema(self):
                client = _make_app(pipeline=_default_pipeline())
                data = client.get("/api/cache/stats").json()
                for key in ("size", "maxsize", "hits", "misses", "hit_rate"):
                              assert key in data, f"Missing key: {key}"

            def test_cache_stats_503_without_pipeline(self):
                      client = _make_app(pipeline=None)
                      resp = client.get("/api/cache/stats")
                      assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/cache/clear
# ---------------------------------------------------------------------------

class TestCacheClearEndpoint:
      def test_cache_clear_returns_200(self):
                client = _make_app(pipeline=_default_pipeline())
                resp = client.post("/api/cache/clear")
                assert resp.status_code == 200

    def test_cache_clear_response_schema(self):
              client = _make_app(pipeline=_default_pipeline())
              data = client.post("/api/cache/clear").json()
              assert "message" in data
              assert "cleared_entries" in data
              assert "previous_stats" in data

    def test_cache_clear_calls_pipeline_clear(self):
              pipeline = _default_pipeline()
              client = _make_app(pipeline=pipeline)
              client.post("/api/cache/clear")
              pipeline.clear_cache.assert_called_once()

    def test_cache_clear_503_without_pipeline(self):
              client = _make_app(pipeline=None)
              resp = client.post("/api/cache/clear")
              assert resp.status_code == 503
