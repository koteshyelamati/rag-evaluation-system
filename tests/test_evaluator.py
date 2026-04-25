import os
from unittest.mock import MagicMock, patch

import pandas as pd


def _mock_ragas_result(scores: dict):
    """Mock that behaves like a ragas 0.4.x EvaluationResult."""
    df = pd.DataFrame([scores])
    result = MagicMock()
    result.to_pandas.return_value = df
    result.__getitem__ = lambda self, k: scores.get(k)
    return result


def _patch_evaluator_init(test_fn):
    """Decorator that patches all __init__ dependencies for RAGEvaluator."""
    return (
        patch("app.evaluator.genai")
    )(
        patch("app.evaluator.llm_factory")(
            patch("app.evaluator.GoogleEmbeddings")(test_fn)
        )
    )


class TestRAGEvaluator:
    @patch("app.evaluator.GoogleEmbeddings")
    @patch("app.evaluator.llm_factory")
    @patch("app.evaluator.genai")
    def test_evaluate_single_returns_dict(self, mock_genai, mock_lf, mock_ge):
        mock_lf.return_value = MagicMock()
        mock_ge.return_value = MagicMock()

        with patch("app.evaluator.RAGEvaluator._build_metrics") as mock_m, \
             patch("app.evaluator.RAGEvaluator._run") as mock_run:

            mock_m.return_value = []
            mock_run.return_value = {"faithfulness": 0.85, "answer_relevancy": 0.90}

            from app.evaluator import RAGEvaluator
            ev = RAGEvaluator()
            result = ev.evaluate_single(
                question="What is ML?",
                answer="ML is a subset of AI.",
                contexts=["ML context"],
            )

        assert isinstance(result, dict)

    @patch("app.evaluator.GoogleEmbeddings")
    @patch("app.evaluator.llm_factory")
    @patch("app.evaluator.genai")
    def test_evaluate_single_scores_between_0_and_1(self, mock_genai, mock_lf, mock_ge):
        mock_lf.return_value = MagicMock()
        mock_ge.return_value = MagicMock()

        with patch("app.evaluator.RAGEvaluator._build_metrics") as mock_m, \
             patch("app.evaluator.RAGEvaluator._run") as mock_run:

            mock_m.return_value = []
            mock_run.return_value = {"faithfulness": 0.75, "answer_relevancy": 0.82}

            from app.evaluator import RAGEvaluator
            ev = RAGEvaluator()
            result = ev.evaluate_single(
                question="What is deep learning?",
                answer="Deep learning uses neural networks.",
                contexts=["DL context"],
            )

        for score in result.values():
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    @patch("app.evaluator.GoogleEmbeddings")
    @patch("app.evaluator.llm_factory")
    @patch("app.evaluator.genai")
    def test_evaluate_single_with_ground_truth_uses_retrieval_metrics(
        self, mock_genai, mock_lf, mock_ge
    ):
        mock_lf.return_value = MagicMock()
        mock_ge.return_value = MagicMock()

        with patch("app.evaluator.RAGEvaluator._build_metrics") as mock_m, \
             patch("app.evaluator.RAGEvaluator._run") as mock_run:

            mock_m.return_value = []
            mock_run.return_value = {
                "faithfulness": 0.80,
                "answer_relevancy": 0.85,
                "context_precision": 0.70,
                "context_recall": 0.75,
            }

            from app.evaluator import RAGEvaluator
            ev = RAGEvaluator()
            result = ev.evaluate_single(
                question="What is RAG?",
                answer="RAG combines retrieval with generation.",
                contexts=["RAG context"],
                ground_truth="RAG is retrieval-augmented generation.",
            )

            mock_m.assert_called_once_with(include_retrieval=True)

        for score in result.values():
            assert 0.0 <= score <= 1.0

    @patch("app.evaluator.GoogleEmbeddings")
    @patch("app.evaluator.llm_factory")
    @patch("app.evaluator.genai")
    def test_evaluate_batch_returns_aggregate_and_individual(
        self, mock_genai, mock_lf, mock_ge
    ):
        mock_lf.return_value = MagicMock()
        mock_ge.return_value = MagicMock()

        with patch("app.evaluator.RAGEvaluator._build_metrics") as mock_m, \
             patch("app.evaluator.RAGEvaluator._run") as mock_run:

            mock_m.return_value = []
            mock_run.return_value = {
                "faithfulness": 0.78,
                "answer_relevancy": 0.88,
                "context_precision": 0.72,
                "context_recall": 0.69,
            }

            from app.evaluator import RAGEvaluator
            ev = RAGEvaluator()
            qa_pairs = [
                {
                    "question": "What is ML?",
                    "answer": "ML is AI.",
                    "contexts": ["context"],
                    "ground_truth": "ML is machine learning.",
                },
                {
                    "question": "What is DL?",
                    "answer": "DL uses neural nets.",
                    "contexts": ["dl context"],
                    "ground_truth": "DL is deep learning.",
                },
            ]
            result = ev.evaluate_batch(qa_pairs=qa_pairs)

        assert "aggregate" in result
        assert "individual" in result
        assert isinstance(result["aggregate"], dict)
        assert isinstance(result["individual"], list)
        assert len(result["individual"]) == 2

    @patch("app.evaluator.GoogleEmbeddings")
    @patch("app.evaluator.llm_factory")
    @patch("app.evaluator.genai")
    def test_evaluate_batch_aggregate_scores_in_range(
        self, mock_genai, mock_lf, mock_ge
    ):
        mock_lf.return_value = MagicMock()
        mock_ge.return_value = MagicMock()

        with patch("app.evaluator.RAGEvaluator._build_metrics") as mock_m, \
             patch("app.evaluator.RAGEvaluator._run") as mock_run:

            mock_m.return_value = []
            mock_run.return_value = {
                "faithfulness": 0.65,
                "answer_relevancy": 0.72,
                "context_precision": 0.58,
                "context_recall": 0.81,
            }

            from app.evaluator import RAGEvaluator
            ev = RAGEvaluator()
            result = ev.evaluate_batch()

        for score in result["aggregate"].values():
            assert 0.0 <= score <= 1.0

    @patch("app.evaluator.GoogleEmbeddings")
    @patch("app.evaluator.llm_factory")
    @patch("app.evaluator.genai")
    def test_evaluate_single_returns_empty_on_failure(
        self, mock_genai, mock_lf, mock_ge
    ):
        mock_lf.return_value = MagicMock()
        mock_ge.return_value = MagicMock()

        with patch("app.evaluator.RAGEvaluator._build_metrics") as mock_m, \
             patch("app.evaluator.RAGEvaluator._run") as mock_run:

            mock_m.return_value = []
            mock_run.side_effect = RuntimeError("Ragas API error")

            from app.evaluator import RAGEvaluator
            ev = RAGEvaluator()
            result = ev.evaluate_single(
                question="q",
                answer="a",
                contexts=["c"],
            )

        assert result == {}

    @patch("app.evaluator.GoogleEmbeddings")
    @patch("app.evaluator.llm_factory")
    @patch("app.evaluator.genai")
    def test_evaluate_batch_uses_default_qa_pairs(
        self, mock_genai, mock_lf, mock_ge
    ):
        mock_lf.return_value = MagicMock()
        mock_ge.return_value = MagicMock()

        with patch("app.evaluator.RAGEvaluator._build_metrics") as mock_m, \
             patch("app.evaluator.RAGEvaluator._run") as mock_run:

            mock_m.return_value = []
            mock_run.return_value = {"faithfulness": 0.9, "answer_relevancy": 0.85}

            from app.evaluator import RAGEvaluator
            from app.document_loader import DEFAULT_QA_PAIRS

            ev = RAGEvaluator()
            result = ev.evaluate_batch()

        assert result["individual"] is DEFAULT_QA_PAIRS
        assert len(result["individual"]) == 10
