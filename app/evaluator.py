import logging
import math
import warnings
from typing import Optional

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from app.config import settings
from app.document_loader import DEFAULT_QA_PAIRS

logger = logging.getLogger(__name__)


class RAGEvaluator:
    def __init__(self) -> None:
        self._ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=settings.OPENAI_API_KEY,
            )
        )
        self._ragas_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=settings.OPENAI_API_KEY,
            )
        )

    def _run(self, dataset: Dataset, metrics: list) -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self._ragas_llm,
                embeddings=self._ragas_embeddings,
            )

        df = result.to_pandas()
        scores: dict = {}
        for col in df.select_dtypes("number").columns:
            val = df[col].mean()
            if not _is_nan(val):
                scores[col] = float(val)
        return scores

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> dict:
        has_gt = bool(ground_truth)
        data: dict = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if has_gt:
            data["ground_truth"] = [ground_truth]

        metric_list = [faithfulness, answer_relevancy]
        if has_gt:
            metric_list += [context_precision, context_recall]

        try:
            dataset = Dataset.from_dict(data)
            return self._run(dataset, metric_list)
        except Exception as exc:
            logger.error("Single evaluation failed: %s", exc)
            return {}

    def evaluate_batch(self, qa_pairs: Optional[list[dict]] = None) -> dict:
        if qa_pairs is None:
            qa_pairs = DEFAULT_QA_PAIRS

        data = {
            "question": [p["question"] for p in qa_pairs],
            "answer": [p["answer"] for p in qa_pairs],
            "contexts": [p["contexts"] for p in qa_pairs],
            "ground_truth": [p.get("ground_truth", "") for p in qa_pairs],
        }

        scores: dict = {}
        for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
            metric_name = getattr(metric, "name", str(metric))
            try:
                dataset = Dataset.from_dict(data)
                result = self._run(dataset, [metric])
                scores.update(result)
            except Exception as exc:
                logger.error("Metric %s failed: %s", metric_name, exc)
                scores[metric_name] = 0.0

        return {"scores": scores, "individual": qa_pairs}


def _is_nan(val) -> bool:
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False