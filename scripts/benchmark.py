"""
Benchmark script for the RAG Evaluation System.
Runs a series of test queries and reports aggregate Ragas metrics.

Usage:
    python scripts/benchmark.py
    """

import httpx
import json
import statistics
from typing import List, Dict

BASE_URL = "http://localhost:8000"

TEST_QUERIES = [
      "What is retrieval-augmented generation?",
      "How does ChromaDB store embeddings?",
      "What are the benefits of using MMR retrieval?",
      "Explain the difference between faithfulness and answer relevancy.",
      "How does LangChain orchestrate the RAG pipeline?",
]


def run_benchmark(queries: List[str]) -> Dict[str, float]:
      """Run benchmark queries and return aggregate metrics."""
      all_scores = {
          "faithfulness": [],
          "answer_relevancy": [],
      }

    print(f"Running {len(queries)} benchmark queries...\n")

    for i, query in enumerate(queries, 1):
              print(f"[{i}/{len(queries)}] {query[:60]}...")
              response = httpx.post(
                  f"{BASE_URL}/api/query",
                  json={"question": query},
                  timeout=30.0,
              )
              data = response.json()
              evaluation = data.get("evaluation", {})

        for metric in all_scores:
                      score = evaluation.get(metric)
                      if score is not None:
                                        all_scores[metric].append(score)

              print("\nBenchmark Results:")
    print("-" * 40)
    for metric, scores in all_scores.items():
              if scores:
                            avg = statistics.mean(scores)
                            print(f"{metric:25s}: {avg:.3f}")

          return {k: statistics.mean(v) for k, v in all_scores.items() if v}


if __name__ == "__main__":
      run_benchmark(TEST_QUERIES)
  
