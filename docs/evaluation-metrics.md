# Evaluation Metrics Guide

This document explains the four Ragas metrics used in the RAG Evaluation System.

## 1. Faithfulness

**Definition:** Measures whether all claims in the generated answer are grounded in the retrieved context.

- Score range: 0.0 to 1.0
- High score (>0.7): The model is not hallucinating; every factual claim traces back to the context.
- Low score (<0.4): The model is generating information not supported by retrieved documents.

**How it's computed:** Each factual statement in the answer is checked against the retrieved context. The final score is the ratio of supported statements to total statements.

## 2. Answer Relevancy

**Definition:** Measures how directly the answer addresses the original question.

- Score range: 0.0 to 1.0
- High score: The answer is focused and on-topic.
- Low score: The answer drifts or provides unnecessary information.

## 3. Context Precision

**Definition:** Evaluates whether the most relevant document chunks are ranked at the top of the retrieved results.

- High score: Relevant chunks appear first — less noise at the top.
- Low score: Relevant content is buried behind irrelevant chunks.

## 4. Context Recall

**Definition:** Measures whether the retrieved context contains all the information needed for a complete answer.

- High score: All necessary information is present in the retrieved chunks.
- Low score: The retriever is missing important context, leading to incomplete answers.

## Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| > 0.7       | Production-ready |
| 0.4 – 0.7   | Acceptable, monitor closely |
| < 0.4       | Pipeline needs tuning |
