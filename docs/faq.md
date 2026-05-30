# Frequently Asked Questions

## General

**Q: What is the purpose of this project?**

A: Most RAG demos stop at "it answers questions." This project adds automated quality scoring to every query using Ragas metrics, making it suitable for production use where answer quality must be monitored continuously.

**Q: Does this cost money to run?**

A: Yes — it uses the OpenAI API for both embeddings and generation. Typical costs are minimal for development use (a few cents per query), but batch evaluation runs can accumulate costs. Monitor your usage at https://platform.openai.com/usage.

**Q: Can I use a different LLM?**

A: Yes. The LangChain integration makes it straightforward to swap out GPT-4o-mini for another model. Update the model name in `app/config.py`.

**Q: Can I use a different vector store?**

A: The system is built around ChromaDB for local persistence, but LangChain supports many vector stores (Pinecone, Weaviate, FAISS, etc.). Replacing ChromaDB requires changes to `app/document_loader.py` and `app/rag_pipeline.py`.

## Evaluation

**Q: Why does evaluation add latency to every query?**

A: Ragas uses the LLM as a judge for metrics like faithfulness and answer relevancy, so it makes additional API calls per query. If latency is critical, you can disable real-time evaluation and run batch evaluation offline.

**Q: What does a faithfulness score of 0 mean?**

A: It means none of the factual claims in the answer could be traced back to the retrieved context — the model may be hallucinating entirely. Check your retrieval pipeline and prompt template.

**Q: My context recall is low. What should I do?**

A: Try increasing `TOP_K_RESULTS` to retrieve more chunks, or decrease `CHUNK_SIZE` so more granular chunks are indexed. Also verify that your documents contain the information needed to answer your test questions.

## Setup

**Q: Does the system work without any documents ingested?**

A: Yes — sample AI/ML documents are auto-indexed on first launch so you can start querying immediately.

**Q: How do I reset the vector store?**

A: Delete the directory specified in `CHROMA_PERSIST_DIR` (default: `./chroma_db`) and restart the server. Documents will be re-indexed automatically.
