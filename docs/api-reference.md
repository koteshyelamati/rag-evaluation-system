# API Reference

Base URL: `http://localhost:8000`

---

## GET /api/health

Returns system status and current document count.

**Response:**
```json
{
  "status": "ok",
    "document_count": 42
    }
    ```

    ---

    ## POST /api/query

    Query the RAG system and receive an answer with evaluation scores.

    **Request:**
    ```json
    {
      "question": "What is retrieval-augmented generation?"
      }
      ```

      **Response:**
      ```json
      {
        "answer": "RAG is a technique that...",
          "source_documents": ["chunk 1...", "chunk 2..."],
            "evaluation": {
                "faithfulness": 0.91,
                    "answer_relevancy": 0.87
                      }
                      }
                      ```

                      ---

                      ## POST /api/ingest

                      Ingest one or more documents into the vector store.

                      **Request:**
                      ```json
                      {
                        "file_paths": ["./data/my_document.pdf"]
                        }
                        ```

                        **Response:**
                        ```json
                        {
                          "status": "success",
                            "chunks_added": 15
                            }
                            ```

                            ---

                            ## POST /api/evaluate

                            Run full Ragas batch evaluation against the built-in test dataset.

                            **Request:** No body required.

                            **Response:**
                            ```json
                            {
                              "faithfulness": 0.88,
                                "answer_relevancy": 0.85,
                                  "context_precision": 0.79,
                                    "context_recall": 0.82
                                    }
                                    ```

                                    ---

                                    ## Error Responses

                                    | Status Code | Meaning |
                                    |-------------|----------|
                                    | 400 | Bad request — invalid input |
                                    | 422 | Validation error — missing required fields |
                                    | 500 | Internal server error |
                                    
