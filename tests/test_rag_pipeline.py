from unittest.mock import MagicMock, patch


def _make_doc(text: str):
    doc = MagicMock()
    doc.page_content = text
    return doc


class TestRAGPipeline:
    @patch("app.rag_pipeline.Chroma")
    @patch("app.rag_pipeline.GoogleGenerativeAIEmbeddings")
    @patch("app.rag_pipeline.ChatGoogleGenerativeAI")
    def test_query_returns_required_keys(self, _llm, _emb, mock_chroma):
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance
        mock_retriever = MagicMock()
        mock_chroma_instance.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [_make_doc("ML context chunk")]

        mock_llm_inst = MagicMock()
        _llm.return_value = mock_llm_inst
        mock_response = MagicMock()
        mock_response.content = "Machine learning is a subset of AI."
        mock_llm_inst.invoke.return_value = mock_response

        from app.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        result = pipeline.query("What is machine learning?")

        assert "answer" in result
        assert "source_documents" in result
        assert "context_chunks" in result

    @patch("app.rag_pipeline.Chroma")
    @patch("app.rag_pipeline.GoogleGenerativeAIEmbeddings")
    @patch("app.rag_pipeline.ChatGoogleGenerativeAI")
    def test_query_answer_is_string(self, _llm, _emb, mock_chroma):
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance
        mock_retriever = MagicMock()
        mock_chroma_instance.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [_make_doc("chunk 1"), _make_doc("chunk 2")]

        mock_llm_inst = MagicMock()
        _llm.return_value = mock_llm_inst
        mock_response = MagicMock()
        mock_response.content = "Some answer"
        mock_llm_inst.invoke.return_value = mock_response

        from app.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        result = pipeline.query("Some question?")

        assert isinstance(result["answer"], str)

    @patch("app.rag_pipeline.Chroma")
    @patch("app.rag_pipeline.GoogleGenerativeAIEmbeddings")
    @patch("app.rag_pipeline.ChatGoogleGenerativeAI")
    def test_query_handles_empty_retrieval(self, _llm, _emb, mock_chroma):
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance
        mock_retriever = MagicMock()
        mock_chroma_instance.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = []

        mock_llm_inst = MagicMock()
        _llm.return_value = mock_llm_inst
        mock_response = MagicMock()
        mock_response.content = "I don't know based on the provided context."
        mock_llm_inst.invoke.return_value = mock_response

        from app.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        result = pipeline.query("An obscure question with no matching docs")

        assert result["source_documents"] == []
        assert result["context_chunks"] == []
        assert isinstance(result["answer"], str)

    @patch("app.rag_pipeline.Chroma")
    @patch("app.rag_pipeline.GoogleGenerativeAIEmbeddings")
    @patch("app.rag_pipeline.ChatGoogleGenerativeAI")
    def test_query_returns_fallback_on_exception(self, _llm, _emb, mock_chroma):
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance
        mock_retriever = MagicMock()
        mock_chroma_instance.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.side_effect = RuntimeError("API error")

        from app.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        result = pipeline.query("question that causes error")

        assert "answer" in result
        assert result["source_documents"] == []
        assert result["context_chunks"] == []
        assert "don't know" in result["answer"]

    @patch("app.rag_pipeline.Chroma")
    @patch("app.rag_pipeline.GoogleGenerativeAIEmbeddings")
    @patch("app.rag_pipeline.ChatGoogleGenerativeAI")
    def test_context_chunks_match_source_documents(self, _llm, _emb, mock_chroma):
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance
        mock_retriever = MagicMock()
        mock_chroma_instance.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [_make_doc("chunk A"), _make_doc("chunk B")]

        mock_llm_inst = MagicMock()
        _llm.return_value = mock_llm_inst
        mock_response = MagicMock()
        mock_response.content = "Answer here"
        mock_llm_inst.invoke.return_value = mock_response

        from app.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        result = pipeline.query("test question")

        assert result["source_documents"] == result["context_chunks"]
        assert result["context_chunks"] == ["chunk A", "chunk B"]
