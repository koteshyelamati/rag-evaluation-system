import os
import logging

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from app.config import settings

logger = logging.getLogger(__name__)

_SYSTEM = (
    "Use the following context to answer the question. "
    "If the answer is not in the context, say "
    "'I don't know based on the provided context.'"
)
_HUMAN = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


class RAGPipeline:
    def __init__(self) -> None:
        

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=settings.GEMINI_API_KEY,
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        self.vectorstore = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.TOP_K_RESULTS},
        )
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human", _HUMAN),
        ])

    def query(self, question: str) -> dict:
        try:
            docs = self.retriever.invoke(question)
            context = _format_docs(docs)
            messages = self._prompt.format_messages(context=context, question=question)
            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)
            chunks = [d.page_content for d in docs]
            return {
                "answer": answer,
                "source_documents": chunks,
                "context_chunks": chunks,
            }
        except Exception as exc:
            logger.error("RAG query failed: %s", exc)
            return {
                "answer": "I don't know based on the provided context.",
                "source_documents": [],
                "context_chunks": [],
            }
