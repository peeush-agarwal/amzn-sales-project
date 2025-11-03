import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from api.models import RAGContext
from data_ingestion.rag_data_ingestion import RagDataIngestor
from rag_chains.qa_chain import QAChain


class ApiRagService:
    """RAG Service for API."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: Path,
        embedding_model_name: str,
        llm_model_name: str,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.qa_chain = None

        self.logger = logging.getLogger(self.__class__.__name__)

    def load(self) -> None:
        """Load the RAG service components."""
        self.logger.info("Loading RAG service components...")
        data_ingestor = RagDataIngestor(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_model_name=self.embedding_model_name,
        )
        self.qa_chain = QAChain(
            model_name=self.llm_model_name,
            vector_store=data_ingestor.get_vector_store(),
        )
        self.logger.info("RAG service loaded successfully.")

    def is_ready(self) -> bool:
        """Check if RAG service is ready."""
        if "GROQ_API_KEY" not in os.environ:
            return False
        return self.qa_chain is not None

    def answer_question(
        self, question: str, top_k: Optional[int] = 3
    ) -> Dict[str, Any]:
        """Answer a question using the RAG QA chain."""
        if not self.is_ready():
            self.logger.warning("RAG service is not ready.")
            raise RuntimeError("RAG service is not ready.")

        if not self.qa_chain:
            self.logger.error("QA chain is not initialized.")
            raise RuntimeError("QA chain is not initialized.")

        self.logger.info(f"Answering question: {question}")
        response = self.qa_chain.answer_question(question)
        result = {
            "answer": response.get("result"),
            "contexts": [],
        }
        for i, doc in enumerate(response.get("source_documents", [])[:top_k], start=1):
            result["contexts"].append(
                RAGContext(
                    sequence_id=i,
                    product_id=doc.metadata.get("product_id", "N/A"),
                    category=doc.metadata.get("category", "N/A"),
                    rating=doc.metadata.get("rating", 0.0),
                    content=doc.page_content,
                    metadata=doc.metadata,
                )
            )
        return result
