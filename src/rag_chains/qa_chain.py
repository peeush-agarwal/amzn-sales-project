import logging
from typing import Any, Dict

from langchain_classic.chains import RetrievalQA


class QAChain:
    def __init__(self, model_name: str, vector_store: Any):
        model = self._build_llm(model_name)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("QAChain initialized with model: %s", model_name)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using the RetrievalQA chain.

        Returns the raw chain response which should include 'result' and
        'source_documents'."""
        self.logger.info("Invoking chain for question: %s", question)
        response = self.qa_chain.invoke({"query": question})
        self.logger.info("Received response: %s", response.get("result"))
        return response

    def _build_llm(self, model_name: str):
        # local import so tests can monkeypatch langchain_groq before import
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name)
