import logging
from typing import Any, Dict

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


class QAChain:
    def __init__(self, model_name: str, vector_store: Any):
        model = self._build_llm(model_name)

        # Custom prompt ensures LLM does not hallucinate
        prompt = PromptTemplate(
            template=(
                "You are a helpful assistant for answering questions using ONLY the provided context.\n"
                'If the answer is not contained in the context, respond strictly with: "I don\'t know".\n\n'
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question"],
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("QAChain initialized with model: %s", model_name)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using the RetrievalQA chain.

        Returns a dict with keys:
        - 'result': Final answer with IDK logic applied
        - 'source_documents': Retrieved docs
        """
        self.logger.info("Invoking chain for question: %s", question)
        response = self.qa_chain.invoke({"query": question})

        source_docs = response.get("source_documents", [])
        raw_answer = response.get("result", "").strip()

        # Case 1: No documents retrieved
        if not source_docs:
            self.logger.info("No source documents found. Responding with IDK.")
            response["result"] = "I don't know"
            return response

        # Case 2: LLM was unsure or prompt forced IDK
        lowered = raw_answer.lower()
        if lowered in {"i don't know", "i dont know", "don't know", "idk"}:
            self.logger.info("LLM returned uncertainty response.")
            response["result"] = "I don't know"

        self.logger.info("Final answer: %s", response["result"])
        return response

    def _build_llm(self, model_name: str):
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name)
