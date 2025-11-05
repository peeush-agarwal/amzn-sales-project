import logging
from typing import Any, Dict

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


class QAChain:
    def __init__(self, model_name: str, vector_store: Any, max_docs: int = 2):
        model = self._build_llm(model_name)

        # Limit retrieved documents to avoid token overflow
        retriever = vector_store.as_retriever(search_kwargs={"k": max_docs})

        # Shorter prompt to reduce tokens
        prompt = PromptTemplate(
            template=(
                "Answer the question using ONLY the context below.\n"
                'If the answer is not in the context, respond: "I don\'t know".\n\n'
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer:"
            ),
            input_variables=["context", "question"],
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("QAChain initialized with model: %s", model_name)

    def answer_question(self, question: str) -> Dict[str, Any]:
        self.logger.info("Invoking chain for question: %s", question)

        response = self.qa_chain.invoke({"query": question})
        source_docs = response.get("source_documents", [])
        raw_answer = response.get("result", "").strip()

        if not source_docs:
            self.logger.info("No source documents found. Responding with IDK.")
            response["result"] = "I don't know"
            return response

        lowered = raw_answer.lower()
        if lowered in {"i don't know", "i dont know", "don't know", "idk"}:
            self.logger.info("LLM returned uncertainty response.")
            response["result"] = "I don't know"

        self.logger.info("Final answer: %s", response["result"])
        return response

    def _build_llm(self, model_name: str):
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name)
