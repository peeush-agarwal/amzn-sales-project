import sys
import types


def _inject_fake_qa_modules(captured):
    # fake langchain_groq.ChatGroq
    groq_mod = types.ModuleType("langchain_groq")

    class ChatGroq:
        def __init__(self, model: str):
            captured["model"] = model

    groq_mod.ChatGroq = ChatGroq  # type: ignore
    sys.modules["langchain_groq"] = groq_mod

    # fake langchain_classic.chains.RetrievalQA
    chains_mod = types.ModuleType("langchain_classic.chains")

    class RetrievalQA:
        @classmethod
        def from_chain_type(cls, llm, retriever, return_source_documents=False):
            class Chain:
                def __init__(self, llm, retriever):
                    self.llm = llm
                    self.retriever = retriever

                def invoke(self, payload):
                    return {"result": "ok", "source_documents": []}

            return Chain(llm, retriever)

    chains_mod.RetrievalQA = RetrievalQA  # type: ignore
    sys.modules["langchain_classic.chains"] = chains_mod


def _test_qa_chain_build_and_answer():
    captured = {}
    _inject_fake_qa_modules(captured)

    # create a simple fake vector store with as_retriever
    class FakeVS:
        def as_retriever(self):
            return lambda q: []

    from rag_chains.qa_chain import QAChain

    qc = QAChain(model_name="my-model", vector_store=FakeVS())
    resp = qc.answer_question("hello")
    assert resp["result"] == "ok"
    assert captured["model"] == "my-model"
