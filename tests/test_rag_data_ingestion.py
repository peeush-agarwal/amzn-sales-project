import sys
import types
import pandas as pd


def _inject_fake_langchain_modules(chroma_called):
    # fake langchain_core.documents.Document
    docs_mod = types.ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    docs_mod.Document = Document  # type: ignore
    sys.modules["langchain_core.documents"] = docs_mod

    # fake HuggingFaceEmbeddings
    hf_mod = types.ModuleType("langchain_huggingface")

    class HuggingFaceEmbeddings:
        def __init__(self, model_name: str):
            self.model_name = model_name

    hf_mod.HuggingFaceEmbeddings = HuggingFaceEmbeddings  # type: ignore
    sys.modules["langchain_huggingface"] = hf_mod

    # fake Chroma
    vc_mod = types.ModuleType("langchain_community.vectorstores")

    class Chroma:
        @classmethod
        def from_documents(
            cls, documents, embedding, collection_name, persist_directory
        ):
            # signal that this was called and capture arguments
            chroma_called.append(
                {
                    "documents": documents,
                    "embedding": embedding,
                    "collection_name": collection_name,
                    "persist_directory": persist_directory,
                }
            )

        def __init__(
            self, collection_name=None, persist_directory=None, embedding_function=None
        ):
            self.collection_name = collection_name
            self.persist_directory = persist_directory
            self.embedding_function = embedding_function

        def as_retriever(self):
            return lambda q: []

    vc_mod.Chroma = Chroma  # type: ignore
    sys.modules["langchain_community.vectorstores"] = vc_mod


def test_require_ingestion(tmp_path):
    # inject fake langchain modules so importing RagDataIngestor won't try to
    # initialize real HuggingFace models during tests
    chroma_called = []
    _inject_fake_langchain_modules(chroma_called)

    # ensure module will be imported fresh so it picks up our fake modules
    if "data_ingestion.rag_data_ingestion" in sys.modules:
        del sys.modules["data_ingestion.rag_data_ingestion"]
    # import after injecting fakes; create a directory
    from data_ingestion.rag_data_ingestion import RagDataIngestor

    persist = tmp_path / "chroma_db"
    r = RagDataIngestor("col", persist, "model")
    # should require ingestion since persist dir does not exist
    assert r.require_ingestion() is True
    # create the dir and test again
    persist.mkdir()
    assert r.require_ingestion() is False


def test_ingest_calls_chroma(tmp_path):
    chroma_called = []
    _inject_fake_langchain_modules(chroma_called)

    # ensure module will be imported fresh so it picks up our fake modules
    if "data_ingestion.rag_data_ingestion" in sys.modules:
        del sys.modules["data_ingestion.rag_data_ingestion"]
    # now import the class (module will use our fake modules)
    from data_ingestion.rag_data_ingestion import RagDataIngestor

    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        [
            {
                "product_id": 1,
                "product_name": "p1",
                "category": "c1",
                "actual_price": 9.99,
                "discount_percentage": 10,
                "rating": 4.5,
                "rating_count": 10,
                "about_product": "good",
                "review_content": "nice",
            }
        ]
    )
    df.to_csv(csv_path, index=False)

    persist = tmp_path / "chroma_db"
    rd = RagDataIngestor("col", persist, "model_name")
    rd.ingest(data_path=csv_path, max_rows=10)

    # ensure Chroma.from_documents was called and documents were created
    assert len(chroma_called) == 1
    entry = chroma_called[0]
    docs = entry["documents"]
    assert len(docs) == 1
    doc = docs[0]
    # metadata keys exist
    assert "product_id" in doc.metadata
    assert doc.metadata["product_name"] == "p1"
    assert entry["collection_name"] == "col"
