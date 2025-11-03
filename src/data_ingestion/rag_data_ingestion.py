import logging
from pathlib import Path
import pandas as pd

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class RagDataIngestor:
    def __init__(
        self, collection_name: str, persist_directory: Path, embedding_model_name: str
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.logger = logging.getLogger(self.__class__.__name__)

    def ingest(self, data_path: Path, max_rows: int = 500) -> None:
        """Ingest data from the specified path into the vector store."""
        self.logger.info(
            "Ingesting data from %s into collection '%s'",
            data_path,
            self.collection_name,
        )
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")

        df = pd.read_csv(data_path).head(max_rows)

        documents = []
        for _, row in df.iterrows():
            content = self._build_product_content(row)

            metadata = {
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "category": row["category"],
                "price": row["actual_price"],
                "discount": row["discount_percentage"],
                "rating": row["rating"],
            }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        # persist_directory parent will be created by Chroma if necessary, but
        # ensure parent exists to avoid accidental failures when path is invalid
        try:
            Chroma.from_documents(
                documents,
                self.embedding_model,
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory),
            )
        except Exception:
            self.logger.exception("Failed to persist documents to Chroma store")
            raise
        self.logger.info("Ingestion complete. %d documents added.", len(documents))

    def get_vector_store(self) -> Chroma:
        """Return the Chroma vector store instance."""
        vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory),
            embedding_function=self.embedding_model,
        )
        return vector_store

    def _build_product_content(self, row: pd.Series) -> str:
        return f"""Product name is {row["product_name"]} and its category is {row["category"]}. The actual price is ${row["actual_price"]} with a discount of {row["discount_percentage"]}%. It has a rating of {row["rating"]}/5.0 based on {row["rating_count"]} reviews. Following is the detailed description and customer reviews:
- About Product: {row["about_product"]}
- Customer Reviews: {row["review_content"]}\n\n"""

    def require_ingestion(self, force: bool = False) -> bool:
        """Check if the data ingestion is required."""
        if force:
            self.logger.info("ingestion forced")
            return True

        if self.persist_directory.exists():
            self.logger.info(
                "Persist directory %s exists; skipping ingestion",
                self.persist_directory,
            )
            return False

        self.logger.info(
            "Persist directory %s does not exist; ingestion required",
            self.persist_directory,
        )
        return True
