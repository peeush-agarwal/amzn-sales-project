import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from logging_utils import setup_logging
from data_ingestion.rag_data_ingestion import RagDataIngestor
from rag_chains.qa_chain import QAChain


load_dotenv(override=True)


def _build_logger(config: Dict[str, Any]):
    os.makedirs(config["logs"]["path"], exist_ok=True)
    return setup_logging(
        log_level=config["logs"]["level"],
        logs_dir=config["logs"]["path"],
        log_filename=config["logs"]["rag_pipeline_file"],
        name=__name__,
    )


def main(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> int:
    """Run the RAG pipeline.

    Args:
        config: configuration dictionary (loaded from params.yaml)
        logger: optional logger instance. If not provided, one will be created.

    Returns:
        int: 0 on success, 1 on failure
    """
    logger = logger or _build_logger(config)

    try:
        logger.info("=" * 80)
        logger.info("Starting RAG Data Ingestion Pipeline")
        logger.info("=" * 80)

        data_path = config["rag"]["data_path"]
        max_rows = config["rag"].get("max_rows", 500)

        rag_data_ingestor = RagDataIngestor(
            collection_name=config["rag"]["collection_name"],
            persist_directory=Path(config["rag"]["persist_directory"]),
            embedding_model_name=config["rag"]["embedding_model_name"],
        )

        if rag_data_ingestor.require_ingestion(
            force=config["rag"]["force_ingestion"] == "true"
        ):
            logger.info("Ingestion required. Proceeding with data ingestion...")
            rag_data_ingestor.ingest(data_path=Path(data_path), max_rows=max_rows)
        else:
            logger.info("Ingestion not required. Skipping data ingestion.")

        logger.info("RAG Data Ingestion Pipeline completed successfully.")
        logger.info("=" * 80)

        logger.info("Build a vector store and RAG QA Chain next.")
        vector_store = rag_data_ingestor.get_vector_store()
        qa_chain = QAChain(
            model_name=config["rag"]["llm_model_name"], vector_store=vector_store
        )

        sample_question = (
            "Provide details about 'MI Usb Type-C Cable Smartphone (Black)'?"
        )
        response = qa_chain.answer_question(sample_question)
        logger.info(f"Sample Question: {sample_question}")
        logger.info(f"Sample Response: {response['result']}")
        source_docs = {}
        for i, doc in enumerate(response["source_documents"], start=1):
            product_id = doc.metadata.get("product_id", "unknown")
            product_name = doc.metadata.get("product_name", "unknown")
            category = doc.metadata.get("category", "unknown")
            price = doc.metadata.get("price", "unknown")
            discount = doc.metadata.get("discount", "unknown")
            rating = doc.metadata.get("rating", "unknown")
            source_docs[f"doc_{i}"] = {
                "product_id": product_id,
                "product_name": product_name,
                "category": category,
                "price": price,
                "discount": discount,
                "rating": rating,
            }
        result = {
            "query": sample_question,
            "result": response["result"],
            "source_documents": source_docs,
        }
        logger.info(f"Final Result: {result}")

        return 0

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.error("=" * 80)
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    # load config only when run as script
    with open("../config/params.yaml", "r") as f:
        import yaml

        config = yaml.safe_load(f)

    exit_code = main(config)
    sys.exit(exit_code)
