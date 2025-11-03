import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from data_ingestion.load_csv import DataLoaderCsv
from data_ingestion.validate_data import DataValidatorBase
from data_ingestion.preprocess_data import DataTransformerBase
from data_ingestion.data_quality import DataQualityMetrics
from logging_utils import setup_logging


def _build_logger(config: Dict[str, Any]):
    os.makedirs(config["logs"]["path"], exist_ok=True)
    return setup_logging(
        log_level=config["logs"]["level"],
        logs_dir=config["logs"]["path"],
        log_filename=config["logs"]["data_ingestion_file"],
        name=__name__,
    )


def main(config: Dict[str, Any], logger: Optional[logging.Logger] = None):
    """Run the data ingestion pipeline."""
    logger = logger or _build_logger(config)

    try:
        logger.info("=" * 80)
        logger.info("Starting Data Ingestion Pipeline")
        logger.info("=" * 80)

        # Step: Load data
        logger.info("Step 1: Loading raw data...")
        data_loader = DataLoaderCsv(Path(config["data"]["path"]))
        df_raw = data_loader.load()
        logger.info(f"OK Loaded {df_raw.shape[0]} rows and {df_raw.shape[1]} columns")

        columns = config["data"]["cols_raw"]
        target_col = config["data"]["target"]

        # Step: Validate data
        logger.info("Step 2: Validating data...")
        data_validator = DataValidatorBase(columns=columns, target_col=target_col)
        if not data_validator.validate(df_raw):
            logger.error("Data validation failed!")
            sys.exit(1)
        logger.info("OK Data validation passed")

        # Step: Select columns
        logger.info("Step 3: Selecting relevant columns...")
        path_selected = config["data"]["path_selected"]
        os.makedirs(os.path.dirname(path_selected), exist_ok=True)
        df_selected = df_raw[columns + [target_col]]
        df_selected.to_csv(path_selected, index=False)
        logger.info(f"OK Selected data saved to {path_selected}")

        # Step: Split data
        logger.info("Step 4: Splitting data into train and test sets...")
        df_train, df_test = train_test_split(
            df_selected,
            test_size=config["data"]["test_size"],
            random_state=config["data"]["random_state"],
        )
        logger.info(
            f"OK Train size: {df_train.shape[0]}, Test size: {df_test.shape[0]}"
        )

        # Step: Preprocess data
        logger.info("Step 5: Preprocessing and feature engineering...")
        data_preprocessor = DataTransformerBase(
            columns=config["data"]["cols_transformed"], target_col=target_col
        )
        X_train, y_train = data_preprocessor.fit_transform(df_train)
        logger.info(f"OK Processed training features shape: {X_train.shape}")

        # Step: Save processed data
        logger.info("Step 6: Saving processed data...")
        pd.DataFrame(X_train, columns=data_preprocessor.features).to_csv(
            config["data"]["path_train_features"], index=False
        )
        pd.Series(y_train, name=target_col).to_csv(
            config["data"]["path_train_target"], index=False
        )
        logger.info("OK All processed data files saved")

        logger.info("OK Saved raw test features and target")
        df_test.drop(columns=[target_col]).to_csv(
            config["data"]["path_test_features"], index=False
        )
        df_test[target_col].to_csv(config["data"]["path_test_target"], index=False)

        # Step: Save the preprocessor object
        logger.info("Step 7: Saving preprocessor artifact...")
        preprocessor_path = config["artifacts"]["preprocessor_path"]
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
        joblib.dump(data_preprocessor, preprocessor_path)
        logger.info(f"OK Preprocessor saved to {preprocessor_path}")

        # Step: Generate data quality metrics
        logger.info("Step 8: Generating data quality metrics...")
        metrics_calculator = DataQualityMetrics()
        metrics = metrics_calculator.calculate_metrics(
            df_raw=df_raw,
            df_selected=df_selected,
            df_train=df_train,
            df_test=df_test,
            X_train=X_train,
            y_train=y_train,
        )

        metrics_path = Path("../artifacts/data_quality_metrics.json")
        metrics_calculator.save_metrics(metrics, metrics_path)
        logger.info(f"OK Metrics saved to {metrics_path}")

        logger.info("=" * 80)
        logger.info("Data Ingestion Pipeline Completed Successfully!")
        logger.info("=" * 80)

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
