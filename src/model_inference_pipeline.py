import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from logging_utils import setup_logging


def _build_logger(config: Dict[str, Any]):
    os.makedirs(config["logs"]["path"], exist_ok=True)
    return setup_logging(
        log_level=config["logs"]["level"],
        logs_dir=config["logs"]["path"],
        log_filename=config["logs"]["model_inference_file"],
        name=__name__,
    )


def main(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> int:
    """Run the model inference pipeline."""
    logger = logger or _build_logger(config)

    try:
        logger.info("=" * 80)
        logger.info("Starting Model Inference Pipeline")
        logger.info("=" * 80)

        data_preprocessor_path = config.get("artifacts", {}).get("preprocessor_path")
        if not data_preprocessor_path:
            logger.error("artifacts.preprocessor_path not set in params.yaml")
            return 1

        logger.info("Loading data preprocessor from %s", data_preprocessor_path)
        data_preprocessor = joblib.load(Path(data_preprocessor_path))

        model_path = config.get("training", {}).get("output_model_path")
        if not model_path:
            logger.error("training.output_model_path not set in params.yaml")
            return 1

        logger.info("Loading model from %s", model_path)
        model = joblib.load(Path(model_path))

        logger.info("Loading test features and target...")
        test_features = pd.read_csv(config["data"]["path_test_features"])
        test_target = pd.read_csv(config["data"]["path_test_target"])

        df_test = pd.concat([test_features, test_target], axis=1)
        X_test, y_test = data_preprocessor.transform(df_test)
        logger.info(f"OK Processed test features shape: {X_test.shape}")

        preds = model.predict(X_test)

        out_df = pd.DataFrame({"prediction": preds})
        out_df[config["data"].get("target", "target")] = y_test

        out_path = Path("../artifacts/predictions.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

        logger.info("Saved predictions to %s", out_path)
        logger.info("=" * 80)
        logger.info("Model Inference Pipeline Completed Successfully!")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e))
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    # load config only when run as script
    with open("../config/params.yaml", "r") as f:
        import yaml

        config = yaml.safe_load(f)

    exit_code = main(config)
    sys.exit(exit_code)
