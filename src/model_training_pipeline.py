import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from logging_utils import setup_logging
from model_trainers.trainer import ModelTuner


def _build_logger(config: Dict[str, Any]):
    os.makedirs(config["logs"]["path"], exist_ok=True)
    return setup_logging(
        log_level=config["logs"]["level"],
        logs_dir=config["logs"]["path"],
        log_filename=config["logs"]["model_training_file"],
        name=__name__,
    )


def main(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> int:
    """Run the model training pipeline."""
    logger = logger or _build_logger(config)

    try:
        logger.info("=" * 80)
        logger.info("Starting Model Training Pipeline")
        logger.info("=" * 80)

        # Step: Load processed data
        logger.info("Step 1: Loading processed data...")
        df_train = pd.read_csv(config["data"]["path_train_features"])
        y_train = pd.read_csv(config["data"]["path_train_target"])
        logger.info(
            f"OK Loaded training data with {df_train.shape[0]} rows and {df_train.shape[1]} columns"
        )
        logger.info(
            f"OK Loaded training target with {y_train.shape[0]} rows and {y_train.shape[1]} columns"
        )

        logger.info("Step 2: Training models...")
        training_cfg = config.get("training", {})
        models_cfg = training_cfg.get("models", {})
        candidate_param_grids = {
            model_name: model_cfg.get("param_grid", {})
            for model_name, model_cfg in models_cfg.items()
        }

        tuner = ModelTuner(
            scoring=training_cfg.get("scoring", "neg_root_mean_squared_error"),
            cv=training_cfg.get("cv", None),
            test_size=training_cfg.get("val_size", 0.2),
            random_state=training_cfg.get("random_state", 42),
            n_jobs=training_cfg.get("n_jobs", -1),
            enable_progress=training_cfg.get("enable_progress", True),
            register_model=training_cfg.get("register_model", False),
        )

        # run tuning
        out = tuner.fit(
            df_train.values,
            y_train.values.ravel(),
            candidate_param_grids=candidate_param_grids,
            output_model_path=training_cfg.get("output_model_path", None),
            mlflow_experiment=training_cfg.get(
                "mlflow_experiment", "amzn-sales-training"
            ),
            model_registry_name=training_cfg.get("model_registry_name", None),
        )

        logger.info("Step 3: Storing experiment results...")
        results = out.get("results", {})
        rows: List[Dict] = []
        for model_name, res in results.items():
            row = {
                "model": model_name,
                "val_rmse": res.get("val_rmse"),
                "val_mae": res.get("val_mae"),
                "val_r2": res.get("val_r2_score"),
                "cv_best_score": res.get("cv_best_score"),
                "best_params": json.dumps(res.get("best_params")),
                "run_id": res.get("run_id"),
            }
            rows.append(row)

        experiments_csv = training_cfg.get(
            "experiments_csv", "../artifacts/experiments.csv"
        )
        exp_path = Path(experiments_csv)
        exp_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(exp_path, index=False)
        logger.info("Saved experiments summary to %s", exp_path)

        logger.info("=" * 80)
        logger.info("Model Training Pipeline Completed Successfully!")
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
