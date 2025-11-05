import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional

import joblib
import numpy as np
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.sklearn import log_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from model_trainers.model_factory import ModelFactory


logger = logging.getLogger(__name__)


class ModelTuner:
    """Generic model tuner that accepts features/target arrays via fit().

    The tuner performs comprehensive validation, multi-metric evaluation,
    and proper MLflow experiment tracking with model registration.

    Args:
        scoring: Primary metric for GridSearchCV optimization
        cv: Number of cross-validation folds (None for simple train/val split)
        test_size: Proportion of data for validation (if cv is None)
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs for GridSearchCV (-1 for all cores)
        enable_progress: Enable progress tracking (requires tqdm)
        register_model: Register best model to MLflow Model Registry
    """

    def __init__(
        self,
        scoring: str = "neg_root_mean_squared_error",
        cv: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        n_jobs: int = -1,
        enable_progress: bool = True,
        register_model: bool = False,
    ):
        self.scoring = scoring
        self.cv = cv
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.enable_progress = enable_progress
        self.register_model = register_model
        self.logger = logging.getLogger(__class__.__name__)

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate input arrays for shape, type, and data quality.

        Args:
            X: Feature array
            y: Target array

        Raises:
            ValueError: If validation fails
        """
        # Type validation
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X must be numpy array, got {type(X)}")
        if not isinstance(y, np.ndarray):
            raise ValueError(f"y must be numpy array, got {type(y)}")

        # Shape validation
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"X: {X.shape[0]}, y: {y.shape[0]}"
            )
        if X.shape[0] == 0:
            raise ValueError("X and y cannot be empty")

        # Data quality validation
        if np.any(np.isnan(X)):
            nan_count = np.sum(np.isnan(X))
            raise ValueError(f"X contains {nan_count} NaN values")
        if np.any(np.isinf(X)):
            inf_count = np.sum(np.isinf(X))
            raise ValueError(f"X contains {inf_count} infinite values")
        if np.any(np.isnan(y)):
            nan_count = np.sum(np.isnan(y))
            raise ValueError(f"y contains {nan_count} NaN values")
        if np.any(np.isinf(y)):
            inf_count = np.sum(np.isinf(y))
            raise ValueError(f"y contains {inf_count} infinite values")

        self.logger.info(
            f"Input validation passed: X shape {X.shape}, y shape {y.shape}"
        )

    def _validate_param_grids(
        self, candidate_param_grids: Dict[str, Dict[str, Any]]
    ) -> None:
        """Validate parameter grids structure.

        Args:
            candidate_param_grids: Dictionary of model name to param grid

        Raises:
            ValueError: If validation fails
        """
        if not candidate_param_grids:
            raise ValueError("candidate_param_grids cannot be empty")

        for model_name, grid in candidate_param_grids.items():
            if not isinstance(model_name, str):
                raise ValueError(f"Model name must be string, got {type(model_name)}")
            if not isinstance(grid, dict):
                raise ValueError(
                    f"Parameter grid for {model_name} must be dict, got {type(grid)}"
                )
            # Validate model name is supported
            try:
                ModelFactory.create(model_name)
            except ValueError as e:
                raise ValueError(f"Invalid model name '{model_name}': {e}")

        self.logger.info(
            f"Parameter grid validation passed for {len(candidate_param_grids)} models"
        )

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:
        """Compute multiple regression metrics.

        Args:
            y_true: True target values
            y_pred: Predicted target values
            prefix: Prefix for metric names (e.g., 'train_', 'val_')

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        # RMSE
        rmse = float(root_mean_squared_error(y_true, y_pred))
        metrics[f"{prefix}rmse"] = rmse

        # MAE
        mae = float(mean_absolute_error(y_true, y_pred))
        metrics[f"{prefix}mae"] = mae

        # R² Score
        r2 = float(r2_score(y_true, y_pred))
        metrics[f"{prefix}r2_score"] = r2

        # MAPE (handle zero values in y_true)
        if np.any(y_true == 0):
            self.logger.warning(
                f"{prefix}MAPE not computed: y_true contains zero values"
            )
        else:
            mape = float(mean_absolute_percentage_error(y_true, y_pred))
            metrics[f"{prefix}mape"] = mape

        return metrics

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        candidate_param_grids: Dict[str, Dict[str, Any]],
        output_model_path: Optional[Path] = None,
        mlflow_experiment: str = "milestone_1_tuning",
        model_registry_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run grid search across candidate models with comprehensive tracking.

        Args:
            X: feature array (n_samples, n_features)
            y: target array (n_samples,)
            candidate_param_grids: mapping model_name -> param_grid for GridSearchCV
            output_model_path: path to save best model artifact
            mlflow_experiment: mlflow experiment name
            model_registry_name: name for model registry (if register_model=True)

        Returns:
            dict with best model info, metrics, and per-candidate results

        Raises:
            ValueError: If validation fails
            RuntimeError: If no model can be trained successfully
        """
        # Validation
        self._validate_inputs(X, y)
        self._validate_param_grids(candidate_param_grids)

        # Log experiment parameters
        self.logger.info(
            f"Starting model tuning with {len(candidate_param_grids)} candidates"
        )
        self.logger.info(f"Data: X={X.shape}, y={y.shape}")
        self.logger.info(f"Scoring: {self.scoring}, CV: {self.cv}")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.logger.info(f"Split: train={X_train.shape[0]}, val={X_val.shape[0]}")

        # Setup MLflow
        mlflow.set_experiment(mlflow_experiment)

        # Track best model
        best_rmse = float("inf")
        best_name = None
        best_model = None
        best_run_id = None
        best_metrics = {}
        results: Dict[str, Any] = {}

        # Setup progress tracking
        if self.enable_progress:
            try:
                from tqdm import tqdm

                model_iterator = tqdm(
                    candidate_param_grids.items(), desc="Training models", unit="model"
                )
            except ImportError:
                self.logger.warning("tqdm not available, progress bar disabled")
                model_iterator = candidate_param_grids.items()
        else:
            model_iterator = candidate_param_grids.items()

        # Train each candidate model
        for model_name, param_grid in model_iterator:
            start_time = time.time()

            with mlflow.start_run(run_name=f"eval_{model_name}") as run:
                try:
                    # Create estimator
                    estimator = ModelFactory.create(model_name)
                    self.logger.info(
                        f"Training {model_name}: {type(estimator).__name__}"
                    )

                    # Log parameters
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("estimator_class", type(estimator).__name__)
                    mlflow.log_param("n_train_samples", X_train.shape[0])
                    mlflow.log_param("n_val_samples", X_val.shape[0])
                    mlflow.log_param("n_features", X_train.shape[1])
                    mlflow.log_param("random_state", self.random_state)

                    # Grid search with parallel execution
                    gs = GridSearchCV(
                        estimator,
                        param_grid,
                        scoring=self.scoring,
                        cv=self.cv,
                        n_jobs=self.n_jobs,
                        verbose=1 if self.logger.level <= logging.DEBUG else 0,
                    )
                    gs.fit(X_train, y_train)

                    # Get best model
                    best_estimator = gs.best_estimator_

                    # Predictions
                    train_preds = best_estimator.predict(X_train)
                    val_preds = best_estimator.predict(X_val)

                    # Compute comprehensive metrics
                    train_metrics = self._compute_metrics(
                        y_train, train_preds, prefix="train_"
                    )
                    val_metrics = self._compute_metrics(y_val, val_preds, prefix="val_")

                    # Log all metrics to MLflow
                    for metric_name, metric_value in {
                        **train_metrics,
                        **val_metrics,
                    }.items():
                        mlflow.log_metric(metric_name, metric_value)

                    # Log best parameters from grid search
                    for param_name, param_value in gs.best_params_.items():
                        mlflow.log_param(f"best_{param_name}", param_value)
                    mlflow.log_param("cv_best_score", gs.best_score_)

                    # Log training time
                    training_time = time.time() - start_time
                    mlflow.log_metric("training_time_seconds", training_time)

                    # Infer and log model signature
                    signature = infer_signature(X_train, train_preds)

                    # # Log model to MLflow with signature
                    # log_model(
                    #     best_estimator,
                    #     name="model",
                    #     signature=signature,
                    #     registered_model_name=None,  # Register separately if best
                    # )

                    # Store results
                    val_rmse = val_metrics["val_rmse"]
                    results[model_name] = {
                        "val_rmse": val_rmse,
                        "val_mae": val_metrics.get("val_mae"),
                        "val_r2_score": val_metrics.get("val_r2_score"),
                        "val_mape": val_metrics.get("val_mape"),
                        "train_rmse": train_metrics["train_rmse"],
                        "train_mae": train_metrics.get("train_mae"),
                        "train_r2_score": train_metrics.get("train_r2_score"),
                        "best_params": gs.best_params_,
                        "cv_best_score": gs.best_score_,
                        "training_time": training_time,
                        "run_id": run.info.run_id,
                    }

                    # Track best model
                    if val_rmse < best_rmse:
                        best_rmse = val_rmse
                        best_name = model_name
                        best_model = best_estimator
                        best_run_id = run.info.run_id
                        best_metrics = {**train_metrics, **val_metrics}

                    self.logger.info(
                        f"{model_name} completed: val_rmse={val_rmse:.4f}, "
                        f"val_r2={val_metrics.get('val_r2_score', 0):.4f}, "
                        f"time={training_time:.2f}s"
                    )

                except Exception as e:
                    self.logger.error(
                        f"Failed to train {model_name}: {str(e)}", exc_info=True
                    )
                    results[model_name] = {"error": str(e)}
                    continue

        # Verify we have a best model
        if best_model is None:
            raise RuntimeError(
                "No model was successfully trained. Check logs for errors."
            )

        self.logger.info(f"Best model: {best_name} with val_rmse={best_rmse:.4f}")

        # Register best model to Model Registry
        if self.register_model and model_registry_name:
            try:
                model_uri = f"runs:/{best_run_id}/model"
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_registry_name,
                )
                self.logger.info(
                    f"Registered model '{model_registry_name}' "
                    f"version {registered_model.version}"
                )
            except Exception as e:
                self.logger.error(f"Failed to register model: {e}", exc_info=True)

        # Save local artifact if requested
        if output_model_path:
            try:
                os.makedirs(Path(output_model_path).parent, exist_ok=True)
                joblib.dump(best_model, output_model_path)
                self.logger.info(f"Saved model to {output_model_path}")

                # Save metadata
                meta_path = Path(output_model_path).with_suffix(".meta.json")
                meta = {
                    "best_name": best_name,
                    "best_run_id": best_run_id,
                    "model_path": str(output_model_path),
                    "metrics": best_metrics,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                self.logger.info(f"Saved metadata to {meta_path}")
            except Exception as e:
                self.logger.error(f"Failed to save model locally: {e}", exc_info=True)

        return {
            "best_name": best_name,
            "best_rmse": best_rmse,
            "best_run_id": best_run_id,
            "model_path": str(output_model_path) if output_model_path else None,
            "metrics": best_metrics,
            "results": results,
        }

    @staticmethod
    def load_model_from_mlflow(run_id: str, artifact_path: str = "model"):
        """Load a model saved in MLflow for a given run id.

        Returns a model object loaded via mlflow.sklearn.load_model with the runs:/ URI.
        """
        from mlflow.sklearn import load_model

        uri = f"runs:/{run_id}/{artifact_path}"
        return load_model(uri)
