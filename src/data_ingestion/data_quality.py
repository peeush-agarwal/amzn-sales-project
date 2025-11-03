import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np


class DataQualityMetrics:
    """Generate data quality metrics for monitoring and validation."""

    def __init__(self):
        self.logger = logging.getLogger(__class__.__name__)

    def calculate_metrics(
        self,
        df_raw: pd.DataFrame,
        df_selected: pd.DataFrame,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive data quality metrics.

        Args:
            df_raw: Raw input dataframe
            df_selected: Selected columns dataframe
            df_train: Training split dataframe
            df_test: Test split dataframe
            X_train: Processed training features
            y_train: Training target
            X_test: Processed test features
            y_test: Test target

        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            "raw_data": self._calculate_raw_metrics(df_raw),
            "selected_data": self._calculate_selected_metrics(df_selected),
            "train_test_split": self._calculate_split_metrics(df_train, df_test),
            "processed_data": self._calculate_processed_metrics(
                X_train, y_train, X_test, y_test
            ),
            "data_quality_checks": self._perform_quality_checks(df_raw, df_selected),
        }

        self.logger.info("Data quality metrics calculated successfully")
        return metrics

    def _calculate_raw_metrics(self, df_raw: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for raw data."""
        return {
            "total_rows": int(df_raw.shape[0]),
            "total_columns": int(df_raw.shape[1]),
            "memory_usage_mb": float(df_raw.memory_usage(deep=True).sum() / 1024**2),
            "duplicate_rows": int(df_raw.duplicated().sum()),
            "columns": df_raw.columns.tolist(),
        }

    def _calculate_selected_metrics(self, df_selected: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for selected data."""
        missing_stats = df_selected.isnull().sum()
        return {
            "total_rows": int(df_selected.shape[0]),
            "total_columns": int(df_selected.shape[1]),
            "missing_values": {
                col: int(count) for col, count in missing_stats.items() if count > 0
            },
            "missing_percentage": {
                col: float(count / len(df_selected) * 100)
                for col, count in missing_stats.items()
                if count > 0
            },
        }

    def _calculate_split_metrics(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate metrics for train-test split."""
        return {
            "train_size": int(df_train.shape[0]),
            "test_size": int(df_test.shape[0]),
            "train_percentage": float(
                df_train.shape[0] / (df_train.shape[0] + df_test.shape[0]) * 100
            ),
            "test_percentage": float(
                df_test.shape[0] / (df_train.shape[0] + df_test.shape[0]) * 100
            ),
        }

    def _calculate_processed_metrics(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for processed data."""

        metrics = {
            "train_features_shape": list(X_train.shape),
            "train_target_shape": list(y_train.shape),
            "target_statistics": {
                "train": {
                    "mean": float(np.mean(y_train)),
                    "std": float(np.std(y_train)),
                    "min": float(np.min(y_train)),
                    "max": float(np.max(y_train)),
                    "median": float(np.median(y_train)),
                },
            },
        }

        if X_test is not None:
            metrics["test_features_shape"] = list(X_test.shape)
        if y_test is not None:
            metrics["test_target_shape"] = list(y_test.shape)
            metrics["target_statistics"]["test"] = {
                "mean": float(np.mean(y_test)),
                "std": float(np.std(y_test)),
                "min": float(np.min(y_test)),
                "max": float(np.max(y_test)),
                "median": float(np.median(y_test)),
            }
        return metrics

    def _perform_quality_checks(
        self, df_raw: pd.DataFrame, df_selected: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform data quality checks."""
        checks = {
            "no_empty_dataframe": df_raw.shape[0] > 0,
            "no_all_null_columns": not df_selected.isnull().all().any(),
            "data_loss_percentage": float(
                (1 - df_selected.shape[0] / df_raw.shape[0]) * 100
                if df_raw.shape[0] > 0
                else 0
            ),
        }

        # Add warning if data loss is significant
        if checks["data_loss_percentage"] > 10:
            self.logger.warning(
                f"Significant data loss detected: {checks['data_loss_percentage']:.2f}%"
            )

        return checks

    def save_metrics(self, metrics: Dict[str, Any], output_path: Path) -> None:
        """Save metrics to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {output_path}")
