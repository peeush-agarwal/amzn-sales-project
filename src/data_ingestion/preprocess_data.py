import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataTransformerBase:
    def __init__(self, columns: List[str], target_col: str):
        """
        Args:
            columns: list of feature column names.
            target_col: name of the target column.
        """
        self.cols_list = columns
        self.target_col = target_col
        self.features = []
        self.label_encoders = {}
        self.scaler = None
        # frequency encoders stored as dict of dicts
        self.freq_encoders = {}
        self.logger = logging.getLogger(__class__.__name__)

    def fit_transform(self, df_: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit encoders/scaler on training data and transform.

        Args:
            df_: input DataFrame for training
        Returns:
            Tuple[np.ndarray, np.ndarray]: features and target arrays
        """
        if df_ is None or df_.empty:
            raise ValueError("Data cannot be None or empty.")

        df = df_.copy()

        y = df[self.target_col] if self.target_col in df.columns else None
        df = df.drop(columns=[self.target_col], errors="ignore")

        y = self._clean_target(y)
        if y is not None and y.isna().sum() > 0:
            self.logger.warning(
                "Target column '%s' contains NaN values after cleaning.",
                self.target_col,
            )
            indices_to_drop = y[y.isna()].index
            df = df.drop(index=indices_to_drop)
            y = y.drop(index=indices_to_drop)
            self.logger.info(
                "Dropped %d rows from the DataFrame due to NaN values in the target column.",
                len(indices_to_drop),
            )

        df = self._preprocess_common(df)

        # Fit label encoders
        for col in ["category", "product_name", "user_name"]:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Frequency encode some high-cardinality fields (e.g., product_id, user_id)
        for col in ["product_id", "user_id"]:
            if col in df.columns:
                freq = df[col].astype(str).value_counts(dropna=False).to_dict()
                self.freq_encoders[col] = freq
                df[f"{col}_freq"] = df[col].astype(str).map(freq).fillna(0)

        df = self._feature_engineering(df)
        num_cols = self._get_numeric_columns(df)
        self.scaler = StandardScaler()
        if len(num_cols) > 0:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])

        # keep only columns that are present in the DataFrame
        present = [c for c in self.cols_list if c in df.columns]
        self.features = present
        df = df[present]

        X_arr = np.asarray(df.values)
        y_arr = np.asarray(y.values) if y is not None else np.array([])
        return X_arr, y_arr

    def transform(self, df_infer: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform inference data using fitted encoders/scaler.

        Args:
            df_infer: input DataFrame for inference
        Returns:
            Tuple[np.ndarray, np.ndarray]: features and target arrays
        """
        if df_infer is None or df_infer.empty:
            raise ValueError("Data cannot be None or empty.")

        df = df_infer.copy()

        y = df[self.target_col] if self.target_col in df.columns else None
        df = df.drop(columns=[self.target_col], errors="ignore")

        y = self._clean_target(y)

        df = self._preprocess_common(df)

        # Transform label encoders
        for col, le in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen labels by mapping to -1
                def safe_transform(x):
                    try:
                        return le.transform([str(x)])[0]
                    except Exception:
                        return -1

                df[col] = df[col].apply(safe_transform)

        # Apply frequency encoders
        for col, freq in self.freq_encoders.items():
            if col in df.columns:
                df[f"{col}_freq"] = df[col].astype(str).map(freq).fillna(0)

        df = self._feature_engineering(df)

        num_cols = self._get_numeric_columns(df)

        if self.scaler is not None and len(num_cols) > 0:
            df[num_cols] = self.scaler.transform(df[num_cols])

        # keep only columns that are present in the DataFrame
        present = [c for c in self.cols_list if c in df.columns]
        self.features = present
        df = df[present]

        X_arr = np.asarray(df.values)
        y_arr = np.asarray(y.values) if y is not None else np.array([])
        return X_arr, y_arr

    def _clean_target(self, y: Optional[pd.Series]) -> Optional[pd.Series]:
        """
        Clean target column if needed.
        """
        if y is None:
            return None

        # Remove '%' and convert to float
        y_cleaned = pd.to_numeric(y.astype(str).str.replace("%", ""), errors="coerce")
        y_cleaned = y_cleaned / 100.0  # convert percentage to decimal
        return y_cleaned

    def _preprocess_common(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Common preprocessing for both train and inference.
        """

        # Clean price-like columns (remove currency symbols, commas) and coerce to float
        def _clean_numeric_column(col_series: pd.Series) -> pd.Series:
            # convert to string, strip common non-numeric chars, then to numeric
            s = col_series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
            s = s.replace({"": None})
            return pd.to_numeric(s, errors="coerce")

        # Handle missing values independently
        if "actual_price" in df.columns:
            df["actual_price"] = _clean_numeric_column(df["actual_price"])
            df["actual_price"] = df["actual_price"].fillna(df["actual_price"].median())
        if "rating" in df.columns:
            # coerce rating to numeric and compute median on numeric values
            numeric_rating = pd.to_numeric(df["rating"], errors="coerce")
            median_rating = numeric_rating.median()
            df["rating"] = numeric_rating.fillna(median_rating)
        if "rating_count" in df.columns:
            df["rating_count"] = pd.to_numeric(
                df["rating_count"], errors="coerce"
            ).fillna(0)
        for col in ["about_product", "review_title", "review_content"]:
            if col in df.columns:
                df[col] = df[col].fillna("")
        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Common feature engineering for both train and inference.
        """
        # Always add review/text based features
        if "review_content" in df.columns:
            df["review_length"] = df["review_content"].apply(lambda x: len(str(x)))
            df["review_word_count"] = df["review_content"].apply(
                lambda x: len(str(x).split())
            )
        if "review_title" in df.columns:
            df["review_title_length"] = df["review_title"].apply(lambda x: len(str(x)))
        if "about_product" in df.columns:
            df["about_product_length"] = df["about_product"].apply(
                lambda x: len(str(x))
            )
        # basic interaction features
        if "rating" in df.columns and "rating_count" in df.columns:
            # average rating impact per count-like feature
            df["rating_x_count"] = df["rating"] * (df["rating_count"].astype(float) + 1)
        return df

    def _get_numeric_columns(self, df: pd.DataFrame) -> list:
        num_cols = [
            "actual_price",
            "rating",
            "rating_count",
            "review_length",
            "discount_percentage",
        ]
        # include any frequency columns we added
        for col in ["product_id", "user_id"]:
            if f"{col}_freq" in df.columns and f"{col}_freq" not in num_cols:
                num_cols.append(f"{col}_freq")
        # additional derived numeric columns
        for extra in [
            "review_word_count",
            "review_title_length",
            "about_product_length",
            "rating_x_count",
        ]:
            if extra in df.columns and extra not in num_cols:
                num_cols.append(extra)
        return [col for col in num_cols if col in df.columns]

    def _to_features_and_target(
        self, df_: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        df = df_.copy()

        X = df.drop(columns=[self.target_col], errors="ignore")
        y = df[self.target_col] if self.target_col in df.columns else None
        X_arr = np.asarray(X.values)
        y_arr = np.asarray(y.values) if y is not None else np.array([])
        return X_arr, y_arr
