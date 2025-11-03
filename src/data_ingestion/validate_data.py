import logging
from typing import List
import pandas as pd


class DataValidatorBase:
    def __init__(self, columns: List[str], target_col: str) -> None:
        """Initialize the Data Validator.
        Args:
            columns: list of expected feature column names.
            target_col: name of the target column.
        """
        self.columns = columns
        self.target_col = target_col
        self.logger = logging.getLogger(__class__.__name__)

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate the DataFrame.
        Args:
            df (pd.DataFrame): DataFrame to validate.
        Returns:
            bool: True if valid, False otherwise.
        """
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input data is not a pandas DataFrame.")
            raise ValueError("Input data is not a pandas DataFrame.")

        if df.empty:
            self.logger.error(
                "Input DataFrame is empty. At least one row is required to continue."
            )
            return False

        cols_in_data = df.columns.tolist()
        missing_cols = [col for col in self.columns if col not in cols_in_data]
        if missing_cols:
            self.logger.error(f"Missing columns in data: {missing_cols}")
            return False

        if self.target_col not in cols_in_data:
            self.logger.error(f"Target column '{self.target_col}' is missing in data.")
            return False

        self.logger.info("All expected columns are present in the data.")
        return True
