import logging
from pathlib import Path

import pandas as pd


class DataLoaderCsv:
    def __init__(self, data_path: Path):
        """Initialize the CSV data loader.
        Args:
            data_path (Path): Path to the CSV file.
        """
        self.data_path = data_path
        self.logger = logging.getLogger(__class__.__name__)

    def load(self) -> pd.DataFrame:
        """Load data from the specified CSV file.

        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the specified path is not a file or not a CSV file.
        """
        if not self.data_path.exists():
            self.logger.error(f"Data path {self.data_path} does not exist.")
            raise FileNotFoundError(f"Data path {self.data_path} does not exist.")
        if not self.data_path.is_file():
            self.logger.error(f"Data path {self.data_path} is not a file.")
            raise ValueError(f"Data path {self.data_path} is not a file.")
        if self.data_path.suffix != ".csv":
            self.logger.error(f"Data path {self.data_path} is not a CSV file.")
            raise ValueError(f"Data path {self.data_path} is not a CSV file.")

        df = pd.read_csv(self.data_path)
        self.logger.info(f"Data loaded from {self.data_path} with shape {df.shape}.")
        self.logger.info(f"Columns: {df.columns.tolist()}")
        return df
