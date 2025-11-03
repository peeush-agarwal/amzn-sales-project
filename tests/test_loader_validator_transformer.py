import pandas as pd
import pytest

from data_ingestion.load_csv import DataLoaderCsv
from data_ingestion.validate_data import DataValidatorBase
from data_ingestion.preprocess_data import DataTransformerBase


def test_data_loader_csv_reads_and_errors(tmp_path):
    # create a sample csv
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    p = tmp_path / "sample.csv"
    df.to_csv(p, index=False)

    loader = DataLoaderCsv(data_path=p)
    df2 = loader.load()
    assert isinstance(df2, pd.DataFrame)
    assert list(df2.columns) == ["a", "b"]

    # non-existent file raises
    with pytest.raises(FileNotFoundError):
        DataLoaderCsv(data_path=tmp_path / "nope.csv").load()


def test_data_validator_base_checks_columns_and_target():
    columns = ["feature1", "feature2", "feature3"]
    target_col = "target"

    v = DataValidatorBase(columns, target_col)
    # build a minimal valid DataFrame with required cols + target
    cols = columns + [target_col]
    df = pd.DataFrame([{c: "x" for c in cols}])
    assert v.validate(df) is True

    # missing expected column
    df2 = df.drop(columns=[columns[0]])
    assert v.validate(df2) is False

    # missing target
    df3 = df.drop(columns=[target_col])
    assert v.validate(df3) is False


def test_data_transformer_fit_transform_and_transform(tmp_path):
    columns = ["feature1", "feature2", "feature3"]
    target_col = "target"

    transformer = DataTransformerBase(columns=columns, target_col=target_col)
    # create a small DataFrame with expected fields
    df = pd.DataFrame(
        [
            {
                "feature1": "p1",
                "feature2": "n1",
                "feature3": "f1",
                "target": 10,
            }
        ]
    )

    X, y = transformer.fit_transform(df)
    assert X.shape[0] == 1
    assert y.shape[0] == 1

    # Transform a new dataframe (unseen product_id/user)
    df2 = df.copy()
    X2, y2 = transformer.transform(df2)
    assert X2.shape[0] == 1
