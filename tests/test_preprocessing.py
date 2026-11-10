"""Тести для модуля preprocessing (конвеєри без витоку даних та поділ даних)."""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import build_preprocessor, split_data


def test_build_preprocessor_with_missing_values():
    """Перевірка ColumnTransformer на числових і категоріальних даних з пропусками."""
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "num2": [10.0, np.nan, 30.0, 40.0, 50.0],
            "cat1": ["A", "B", np.nan, "A", "C"],
        }
    )
    preprocessor = build_preprocessor(
        numeric_features=["num1", "num2"],
        categorical_features=["cat1"],
        scale_numeric=True,
    )
    # Навчаємо трансформер на df
    transformed = preprocessor.fit_transform(df)
    assert transformed is not None
    # Перевірка відсутності NaN після трансформації
    assert not np.isnan(transformed).any()


def test_split_data_stratified():
    """Перевірка випадкового стратифікованого поділу."""
    df = pd.DataFrame(
        {
            "feat1": range(100),
            "target": ["class1"] * 70 + ["class2"] * 30,
        }
    )
    X_train, X_test, y_train, y_test, desc = split_data(
        df=df,
        feature_cols=["feat1"],
        target_col="target",
        test_size=0.2,
        task_type="classification",
        random_state=42,
    )
    assert len(X_train) == 80
    assert len(X_test) == 20
    # Перевірка збереження пропорції класів у тестовій вибірці
    assert y_test.value_counts()["class1"] == 14
    assert y_test.value_counts()["class2"] == 6
    assert "стратифікований" in desc


def test_split_data_chronological():
    """Перевірка хронологічного поділу за датою (минуле -> майбутнє без shuffle)."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "feat": range(10),
            "target": [0, 1] * 5,
            "record_date": dates[::-1],  # свідомо передаємо у зворотному порядку
        }
    )
    X_train, X_test, y_train, y_test, desc = split_data(
        df=df,
        feature_cols=["feat"],
        target_col="target",
        test_size=0.3,
        date_col="record_date",
        task_type="classification",
    )
    assert len(X_train) == 7
    assert len(X_test) == 3
    assert "Хронологічний поділ" in desc
