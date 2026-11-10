"""Комплексні тести для обробки 'брудних' та граничних даних (пропуски, один клас, текст у числі, <30 рядків)."""
import io
import numpy as np
import pandas as pd
import pytest

from src.classification import create_classifier_pipeline, evaluate_classifier_model
from src.data_loader import analyze_dataset, load_dataset
from src.preprocessing import build_preprocessor, split_data
from src.validation import (
    validate_classification_target,
    validate_clean_rows_count,
    validate_dataframe,
    validate_features_and_target,
    validate_regression_target,
)


def test_dirty_data_text_in_numeric_column_for_regression():
    """Тест: текстові значення у стовпці, що обраний як ціль для регресії."""
    dirty_series = pd.Series(["100", "200", "невідомо", "текст замість числа", "NaN"])
    is_valid, errors, _ = validate_regression_target(dirty_series)
    assert not is_valid
    assert any("не є числовою" in err for err in errors)


def test_dirty_data_single_class_or_rare_class():
    """Тест: один клас або надто рідкісний клас у класифікації."""
    # Тільки 1 клас
    single_class_y = pd.Series(["успіх", "успіх", "успіх"])
    valid1, errs1, _ = validate_classification_target(single_class_y)
    assert not valid1
    assert any("лише один унікальний клас" in e for e in errs1)

    # Рідкісний клас (1 зразок при необхідному мінімумі 2)
    rare_y = pd.Series(["A"] * 20 + ["B"])
    valid2, errs2, _ = validate_classification_target(rare_y, min_samples_per_class=2)
    assert not valid2
    assert any("надто малою кількістю зразків" in e for e in errs2)


def test_dirty_data_few_rows_warning():
    """Тест: замало рядків після очищення або у датасеті (< 30)."""
    valid, err = validate_clean_rows_count(15, min_required=30)
    assert not valid
    assert "Замало рядків" in err

    df_small = pd.DataFrame({"x": range(10), "y": range(10)})
    valid_df, msgs = validate_dataframe(df_small, min_rows=30)
    assert valid_df
    assert any("лише 10 рядків" in m for m in msgs)


def test_dirty_data_target_included_in_features():
    """Тест: цільова змінна випадково входить до списку ознак (витік даних)."""
    valid, errors = validate_features_and_target(
        features=["feat1", "feat2", "target_col"],
        target="target_col",
        require_target=True,
    )
    assert not valid
    assert any("не повинна входити до списку ознак" in e for e in errors)


def test_dirty_data_all_nan_column():
    """Тест: стовпець повністю складається з NaN."""
    df_all_nan = pd.DataFrame(
        {
            "good_feat": [1, 2, 3, 4, 5],
            "dead_col": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    valid, msgs = validate_dataframe(df_all_nan)
    assert any("100% NaN" in m for m in msgs)
    assert any("dead_col" in m for m in msgs)


def test_pipeline_robustness_with_heavy_missing_data():
    """Тест: робота конвеєра при великій кількості пропусків у тренувальних та тестових даних."""
    df = pd.DataFrame(
        {
            "num1": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0] * 4,
            "cat1": ["A", "B", np.nan, "B", "A", np.nan, "B", "A", np.nan, "A"] * 4,
            "target": ["Yes", "No"] * 20,
        }
    )
    X_train, X_test, y_train, y_test, _ = split_data(
        df=df,
        feature_cols=["num1", "cat1"],
        target_col="target",
        test_size=0.25,
        random_state=42,
    )
    preprocessor = build_preprocessor(numeric_features=["num1"], categorical_features=["cat1"])
    pipeline = create_classifier_pipeline(preprocessor, model_type="logistic_regression")

    # Перевірка: навчання та передбачення мають пройти без винятків
    res = evaluate_classifier_model(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="logistic_regression",
    )
    assert res.metrics.accuracy >= 0.0
