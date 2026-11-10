"""Тести для модуля regression."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.plots import plot_regression_predictions, plot_regression_residuals
from src.preprocessing import build_preprocessor, split_data
from src.regression import (
    compare_regression_models,
    create_regression_pipeline,
    evaluate_regression_model,
)


@pytest.fixture
def sample_regression_data():
    """Синтетичний набір даних для регресії."""
    np.random.seed(42)
    n = 60
    area = np.random.uniform(30, 120, size=n)
    rooms = np.random.choice([1, 2, 3, 4], size=n)
    district = np.random.choice(["Печерськ", "Оболонь", "Поділ"], size=n)
    noise = np.random.normal(0, 5000, size=n)
    price = area * 1000 + rooms * 5000 + noise

    return pd.DataFrame(
        {
            "area": area,
            "rooms": rooms,
            "district": district,
            "price": price,
        }
    )


def test_regression_training_and_metrics(sample_regression_data):
    """Перевірка навчання Ridge та RandomForestRegressor та розрахунку MAE, RMSE, R²."""
    df = sample_regression_data
    X_train, X_test, y_train, y_test, _ = split_data(
        df=df,
        feature_cols=["area", "rooms", "district"],
        target_col="price",
        test_size=0.25,
        task_type="regression",
        random_state=42,
    )

    preprocessor = build_preprocessor(
        numeric_features=["area", "rooms"],
        categorical_features=["district"],
    )

    # 1. Ridge
    ridge_pipe = create_regression_pipeline(preprocessor, model_type="ridge", alpha=1.0)
    res_ridge = evaluate_regression_model(
        pipeline=ridge_pipe,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="ridge",
        target_name="price",
    )

    assert res_ridge.metrics.mae >= 0
    assert res_ridge.metrics.rmse >= 0
    assert -10.0 <= res_ridge.metrics.r2 <= 1.0
    assert len(res_ridge.y_pred) == len(y_test)

    # 2. RandomForestRegressor
    rf_pipe = create_regression_pipeline(preprocessor, model_type="random_forest")
    res_rf = evaluate_regression_model(
        pipeline=rf_pipe,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="random_forest",
        target_name="price",
    )

    assert res_rf.metrics.mae >= 0
    assert res_rf.metrics.rmse >= 0

    # 3. Порівняння моделей
    comp_df = compare_regression_models({"Ridge": res_ridge, "RF": res_rf})
    assert len(comp_df) == 2
    assert "MAE (Середня абс. помилка)" in comp_df.columns
    assert "R² (Якість апроксимації)" in comp_df.columns


def test_regression_plots():
    """Перевірка генерації графіків регресії."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, 28.0, 42.0])

    fig_pvf = plot_regression_predictions(y_true, y_pred, title="Test Pred vs Fact", target_name="Ціна")
    assert fig_pvf is not None
    plt.close(fig_pvf)

    fig_res = plot_regression_residuals(y_true, y_pred, title="Test Residuals")
    assert fig_res is not None
    plt.close(fig_res)
