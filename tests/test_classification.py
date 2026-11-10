"""Тести для модуля classification."""
import numpy as np
import pandas as pd
import pytest

from src.classification import (
    compare_classification_models,
    create_classifier_pipeline,
    evaluate_classifier_model,
)
from src.preprocessing import build_preprocessor, split_data


@pytest.fixture
def sample_classification_data():
    """Синтетичний набір даних для тестів класифікації."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame(
        {
            "num_feat": np.random.randn(n),
            "cat_feat": np.random.choice(["Київ", "Львів", "Одеса"], size=n),
            "target": np.random.choice(["Клас_А", "Клас_Б"], size=n, p=[0.7, 0.3]),
        }
    )


def test_classification_training_and_metrics(sample_classification_data):
    """Перевірка навчання та розрахунку метрик для LogisticRegression та RandomForest."""
    df = sample_classification_data
    X_train, X_test, y_train, y_test, _ = split_data(
        df=df,
        feature_cols=["num_feat", "cat_feat"],
        target_col="target",
        test_size=0.25,
        task_type="classification",
        random_state=42,
    )

    preprocessor = build_preprocessor(
        numeric_features=["num_feat"],
        categorical_features=["cat_feat"],
    )

    # 1. Логістична регресія
    lr_pipe = create_classifier_pipeline(preprocessor, model_type="logistic_regression", use_balanced_weights=True)
    res_lr = evaluate_classifier_model(
        pipeline=lr_pipe,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="logistic_regression",
    )
    assert 0.0 <= res_lr.metrics.accuracy <= 1.0
    assert 0.0 <= res_lr.metrics.f1_macro <= 1.0
    assert res_lr.metrics.confusion_matrix.shape == (2, 2)
    assert "precision" in res_lr.metrics.report_df.columns

    # 2. Випадковий ліс
    rf_pipe = create_classifier_pipeline(preprocessor, model_type="random_forest", use_balanced_weights=True)
    res_rf = evaluate_classifier_model(
        pipeline=rf_pipe,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="random_forest",
    )
    assert 0.0 <= res_rf.metrics.accuracy <= 1.0

    # 3. Порівняння моделей
    comp_df = compare_classification_models({"LR": res_lr, "RF": res_rf})
    assert len(comp_df) == 2
    assert "F1-Score (Macro)" in comp_df.columns
