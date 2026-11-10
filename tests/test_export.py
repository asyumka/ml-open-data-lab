"""Тести для модуля export (експорт результатів у CSV та серіалізація моделей у joblib)."""
import io
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.export import (
    export_dataframe_to_csv,
    export_metrics_summary_to_csv,
    export_model_to_joblib,
    export_test_predictions_to_csv,
    load_model_from_joblib,
)


def test_export_dataframe_to_csv():
    """Перевірка експорту датафрейму у байти CSV."""
    df = pd.DataFrame({"місто": ["Київ", "Львів"], "індекс": [100, 200]})
    csv_bytes = export_dataframe_to_csv(df)
    assert isinstance(csv_bytes, bytes)
    # Зворотне зчитування для перевірки цілісності
    recovered_df = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig")
    assert recovered_df.shape == (2, 2)
    assert list(recovered_df.columns) == ["місто", "індекс"]


def test_export_test_predictions_to_csv():
    """Перевірка експорту прогнозів із розрахунком помилки."""
    y_test = pd.Series([100.0, 200.0, 300.0], name="price")
    y_pred = np.array([105.0, 190.0, 310.0])
    X_test = pd.DataFrame({"area": [50, 70, 90]})

    csv_bytes = export_test_predictions_to_csv(
        y_test=y_test,
        y_pred=y_pred,
        target_name="price",
        X_test=X_test,
    )
    df_pred = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig")
    assert "Факт (price)" in df_pred.columns
    assert "Прогноз (price)" in df_pred.columns
    assert "Абсолютна помилка |Факт - Прогноз|" in df_pred.columns
    assert "area" in df_pred.columns
    assert df_pred.shape == (3, 4)


def test_model_joblib_serialization_and_loading():
    """Перевірка збереження та завантаження навченого Pipeline через joblib."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    pipe.fit(X, y)

    # Серіалізація
    model_bytes = export_model_to_joblib(pipe)
    assert isinstance(model_bytes, bytes)
    assert len(model_bytes) > 0

    # Завантаження
    loaded_pipe = load_model_from_joblib(model_bytes)
    test_X = np.array([[6.0], [7.0]])
    pred_orig = pipe.predict(test_X)
    pred_loaded = loaded_pipe.predict(test_X)

    np.testing.assert_allclose(pred_orig, pred_loaded)
