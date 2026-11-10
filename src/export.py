"""Модуль експорту результатів моделювання, прогнозів, метрик та навчених моделей (joblib)."""
from __future__ import annotations

import io
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd


def export_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """Конвертує DataFrame у байти CSV у кодуванні UTF-8 із BOM (для коректного відкриття в Excel).

    Args:
        df: Вхідна таблиця.

    Returns:
        Байтовий масив CSV.
    """
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def export_test_predictions_to_csv(
    y_test: pd.Series,
    y_pred: np.ndarray,
    target_name: str = "target",
    X_test: Optional[pd.DataFrame] = None,
) -> bytes:
    """Формує CSV-файл із фактичними та прогнозованими значеннями на тестовій вибірці.

    Args:
        y_test: Фактичні значення цілі на тесті.
        y_pred: Прогнози моделі.
        target_name: Назва цільової колонки.
        X_test: Опціональні тестові ознаки для включення до файлу.

    Returns:
        Байтовий масив CSV.
    """
    res_df = pd.DataFrame(
        {
            f"Факт ({target_name})": y_test.values,
            f"Прогноз ({target_name})": y_pred,
        },
        index=y_test.index,
    )

    if pd.api.types.is_numeric_dtype(y_test):
        res_df["Абсолютна помилка |Факт - Прогноз|"] = np.abs(y_test.values - y_pred)

    if X_test is not None:
        full_df = pd.concat([X_test, res_df], axis=1)
    else:
        full_df = res_df

    return export_dataframe_to_csv(full_df)


def export_metrics_summary_to_csv(metrics_df: pd.DataFrame) -> bytes:
    """Експортує таблицю метрик або звіт класифікації у CSV."""
    return export_dataframe_to_csv(metrics_df)


def export_model_to_joblib(pipeline: Any) -> bytes:
    """Серіалізує навчений Pipeline у бінарний потік за допомогою joblib.

    Args:
        pipeline: Об'єкт sklearn Pipeline / Estimator.

    Returns:
        Байтовий масив .joblib файлу.
    """
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def load_model_from_joblib(model_bytes: bytes) -> Any:
    """Десеріалізує модель із байтового масиву joblib.

    Args:
        model_bytes: Байти файлу моделі.

    Returns:
        Завантажений об'єкт Pipeline / Estimator.
    """
    buffer = io.BytesIO(model_bytes)
    return joblib.load(buffer)
